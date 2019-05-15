from itertools import product
import logging

from numpy import array, bincount, ceil, floor, prod, stack
from numpy.random import choice
import numpy
from scipy.optimize import minimize

from war.cformat import ColorFormat


def sec2time(sec, n_msec=3):
    """Convert seconds to 'D days, HH:MM:SS.FFF'."""
    # pylint: disable=C0103
    if hasattr(sec, '__len__'):
        return [sec2time(s) for s in sec]
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if n_msec > 0:
        pattern = '%%02d:%%02d:%%0%d.%df' % (n_msec + 3, n_msec)
    else:
        pattern = r'%02d:%02d:%02d'
    if d == 0:
        return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)


def optimize_task_config(available_slots, max_parallel_tasks,
                         max_validation_njobs, max_estimator_njobs):

    def _evaluate_task_config(x):
        return numpy.abs(prod(x) - available_slots)

    opt_res = minimize(
        _evaluate_task_config,
        (1, max_validation_njobs, max_estimator_njobs),
        method='SLSQP',
        bounds=(
            (1, max_parallel_tasks),
            (1, max_estimator_njobs),
            (1, max_validation_njobs)))
    fx, cx = floor(opt_res.x), ceil(opt_res.x)
    borders = stack([fx - 1, fx, cx, cx + 1])

    best_p, best = None, 0
    for on_task, on_est, on_cv in product(*borders.transpose().tolist()):
        total = on_task * on_est * on_cv
        if (on_task > max_parallel_tasks
            or on_est > max_estimator_njobs
                or on_cv > max_validation_njobs):
            continue
        if total == 0 or total > available_slots or total < best:
            continue
        best_p = dict(
            tasks=int(on_task),
            njobs_on_estimator=int(on_est),
            njobs_on_validation=int(on_cv))
        best = total

    if not best_p:
        raise ValueError('Could not solve optimization problem')
    return best_p


class Scheduler:

    def __init__(self, strategies, nconsumers, max_threads_per_evaluation):
        self.strategies = dict()
        self.nconsumers = nconsumers
        self.max_threads_per_evaluation = max_threads_per_evaluation
        self.slots_running = 0
        self._populate(strategies)
        self.tasks_finished = 0
        self.report_at_ntasks = 100
        self.improved_since_last_report = False
        self.last_error = None

    def _populate(self, strategy_list):
        for strat in strategy_list:
            self.strategies[strat] = {
                'cumulative_time': strat.cache['cumulative_time'],
                'best': strat.cache['best'],
                'finished': strat.cache['finished'],
                'running': 0,
                'exhausted': False
            }

    def collect(self, result):
        logger = logging.getLogger('war.scheduler')
        self.slots_running -= result.jobs
        assert self.slots_running >= 0
        for strat in self.strategies:
            if hash(strat) != result.task.strategy_id:
                continue
            self.tasks_finished += 1
            strat.collect(result)
            self.strategies[strat]['running'] -= 1
            self.strategies[strat]['cumulative_time'] += result.elapsed_time
            self.strategies[strat]['finished'] += 1
            if result.status == 'FAILED':
                logger.error(
                    '%s task failed: %s', strat.name,
                    result.error_info['message'])
                logger.error('Task id: %s/%s', strat.__class__.__name__,
                             result.task.id())
                self.last_error = result
                return
            score = result.agg['avg']
            if self.strategies[strat]['best']['agg']['avg'] < score:
                logger.info(
                    (str(ColorFormat('%s').bold.green) +
                     str(ColorFormat(' improvement: %.4f -> %.4f').green)),
                    strat.name,
                    self.strategies[strat]['best']['agg']['avg'],
                    score
                )
                self.strategies[strat]['best'] = {
                    'agg': result.agg,
                    'scores': result.scores,
                }
                self.improved_since_last_report = True
            if self.tasks_finished % self.report_at_ntasks == 0:
                self.report_results()
            assert self.strategies[strat]['running'] >= 0
            return
        raise ValueError('Strategy not found not mark task as finished.')

    def report_last_error(self):
        logger = logging.getLogger('war.scheduler')
        if not self.last_error:
            logger.info('No error occurred during this session.')
            return
        message = self.last_error.error_info['message']
        traceback_msg = self.last_error.error_info['traceback']
        print('\n')
        print(str(ColorFormat('Traceback:').bold))
        print(traceback_msg)
        print(str(ColorFormat('Message:').bold))
        print('\t' + message)

    def report_results(self):
        self.improved_since_last_report = False
        logger = logging.getLogger('war.scheduler')
        # We need to compute the length of the largest name.
        name_len = max([len(strat.name) for strat in self.strategies])
        # Total time used.
        total_time = sum([info['cumulative_time']
                          for info in self.strategies.values()]) + 1e-4
        # Best score.
        best = max([info['best']['agg']['avg']
                    for info in self.strategies.values()])
        # Max running.
        max_running = max([info['running']
                          for info in self.strategies.values()])
        running_len = len(str(max_running))
        # Length for cumulative time
        cumtime_len = max(
            [len(('{} ({:5.1%})'.format(sec2time(info['cumulative_time'], 0),
                                        info['cumulative_time'] / total_time)))
             for info in self.strategies.values()])
        # Build header
        header_list = list()
        size_list = list()

        def _add_header(x, size):
            header_list.append(str(ColorFormat(x.ljust(size)).bold))
            size_list.append(size)
        COLS = [
            ('ID', 2),
            ('Name', name_len),
            ('Total Time', cumtime_len),
            ('R', running_len),
            ('Ended', 5),
            ('Best', 6),
            ('95% CI', 6),
            ('Min', 6),
            ('Max', 6),
        ]
        for hinfo in COLS:
            _add_header(*hinfo)
        header = ' | '.join(header_list)
        cols = sum([size for _, size in COLS]) + 3 * (len(COLS) - 1)
        col_sep = ' | '.join(['-' * size for size in size_list])

        logger.info(ColorFormat('Overall scheduler info:').bold.magenta)
        logger.info('-' * cols)
        logger.info(header)
        logger.info(col_sep)
        for idx, (strat, info) in enumerate(self.strategies.items(), 1):
            agg = info['best']['agg']
            to_ci = 2.0 / (numpy.sqrt(len(info['best']['scores'])) + 1e-4)
            logger.info(
                '\033[%sm%s | %s | %s | %s | %s | %s | %s | %s | %s\033[0m',
                '1;34' if agg['avg'] == best else '0',
                str(idx).rjust(size_list[0]),
                strat.name.ljust(size_list[1]),
                ('{} ({:5.1%})'.format(
                    sec2time(info['cumulative_time'], 0),
                    info['cumulative_time'] / total_time)
                 ).rjust(size_list[2]),
                f"{info['running']:,d}".rjust(size_list[3]),
                f"{info['finished']:,d}".rjust(size_list[4]),
                f"{agg['avg']:.4f}".rjust(size_list[5]),
                f"{agg['std'] * to_ci:.4f}".rjust(size_list[6]),
                f"{agg['min']:.4f}".rjust(size_list[7]),
                f"{agg['max']:.4f}".rjust(size_list[8]))
        logger.info('-' * cols)
        probs = self._probs()
        logger.info(ColorFormat('Probabilites: [%s]').magenta,
                    ', '.join([f'{prob:.0%}' for prob in probs]))

    def available_slots(self):
        return self.nconsumers - self.slots_running

    def next(self):
        assert self.slots_running <= self.nconsumers
        logger = logging.getLogger('war.scheduler')
        # Estimate available slots (CPU cores to use).
        available_slots = self.available_slots()
        if not available_slots:
            return []
        logger.debug(ColorFormat('We have %d slots to use').light_gray,
                     available_slots)
        # Get best scores plus eps (to avoid division by zero)
        probs = self._probs()
        # Sample from discrete probability function.
        selected = choice(len(self.strategies), size=available_slots, p=probs)
        selected = bincount(selected)
        # logger.debug('Selected estimators: %s', ', '.join(map(str, selected)))
        # Get maximum of parallelization on a (cross-)valitation's fit.
        if self.max_threads_per_evaluation:
            max_per_val = self.max_threads_per_evaluation
        else:
            max_per_val = self.nconsumers - self.max_threads_per_evaluation
        task_list = list()
        # Generate tasks
        for slots, strat in zip(selected, self.strategies):
            if not slots:
                continue
            # Get maximum of parallelization on an estimator's fit.
            if strat.max_threads_per_estimator > 0:
                max_per_est = strat.max_threads_per_estimator
            else:
                max_per_est = self.nconsumers + strat.max_threads_per_estimator
            # Get maximum of parallelization on an estimator's fit.
            if strat.max_parallel_tasks > 0:
                max_tasks = strat.max_parallel_tasks
            else:
                max_tasks = self.nconsumers + strat.max_parallel_tasks
            config = optimize_task_config(
                available_slots=slots,
                max_parallel_tasks=max_tasks,
                max_validation_njobs=max_per_val,
                max_estimator_njobs=max_per_est)
            running = config['tasks'] * \
                config['njobs_on_validation'] * \
                config['njobs_on_estimator']
            created = 0
            for tid in range(config['tasks']):
                if (config['njobs_on_estimator'] == 0
                    or config['njobs_on_validation'] == 0):
                    continue
                try:
                    task = strat.next(nthreads=config['njobs_on_estimator'])
                    if not task:
                        raise ValueError('no task received to execute')
                    task.n_jobs = config['njobs_on_validation']
                    task.total_jobs = config['njobs_on_estimator'] * \
                        config['njobs_on_validation']
                    self.strategies[strat]['running'] += 1
                    created += 1
                    self.slots_running += 1
                    task_list.append(task)
                except StopIteration:
                    self.strategies[strat]['exhausted'] = True
                    logger.info(
                        ColorFormat('%s is exhausted').bold.bottle_green,
                        strat.name)
                    break
                except Exception as err:
                    logger.error(
                        'Failed to create a task for %s: %s',
                        strat.name,
                        '{}: {}'.format(type(err).__name__, err))
            if created:
                logger.info(
                    ColorFormat('New %d × %s cv=%d fit=%d').dark_gray,
                    created,
                    strat.name,
                    config['njobs_on_estimator'],
                    config['njobs_on_validation'])
        return task_list

    def _probs(self):
        scores = array(
            [info['best']['agg']['avg'] \
             + max(strat.sugar, 2 * (strat.warm_up - info['finished']))
             if (strat.max_tasks == -1
                 or strat.max_tasks > info['finished'])
                 and not info['exhausted']
             else 0
             for strat, info in self.strategies.items()])
        scores = scores
        probs = scores / sum(scores)
        return probs
