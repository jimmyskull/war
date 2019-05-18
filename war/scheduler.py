from itertools import product
from math import ceil
import logging
import multiprocessing
import psutil
import time

from numpy import array, argsort, bincount, ceil, floor, prod, stack
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
    return ('%dd, ' + pattern) % (d, h, m, s)


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
            (1, max_validation_njobs),
            (1, max_estimator_njobs)))
    fx, cx = floor(opt_res.x), ceil(opt_res.x)
    borders = stack([fx - 1, fx, cx, cx + 1])

    best_p, best = None, 0
    for on_task, on_cv, on_est in product(*borders.transpose().tolist()):
        total = on_task * on_cv * on_est
        if (on_task > max_parallel_tasks
            or on_cv > max_validation_njobs
            or on_est > max_estimator_njobs):
            continue
        if total == 0 or total > available_slots or total < best:
            continue
        best_p = dict(
            tasks=int(on_task),
            njobs_on_validation=int(on_cv),
            njobs_on_estimator=int(on_est))
        best = total

    if not best_p:
        raise ValueError('Could not solve optimization problem')
    return best_p


class Scheduler:

    def __init__(self, strategies, nconsumers, max_threads_per_evaluation,
                 cooperate):
        self.strategies = dict()
        self.nconsumers = nconsumers
        self.max_slots = nconsumers
        self.max_threads_per_evaluation = max_threads_per_evaluation
        self.slots_running = 0
        self._populate(strategies)
        self.tasks_finished = 0
        self.report_at_ntasks = 100
        self.improved_since_last_report = False
        self.last_error = None
        self._cooperate = cooperate
        self.proc = psutil.Process()
        self.cpu_count = multiprocessing.cpu_count()
        self.last_coop_time = None
        self.proc_children = list(self.proc.children(recursive=True))
        self._init_proc()

    def _init_proc(self):
        self.proc.cpu_percent()
        for child in self.proc_children:
            child.cpu_percent()

    def _populate(self, strategy_list):
        for strat in strategy_list:
            self.strategies[strat] = {
                'cumulative_time': strat.cache['cumulative_time'],
                'best': strat.cache['best'],
                'finished': strat.cache['finished'],
                'running': 0,
                'slots': 0,
                'exhausted': False
            }

    def set_max_slots(self, value):
        if self.max_slots == value:
            return
        if value > self.nconsumers:
            raise ValueError('Maximum number of slots must be up to {}'.format(
                self.nconsumers))
        curr = self.max_slots
        self.max_slots = value
        logger = logging.getLogger('war.scheduler')
        logger.info(
            ColorFormat('Changed number of slots from %d to %d').yellow,
            curr, self.max_slots)
        excess = self.slots_running - self.max_slots
        if excess > 0:
            logger.info(
                ColorFormat(
                    'There are %d slots above the current limit. '
                    'Waiting for end of normal execution.').yellow,
                excess )

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
            self.strategies[strat]['slots'] -= result.jobs
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

    def report_counters(self):
        logger = logging.getLogger('war.scheduler')
        header = '-' * 80
        logger.info(ColorFormat('Scheduler Counters').bold)
        logger.info(header)
        logger.info('Scheduler thread CPU usage : %.f%%',
            self.proc.cpu_percent())
        logger.info('CPU count                  : %d', self.cpu_count)
        logger.info('Number of consumers        : %d', self.nconsumers)
        logger.info('Max. number of slots       : %d', self.max_slots)
        logger.info('Max. threads in validation : %d',
            self.max_threads_per_evaluation)
        logger.info('Slots running              : %d', self.slots_running)
        logger.info('Tasks ended in this session: %d',
            self.tasks_finished)
        logger.info('Cooperate                  : %s', self._cooperate)
        logger.info('Cooperation resting        : %ds, since %s',
            time.time() - self.last_coop_time,
            time.strftime('%c', time.localtime(self.last_coop_time)))
        logger.info(header)

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

    def strategy_by_id(self, idx):
        return list(self.strategies.keys())[idx - 1]

    def set_weight(self, idx, weight):
        strategy = self.strategy_by_id(idx)
        curr = strategy.weight
        strategy.weight = weight
        logger = logging.getLogger('war.scheduler')
        logger.info(
            'Strategy %s weight changed from %.4f to %.4f.',
            strategy.name,
            curr, strategy.weight)

    def report_best(self, idx):
        logger = logging.getLogger('war.scheduler')
        if idx > len(self.strategies):
            logger.error('No strategy was found at index %d', idx)
            return
        import pprint
        from pygments import highlight
        from pygments.formatters import TerminalFormatter
        from pygments.lexers import PythonLexer
        pp = pprint.PrettyPrinter()
        strategy = self.strategy_by_id(idx)
        print(ColorFormat(strategy.name).bold)
        code = pp.pformat(self.strategies[strategy])
        fmt = highlight(code, PythonLexer(), TerminalFormatter())
        print(fmt)

    def report_results(self):
        self.improved_since_last_report = False
        logger = logging.getLogger('war.scheduler')


        from war.table import TerminalTable

        table = TerminalTable()
        table.set_header([
            'ID', '↓Rank', 'Name', 'Total Time', 'T', 'S', 'Ended',
            'Best', '95% CI', 'Min', 'Max', 'Prob', 'Weight'
        ])

        def _count(x):
            return f'{x:,d}'

        def _score(x):
            if x < 0:
                return f'{x:.3f}'
            return f'{x:.4f}'

        def _weight(x):
            if x < 0:
                return f'{x:.1f}'
            return f'{x:.2f}'

        scores = [info['best']['agg']['avg']
                  for info in self.strategies.values()]
        rank_score = argsort(-array(scores))
        sorted_scores = sorted(scores)

        total_time = sum([info['cumulative_time']
                          for info in self.strategies.values()]) + 1e-4
        probs = self._probs()

        from war.table import Cell

        rows = list()

        for idx, (strat, info) in enumerate(self.strategies.items(), 1):
            agg = info['best']['agg']
            to_ci = 2.0 / (numpy.sqrt(len(info['best']['scores'])) + 1e-4)
            attr = None
            if agg['avg'] == sorted_scores[-1]:
                attr = ['bold', 'blue']
            rank = numpy.where(rank_score == idx - 1)[0][0] + 1
            rows.append(
                (rank,
                 [
                    str(idx),
                    str(rank),
                    Cell(strat.name, attr=['ljust']),
                    Cell(('{} ({:5.1%})'.format(
                          sec2time(info['cumulative_time'], 0),
                          info['cumulative_time'] / total_time)),
                         attr=['rjust']
                     ),
                    _count(info['running']),
                    _count(info['slots']),
                    _count(info['finished']),
                    _score(agg['avg']),
                    _score(agg['std'] * to_ci),
                    _score(agg['min']),
                    _score(agg['max']),
                    f'{probs[idx - 1]:.0%}',
                    _weight(strat.weight),
                 ],
                 attr)
            )

        rows = sorted(rows, key=lambda x: x[0])
        for row in rows:
            table.add_row(row[1], attr=row[2])

        # import sys
        # sys.stderr.write('\033c\033[3J')
        # sys.stderr.flush()
        for table_row in table.format().split('\n'):
            logger.info(table_row)

    def available_slots(self):
        return max(0, self.max_slots - self.slots_running)

    def next(self):
        #assert self.slots_running <= self.nconsumers
        logger = logging.getLogger('war.scheduler')

        if self.last_coop_time is None:
            self.last_coop_time = time.time()

        if self._cooperate:
            self.cooperate()

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
            allocated_slots_per_task = config['njobs_on_validation'] * \
                                       config['njobs_on_estimator']
            allocated_slots = config['tasks'] * allocated_slots_per_task
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
                    task.total_jobs = allocated_slots_per_task
                    self.strategies[strat]['running'] += 1
                    self.strategies[strat]['slots'] += allocated_slots_per_task
                    created += 1
                    self.slots_running += allocated_slots_per_task
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

    def toggle_cooperate(self):
        logger = logging.getLogger('war.scheduler')
        if self._cooperate:
            self._cooperate = False
            logger.info(
                ColorFormat('Cooperation has been disabled.').cyan.bold)
            if self.max_slots < self.nconsumers:
                logger.info(
                    ColorFormat('Increasing slots from %d to %d.').cyan,
                    self.max_slots, self.nconsumers)
                self.nconsumers = self.max_slots
        else:
            self._cooperate = True
            logger.info(
                ColorFormat('Cooperation has been enabled.').cyan.bold)
            logger.info(
                ColorFormat('The current number of slots is %d.').cyan,
                self.max_slots)
            logger.info(
                ColorFormat('Collecting information for analysis.').cyan)
            self.last_coop_time = time.time()
            self._init_proc()

    def _probs(self):
        weights = list()
        min_score = min(max(0, info['best']['agg']['avg'] * strat.weight)
                        for strat, info in self.strategies.items()
                        if not info['exhausted'])
        max_score = max(max(1, info['best']['agg']['avg'] * strat.weight)
                        for strat, info in self.strategies.items()
                        if not info['exhausted'])
        for strat, info in self.strategies.items():
            max_tasks = strat.max_tasks
            exhausted, finished = info['exhausted'], info['finished']
            if not (max_tasks == -1 or max_tasks > finished) or exhausted:
                weights.append(0)
                continue
            best_avg_score = info['best']['agg']['avg'] * strat.weight
            best_score = min(1, max(0, best_avg_score))
            norm_score = (best_score - min_score) / (max_score - min_score)
            warm_up = 2 * (strat.warm_up - info['finished'])
            weight = max(0, max(norm_score + 1e-6, warm_up))
            weights.append(weight)
        weights = array(weights) + 1e-6
        probs = weights / sum(weights)
        return probs

    def _averate_worker_cpu_usage(self):
        logger = logging.getLogger('war.scheduler')
        perc_expected = self.slots_running / self.cpu_count
        ratios = list()
        for child in self.proc_children:
            perc_usage = child.cpu_percent() / 100
            ratio = perc_usage / (perc_expected + 1e-6)
            logger.debug(ColorFormat('CPU Usage: %.2f').light_gray,
                ratio)
            if ratio > 0:
                ratios.append(ratio)
        if not ratios:
            return (0, 0)
        return (len(ratios), numpy.mean(ratios))

    def report_worker_usage(self):
        logger = logging.getLogger('war.scheduler')
        nactive, ratio = self._averate_worker_cpu_usage()
        logger.info(
            ColorFormat('%d active workers, average CPU usage: %.0f%%').cyan,
            nactive, ratio * 100)

    def cooperate(self, force=False):
        if not force and (time.time() - self.last_coop_time) < 60:
            return
        self.last_coop_time = time.time()
        logger = logging.getLogger('war.scheduler')
        nactive, ratio = self._averate_worker_cpu_usage()
        logger.info(
            ColorFormat('%d active workers, average CPU usage: %.0f%%').cyan,
            nactive, ratio * 100)
        if ratio == 0:
            return
        if self.slots_running > self.max_slots:
            logger.info(
                ColorFormat(
                    ('There are %d slots running above current limit %d. '
                     'Waiting them to finish.')).cyan,
                self.slots_running - self.max_slots, self.max_slots)
            return
        if ratio < 0.95 and self.max_slots > max(2, self.nconsumers // 2):
            max_slots = int(max(ceil(self.max_slots * ratio), 2))
            if max_slots != self.max_slots:
                # It's possible the reduction will not happen when
                # working if few slots.
                logger.warning(
                    ('Average worker CPU usage is at %.0f%%, '
                     'decreasing slots from %d to %d.'),
                    ratio * 100,
                    self.max_slots, max_slots)
                self.max_slots = max_slots
        elif ratio > 1.10 and self.max_slots < self.nconsumers:
            max_slots = self.max_slots + 1
            logger.warning(
                ('It seems we can use more CPU. '
                 'Increasing slots from %d to %d.'),
                self.max_slots, max_slots)
            self.max_slots = max_slots
