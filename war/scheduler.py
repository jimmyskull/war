"""Task scheduler based on strategy performance."""
import logging
import multiprocessing
import time

from numpy import array, bincount, ceil
from numpy.random import choice
import numpy
import psutil

from war.cformat import ColorFormat as CF
from war.optimize import optimize_slots_config


class Scheduler:

    def __init__(self, strategies, nconsumers, max_slots_per_evaluation,
                 cooperate):
        self.logger = logging.getLogger('war.scheduler')
        self.strategies = dict()
        self.nconsumers = nconsumers
        self.max_slots = nconsumers
        self.max_slots_per_evaluation = max_slots_per_evaluation
        self.slots_running = 0
        self._populate(strategies)
        self.tasks_finished = 0
        self.report_at_ntasks = 100
        self.improved_since_last_report = False
        self.last_error = None
        self._cooperate = cooperate
        self.proc = psutil.Process()
        self.cpu_count = multiprocessing.cpu_count()
        self.last_coop_time = time.time()
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
                'tasks_since_last_improvement':
                    strat.cache['tasks_since_last_improvement'],
                'time_since_last_improvement':
                    strat.cache['time_since_last_improvement'],
                'finished': strat.cache['finished'],
                'running': 0,
                'slots': 0,
                'exhausted': False
            }

    @property
    def cooperation_mode(self):
        return self._cooperate

    @cooperation_mode.setter
    def cooperation_mode(self, value):
        self._cooperate = value
        if value:
            self.last_coop_time = time.time()
            self._init_proc()

    def set_max_slots(self, value):
        if self.max_slots == value:
            return
        if value > self.nconsumers:
            raise ValueError('Maximum number of slots must be up to {}'.format(
                self.nconsumers))
        curr = self.max_slots
        self.max_slots = value
        self.logger.info(CF('Changed number of slots from %d to %d').yellow,
                         curr, self.max_slots)
        excess = self.slots_running - self.max_slots
        if excess > 0:
            self.logger.info(CF('There are %d slots above the current limit. '
                                'Waiting for end of normal execution.').yellow,
                              excess)

    def collect(self, result):
        self.slots_running -= result.jobs
        assert self.slots_running >= 0
        for strat in self.strategies:
            if hash(strat) != result.task.strategy_id:
                continue
            self.tasks_finished += 1
            info = self.strategies[strat]
            info['running'] -= 1
            info['slots'] -= result.jobs
            info['cumulative_time'] += result.elapsed_time
            info['tasks_since_last_improvement'] += 1
            info['time_since_last_improvement'] += result.elapsed_time
            info['finished'] += 1
            if result.status == 'FAILED':
                self.logger.error('%s task failed: %s', strat.name,
                                  result.error_info['message'])
                self.logger.error('Task id: %s', result.task.full_id())
                self.last_error = result
                return
            score = result.agg['avg']
            if self.strategies[strat]['best']['agg']['avg'] < score:
                self.logger.info(
                    (str(CF('%s').bold.green) +
                     str(CF(' improvement: %.4f -> %.4f').green)),
                    strat.name,
                    self.strategies[strat]['best']['agg']['avg'],
                    score
                )
                self.strategies[strat]['best'] = {
                    'agg': result.agg,
                    'scores': result.scores,
                }
                self.strategies[strat]['tasks_since_last_improvement'] = 0
                self.strategies[strat]['time_since_last_improvement'] = 0
                self.improved_since_last_report = True
            assert self.strategies[strat]['running'] >= 0
            return
        raise ValueError('Strategy not found not mark task as finished.')

    def strategy_by_id(self, idx):
        return list(self.strategies.keys())[idx - 1]

    def set_weight(self, idx, weight):
        strategy = self.strategy_by_id(idx)
        curr = strategy.weight
        strategy.weight = weight
        self.logger.info('Strategy %s weight changed from %.4f to %.4f.',
                         strategy.name, curr, strategy.weight)

    def available_slots(self):
        return max(0, self.max_slots - self.slots_running)

    def _get_validation_bounds(self):
        assert self.max_slots_per_evaluation > 0
        return (1, self.max_slots_per_evaluation)

    def _make_tasks(self, slots, strat):
        # Get maximum of parallelization on an estimator's fit.
        try:
            config = optimize_slots_config(
                available_slots=slots,
                tasks_bounds=strat.get_tasks_bounds(self.nconsumers),
                validation_bounds=self._get_validation_bounds(),
                fit_bounds=strat.get_fit_bounds(self.nconsumers))
        except RuntimeError:
            # Could not find a single configuration to satisfy.
            return list()
        allocated_slots_per_task = config['njobs_on_validation'] * \
                                   config['njobs_on_estimator']
        created = 0
        task_list = list()
        for _ in range(config['tasks']):
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
                self.logger.info(CF('%s is exhausted').bold.bottle_green,
                                 strat.name)
                break
            except Exception as err:  # pylint: disable=W0703
                self.logger.error('Failed to create a task for %s: %s',
                                  strat.name,
                                  '{}: {}'.format(type(err).__name__, err))
        if created:
            self.logger.info(CF('New %d × %s cv=%d fit=%d').dark_gray,
                             created, strat.name,
                             config['njobs_on_estimator'],
                             config['njobs_on_validation'])
        return task_list

    def next(self):
        self.cooperate()
        # Estimate available slots (CPU cores to use).
        available_slots = self.available_slots()
        if not available_slots:
            return []
        self.logger.debug(CF('We have %d slots to use').light_gray,
                          available_slots)
        # Sample from discrete probability function.
        probs = self.probabilities()
        sample = choice(len(self.strategies), size=available_slots, p=probs)
        selected = bincount(sample)
        # Get maximum of parallelization on a (cross-)valitation's fit.
        task_list = list()
        # Generate tasks
        for slots, strat in zip(selected, self.strategies):
            if slots:
                task_list += self._make_tasks(slots, strat)
        return task_list

    def _get_scores(self):
        # Get weighted scores for probability calculation.
        scores = list()
        for strat, info in self.strategies.items():
            if not info['exhausted']:
                score = info['best']['agg']['avg'] * strat.weight
                scores.append(score)
        return numpy.array(scores)

    def probabilities(self):
        weights = list()
        scores = self._get_scores()
        min_score = max(0, min(scores))
        max_score = min(1, max(scores)) + 1e-6
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

    def _average_worker_cpu_usage(self):
        perc_expected = self.slots_running / self.cpu_count
        ratios = list()
        for child in self.proc_children:
            perc_usage = child.cpu_percent() / 100
            ratio = perc_usage / (perc_expected + 1e-6)
            self.logger.debug(CF('CPU Usage: %5.1f%%').light_gray, 100 * ratio)
            if ratio > 0:
                ratios.append(ratio)
        if not ratios:
            return (0, 0)
        active = max(self.slots_running, len(ratios))
        nactive = len(ratios)
        ratio = numpy.sum(ratios) / (active + 1e-6)
        self.logger.info(
            CF('%d active workers, %d slots, average CPU usage: %.0f%%').cyan,
            nactive, self.slots_running, ratio * 100)
        return ratio

    def report_worker_usage(self):
        self._average_worker_cpu_usage()

    def cooperate(self, force=False):
        if not force and (time.time() - self.last_coop_time) < 60:
            return
        self.last_coop_time = time.time()
        logger = self.logger
        ratio = self._average_worker_cpu_usage()
        if ratio == 0:
            return
        if self.slots_running > self.max_slots:
            logger.info(CF(('There are %d slots running above current '
                            'limit %d. Waiting them to finish.')).cyan,
                        self.slots_running - self.max_slots, self.max_slots)
            return
        if ratio < 0.95 and self.max_slots > max(2, self.nconsumers // 2):
            ideal = ceil(2 * self.max_slots - self.max_slots / ratio)
            max_slots = int(max(ideal, 2))
            if max_slots != self.max_slots:
                # It's possible the reduction will not happen when
                # working if few slots.
                logger.warning(('Average worker CPU usage is at %.0f%%, '
                                'decreasing slots from %d to %d.'),
                               ratio * 100, self.max_slots, max_slots)
                self.max_slots = max_slots
        elif ratio > 1.10 and self.max_slots < self.nconsumers:
            max_slots = self.max_slots + 1
            logger.warning(('It seems we can use more CPU. '
                            'Increasing slots from %d to %d.'),
                           self.max_slots, max_slots)
            self.max_slots = max_slots
