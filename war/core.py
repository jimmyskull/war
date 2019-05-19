"""Core classes."""
from datetime import datetime
import logging

import numpy

from war.cformat import ColorFormat
from war.database import Database
from war.task import Task


class Strategy:
    """
    Base strategy.

    War works with several simulteaneous strategies.
    """

    def __init__(self, name=None, parallel_tasks_bounds=(1, -1),
                 parallel_fit_bounds=(1, -1), max_tasks=-1,
                 weight=1.0, warm_up=20):
        self.name = name if name else self.__class__.__name__
        self.parallel_tasks_bounds = parallel_tasks_bounds
        self.parallel_fit_bounds = parallel_fit_bounds
        self.max_tasks = max_tasks
        # Additional weight for probability scheduling.
        # May be positive and negative.
        self.weight = weight
        # Ask to run at least warm_up tasks before
        # being susceptible to be dominated.
        self.warm_up = warm_up
        # Use class name as userspace, to avoid spaces and special
        # characters in the path.
        self.database = Database(namespace=self.__class__.__name__)
        # Cache only update in load_cache().
        self.cache = {
            'cumulative_time': 0,
            'best': dict(agg=dict(avg=0, std=0, min=0, max=0),
                         scores=list(),
                         params=dict()),
            'tasks_since_last_improvement': 0,
            # This is the sum of elapsed time, _not_ the difference
            # of real times between now and the last improvement.
            'time_since_last_improvement': 0,
            'finished': 0,
        }
        self.load_cache()

    def get_tasks_bounds(self, nconsumers):
        bounds = self.parallel_tasks_bounds
        if bounds[1] > 0:
            return bounds
        return (bounds[0], nconsumers - (bounds[1] + 1))

    def get_fit_bounds(self, nconsumers):
        bounds = self.parallel_fit_bounds
        if bounds[1] > 0:
            return bounds
        assert bounds[1] != 0, 'Upper bounds must be positive or negative.'
        return (bounds[0], nconsumers - (bounds[1] + 1))

    def load_cache(self):
        logger = logging.getLogger('war.strategy')
        logger.debug(ColorFormat('%s: loading cache').dark_gray, self.name)
        best = dict(agg=dict(avg=0, std=0, min=0, max=0), scores=list())
        cumulative_time = 0
        count = 0
        history = list()
        for _, result in self.database.iterate():
            if result['type'] != 'result':
                continue
            count += 1
            result = result['data']
            if result['status'] != 'OK':
                continue
            dt = datetime.strptime(result['begin_time'], '%Y-%m-%d %H:%M:%S')
            score = result['agg']['avg']
            elapsed = result['elapsed_time']
            history.append((dt, score, elapsed))
            if not best or best['agg']['avg'] < score:
                best = result
            cumulative_time += elapsed
        history = sorted(history, key=lambda x: x[0])
        last_improvement = numpy.argmax(item[1] for item in history)
        tsli = max(0, len(history) - last_improvement - 1)
        timesli = sum(item[2] for item in history)
        self.cache['tasks_since_last_improvement'] = tsli
        self.cache['time_since_last_improvement'] = timesli
        logger.info(ColorFormat('%s: loaded %d cached results').yellow,
                    self.name, count)
        self.cache['cumulative_time'] = cumulative_time
        self.cache['best'] = best
        self.cache['finished'] += count
        return best

    def make_task(self, estimator, params):
        task = Task(
            strategy=self,
            strategy_id=hash(self),
            estimator=estimator,
            params=params
        )
        if self.database.find(task.id()):
            return None
        logger = logging.getLogger('war.strategy')
        logger.debug(ColorFormat('New task %s').dark_gray, task.full_id())
        return task

    def collect(self, result):
        self.database.store(
            oid=result.task.id(),
            obj={'type': 'result', 'data': result.data()})
