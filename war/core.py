import logging

from war.database import Database
from war.task import Task
from war.cformat import ColorFormat


class Strategy:

    def __init__(self, name=None, max_parallel_tasks=1,
                 max_threads_per_estimator=1, max_tasks=-1,
                 weight=0.01, warm_up=20):
        self.name = name if name else self.__class__.__name__
        self.max_parallel_tasks = max_parallel_tasks
        self.max_threads_per_estimator = max_threads_per_estimator
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
            'best': dict(agg=dict(avg=0, std=0, min=0, max=0), scores=list()),
            'finished': 0,
        }
        self.load_cache()

    def load_cache(self):
        logger = logging.getLogger('war.strategy')
        logger.debug(ColorFormat('%s: loading cache').dark_gray, self.name)
        best = dict(agg=dict(avg=0, std=0, min=0, max=0), scores=list())
        cumulative_time = 0
        count = 0
        for _, result in self.database.iterate():
            if result['type'] != 'result':
                continue
            count += 1
            result = result['data']
            if result['status'] != 'OK':
                continue
            if not best or best['agg']['avg'] < result['agg']['avg']:
                best = result
            cumulative_time += result['elapsed_time']
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
