"""Core classes."""
from datetime import datetime, timedelta
import logging

import numpy

from war.cformat import ColorFormat as CF
from war.database import Database
from war.task import Task


class Strategy:
    """
    Base strategy.

    War works with several simulteaneous strategies.  A strategy
    resides in the same thread of the scheduler, which is also the
    same of the UI Dashboard.  It generates tasks with candidate
    estimators whenever the engine's scheduler requires.

    Parameters
    ----------
    name : str, optional, default: class' name
        The name for the strategy. This does not change a strategy
        id composition.
    parallel_tasks_bounds : tuple, default: (1, max_slots)
        (lower, upper) inclusive bounds that tell how many parallel
        candidates may the run.
    parallel_fit_bounds : tuple, default: (1, max_slots)
        (lower, upper) inclusive bounds that tell how many parallel
        jobs may be used in each fit call.  This is passed to the
        strategy's next() method, so that it constructs the estimator
        with the necessary n_jobs configuration.
    max_tasks : int, default: no limit
        A fixed limit of the maximum number of tasks that may be run.
    weight : float, default: 1.0
        A float-point weight that multiplies the average score of the
        best candidate for the strategy during the probability
        composition. By default, all strategies have weight 1.0.
        At the very first beginning of a session, the best score of
        any strategy is 0.0, so we multiply 0.0 * weight.  That is,
        the weight will have effetive changes only after the end of
        the first task.
    warm_up : int, default: 20
        The minimum number of candidates solutions the strategy should
        get to try before its interest from the engine declines due
        to better strategies.  The warm_up value is sumed to the
        best score when computing probabilities, thus this effectively
        affects probabilites since the very first task.
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
            'best': dict(agg=dict(avg=0), params=dict()),
            'tasks_since_last_improvement': 0,
            # This is the sum of elapsed time, _not_ the difference
            # of real times between now and the last improvement.
            'time_since_last_improvement': 0,
            'finished': 0,
        }
        self.load_cache()

    def get_tasks_bounds(self, nconsumers):
        """Return the inclusive bounds for parallel tasks."""
        bounds = self.parallel_tasks_bounds
        if bounds[1] > 0:
            return bounds
        return (bounds[0], nconsumers - (bounds[1] + 1))

    def get_fit_bounds(self, nconsumers):
        """Return the inclusive bounds for parallel fit."""
        bounds = self.parallel_fit_bounds
        if bounds[1] > 0:
            return bounds
        assert bounds[1] != 0, 'Upper bounds must be positive or negative.'
        return (bounds[0], nconsumers - (bounds[1] + 1))

    def load_cache(self):
        """Load the cached results."""
        logger = logging.getLogger('war.strategy')
        logger.debug(CF('%s: loading cache').dark_gray, self.name)
        best = dict(agg=dict(avg=0, std=0, min=0, max=0),
                    scores=list(), params=dict())
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
            score = result['agg']['avg']
            elapsed = result['elapsed_time']
            dat = datetime.strptime(result['begin_time'], '%Y-%m-%d %H:%M:%S')
            dat += timedelta(seconds=elapsed)
            history.append((dat, score, elapsed))
            cumulative_time += elapsed
            if not best or best['agg']['avg'] < score:
                best = result
        history = sorted(history, key=lambda x: x[0])
        tsli, timesli = 0, 0
        if history:
            last_improvement = numpy.argmax([item[1] for item in history])
            tsli = max(0, len(history) - last_improvement - 1)
            timesli = sum(item[2] for item in history)
        self.cache['tasks_since_last_improvement'] = tsli
        self.cache['time_since_last_improvement'] = timesli
        logger.info(CF('%s: loaded %d cached results').yellow,
                    self.name, count)
        self.cache['cumulative_time'] = cumulative_time
        self.cache['best'] = best
        self.cache['finished'] += count
        return best

    def make_task(self, estimator, params):
        """
        Return a task if results are not available in the database.

        Parameters
        ----------
        estimator : estimator
            A fit/predict object.
        params : dict-like
            The parameters used to create the estimator.

        Returns
        -------
        Task
            A new task if task's result are not in database.
            None otherwise.
        """
        task = Task(
            strategy=self,
            strategy_id=hash(self),
            estimator=estimator,
            params=params
        )
        if self.database.find(task.id()):
            return None
        logger = logging.getLogger('war.strategy')
        logger.debug(CF('New task %s').dark_gray, task.full_id())
        return task

    def collect(self, result):
        """
        Store a task's result into the database.

        Parameters
        ----------
        result : Result
            The result object with contents for the task.
        """
        self.database.store(
            oid=result.task.id(),
            obj={'type': 'result', 'data': result.data()})
