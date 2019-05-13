
from unittest.mock import Mock

from pandas import read_csv
import pytest

from war.scheduler import Scheduler
from war.strategies import RandomSearchLogisticRegression
from war.strategy import Strategy


class OneShotLogisticRegression(Strategy):

    def __init__(self):
        super().__init__(max_parallel_tasks=-1, max_threads_per_estimator=1)

    def next(self, nthreads):
        return Mock()


def test_scheduler():
    train = read_csv('/home/jimmy/dev/chef/GiveMeSomeCredit/cs-training.csv',
                     index_col=0)
    test = read_csv('/home/jimmy/dev/chef/GiveMeSomeCredit/cs-test.csv',
                    index_col=0)
    X, y = train.drop('SeriousDlqin2yrs', axis=1), train['SeriousDlqin2yrs']

    strategies = [
        RandomSearchLogisticRegression()
    ]
    nconsumers = 8
    nfolds = 8

    sched = Scheduler(strategies, nconsumers, nfolds)
    tasks = sched.next()

    print(tasks)

    task = tasks[0]
    task.features = X
    task.target = y
    task.cv = nfolds
    result = task()
    print(result)

    sched.collect(result)

    assert len(tasks) == 0
