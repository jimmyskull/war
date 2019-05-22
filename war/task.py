"""Worker task."""
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import time
import traceback
import warnings
import psutil

import numpy

from war.result import Result


class Task:
    """
    A Task is an object that travels between processes, to run inside
    a worker process.

    Parameters
    ----------
    strategy : Strategy
        The strategy that generated this task.
    strategy_id : int
        The hash of this strategy in the main thread.
    estimator : object
        An fit/predict_proba estimator pipeline.
    params : dict-like
        A dictionary that is stored in the result object.
    """

    def __init__(self, strategy, strategy_id, estimator, params):
        self.strategy = strategy
        self.strategy_id = strategy_id
        self.estimator = estimator
        self.params = params    # Parameters used for the estimator
        self.features = None    # Features dataframe/matrix
        self.target = None      # Target series/array
        self.data_id = None     # SHA-1 of features + target
        self.trials = None      # Number of trials in validation
        self.n_jobs = None      # Validation n_jobs
        self.total_jobs = None  # Validation n_jobs * fit n_jobs
        self.validator = None   # Callable for validatios
        self.scoring = None     # Scorer callable
        self.creation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def __call__(self):
        # pylint: disable=E1102
        start_time = time.time()
        begin_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        proc = psutil.Process()
        agg = None
        scores = None
        error_info = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                scores = self.validator(
                    estimator=self.estimator,
                    X=self.features,
                    y=self.target,
                    scoring=self.scoring,
                    cv=self.trials,
                    n_jobs=self.n_jobs
                )
                status = 'OK'
                agg = dict(
                    avg=numpy.mean(scores),
                    std=numpy.std(scores),
                    min=numpy.min(scores),
                    max=numpy.max(scores)
                )
                scores = scores.tolist()
        except Exception as err:  # pylint: disable=W0703
            status = 'FAILED'
            error_info = {
                'message': '{}: {}'.format(type(err).__name__, str(err)),
                'traceback': '\n'.join(traceback.format_tb(err.__traceback__))
            }
        cpu_times = proc.cpu_times()
        times = {
            'user': cpu_times[0],
            'system': cpu_times[1],
            'children_user': cpu_times[2],
            'children_system': cpu_times[3],
        }
        elapsed_time = time.time() - start_time
        result = Result(
            task=self,
            begin_time=begin_time,
            real_times=dict(elapsed=elapsed_time,
                            total=elapsed_time * self.total_jobs),
            cpu_times=times,
            status=status,
            error_info=error_info,
            agg=agg,
            scores=scores,
            njobs=dict(valid=self.n_jobs,
                       fit=self.total_jobs // self.n_jobs)
        )
        # Save result to the database.
        self.strategy.collect(result)
        return result

    def scoring_name(self):
        # pylint: disable=E1101,W0212
        if isinstance(self.scoring, str):
            return self.scoring
        assert callable(self.scoring)
        return self.scoring._score_func.__name__

    def validator_name(self):
        # pylint: disable=E1101
        assert callable(self.validator)
        return self.validator.__name__

    def data(self):
        data = {
            'params': self.params,
            'njobs': dict(valid=self.n_jobs,
                          fit=self.total_jobs // self.n_jobs),
            'creation_date': self.creation_date,
        }
        return data

    def full_id(self):
        """Return the full ID for this task."""
        name = self.strategy.__class__.__name__
        return f'{name}/{self.id()}'

    def id(self):
        """Return SHA-1 hex digest of this task."""
        info = [
            ('strategy', self.strategy.__class__.__name__),
            ('data', self.data_id),
            ('trials', self.trials),
            ('validator', self.validator_name()),
            ('scoring', self.scoring_name()),
        ]
        if not isinstance(self.params, dict):
            params = OrderedDict(**self.params)
        params = [(k, v) for k, v in self.params.items()]
        params = sorted(params, key=lambda x: x[0])
        info.append(('params', str(params)))
        info = sorted(info, key=lambda x: x[0])
        sha1 = hashlib.sha1(json.dumps(info).encode('utf-8'))
        return sha1.hexdigest()
