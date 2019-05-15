from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import time
import traceback
import warnings

from sklearn.model_selection import cross_val_score
import numpy

from war.result import Result


class Task(object):

    def __init__(self, strategy, strategy_id, estimator, params):
        self.strategy = strategy
        self.strategy_id = strategy_id
        self.estimator = estimator
        self.params = params
        self.features = None
        self.target = None
        self.cv = None
        self.n_jobs = None
        self.total_jobs = None
        self.scoring = 'roc_auc'
        self.estimator_njobs = None

    def __call__(self):
        start_time = time.time()
        begin_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        agg = None
        scores = None
        error_info = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                cv = cross_val_score(
                    estimator=self.estimator,
                    X=self.features,
                    y=self.target,
                    scoring=self.scoring,
                    cv=self.cv,
                    n_jobs=self.n_jobs
                )
                status = 'OK'
                message = ''
                agg = dict(
                    avg=numpy.mean(cv),
                    std=numpy.std(cv),
                    min=numpy.min(cv),
                    max=numpy.max(cv)
                )
                scores = cv.tolist()
        except Exception as err:
            status = 'FAILED'
            error_info = {
                'message': '{}: {}'.format(type(err).__name__, err),
                'traceback': '\n'.join(traceback.format_tb(err.__traceback__))
            }

        elapsed_time = time.time() - start_time
        result = Result(
            task=self,
            begin_time=begin_time,
            elapsed_time=elapsed_time,
            total_time=elapsed_time * self.total_jobs,
            status=status,
            error_info=error_info,
            agg=agg,
            scores=scores,
            jobs=self.total_jobs
        )
        return result

    def id(self):
        info = [
            ('strategy', self.strategy.__class__.__name__),
            ('estimator', repr(self.estimator)),
            ('scoring', self.scoring),
        ]
        if isinstance(self.params, dict):
            params = [(k, v) for k, v in self.params.items()]
        else:
            params = OrderedDict(**self.params)
        info.append(('params', str(params)))
        info = sorted(info)
        sha1 = hashlib.sha1(json.dumps(info).encode('utf-8'))
        return sha1.hexdigest()
