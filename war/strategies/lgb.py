import numpy

from sklearn.base import BaseEstimator, ClassifierMixin

from war.core import Strategy


class LGBMModel(BaseEstimator, ClassifierMixin):

    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X, y=None):
        pred = self.model.predict(X).reshape(-1, 1)
        return numpy.concatenate((1 - pred, pred), axis=1)


class RandomSearchLGBM(Strategy):

    def __init__(self):
        super().__init__(name='RS Full LightGBM',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=-1)
        self._cs = None

    def init(self, info):
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        P = info['features'].shape[1]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter(
                'boosting_type', choices=['gbdt', 'dart', 'goss']),
            CSH.UniformIntegerHyperparameter(
                'num_leaves', lower=2, upper=63, default_value=31),
            CSH.UniformFloatHyperparameter(
                'learning_rate', lower=0.0001, upper=1, log=True,
                default_value=0.1),
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=1000, default_value=100),
            CSH.CategoricalHyperparameter(
                'class_weight', choices=['None', 'balanced']),
            CSH.CategoricalHyperparameter(
                'min_split_gain', choices=[0, 0.001, 0.01], default_value=0),
            CSH.CategoricalHyperparameter(
                'min_child_weight', choices=[1e-3, 1e-2, 1e-1],
                default_value=1e-3),
            CSH.CategoricalHyperparameter(
                'min_child_samples',
                choices=[10, 20, 30, 40, 50, 60, 70, 80, 90,
                         100, 110, 120, 130, 140, 150],
                default_value=20),
            CSH.UniformFloatHyperparameter(
                'subsample', lower=0.5, upper=1, default_value=1),
            CSH.UniformFloatHyperparameter(
                'colsample_bytree', lower=0.5, upper=1, default_value=1),
            CSH.UniformFloatHyperparameter(
                'reg_alpha', lower=0.0001, upper=100, log=True,
                default_value=0.0001),
            CSH.UniformFloatHyperparameter(
                'reg_lambda', lower=0.0001, upper=100, log=True,
                default_value=0.0001),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        import lightgbm as lgb
        params = dict(**self._cs.sample_configuration())
        if params['class_weight'] == 'None':
            del params['class_weight']
        model = make_pipeline(
            Imputer(),
            LGBMModel(
                lgb.LGBMModel(
                    **params,
                    objective='binary',
                    silent=True,
                    random_state=6, # Garanteed to be random.
                    n_jobs=nthreads
            )))
        return self.make_task(model, dict())
