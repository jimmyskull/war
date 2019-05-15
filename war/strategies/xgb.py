import numpy

from war.core import Strategy


class RandomSearchXGB(Strategy):

    def __init__(self):
        super().__init__(name='RS XGBoost',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=-1)
        self._cs = None

    def init(self, info):
        bins = numpy.bincount(info['target'])
        half_balanced = bins[0] / (bins[1] * 2)
        balanced = bins[0] / bins[1]
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        P = info['features'].shape[1]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=20, default_value=3),
            CSH.UniformFloatHyperparameter(
                'learning_rate', lower=0.0001, upper=1, log=True,
                default_value=0.1),
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=1000, default_value=100),
            CSH.CategoricalHyperparameter(
                'booster', choices=['gbtree', 'dart']), #  remove 'gblinear'
            CSH.UniformFloatHyperparameter(
                'gamma', lower=0.0001, upper=100, log=True,
                default_value=0.0001),
            CSH.UniformIntegerHyperparameter(
                'min_child_weight', lower=1, upper=10, default_value=1),
            CSH.UniformFloatHyperparameter(
                'colsample_bytree', lower=0.5, upper=1, default_value=1),
            CSH.UniformFloatHyperparameter(
                'colsample_bylevel', lower=0.5, upper=1, default_value=1),
            CSH.UniformFloatHyperparameter(
                'colsample_bynode', lower=0.5, upper=1, default_value=1),
            CSH.UniformFloatHyperparameter(
                'reg_alpha', lower=0.0001, upper=100, log=True,
                default_value=0.0001),
            CSH.UniformFloatHyperparameter(
                'reg_lambda', lower=0.0001, upper=100, log=True,
                default_value=0.0001),
            CSH.CategoricalHyperparameter(
                'scale_pos_weight', choices=[1, half_balanced, balanced],
                default_value=1),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        import xgboost
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            xgboost.XGBClassifier(
                **params,
                verbosity=0,
                objective='binary:logistic',
                random_state=6,  # Guaranteed to be random.
                n_jobs=nthreads
            ))
        return self.make_task(model, dict())
