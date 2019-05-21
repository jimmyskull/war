import warnings

import numpy

from war.core import Strategy


class RandomSearchCatBoost(Strategy):

    def __init__(self):
        super().__init__(name='RS CatBoost',
                         parallel_fit_bounds=(1, 1))

    def init(self, info):
        bins = numpy.bincount(info['target'])
        half_balanced = bins[0] / (bins[1] * 2)
        balanced = bins[0] / bins[1]
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'iterations', lower=50, upper=2000, default_value=1000),
            CSH.UniformFloatHyperparameter(
                'learning_rate', lower=0.001, upper=1, log=True,
                default_value=0.03),
            CSH.UniformFloatHyperparameter(
                'l2_leaf_reg', lower=0.1, upper=100, log=True,
                default_value=3.0),
            CSH.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=10,
                default_value=6),
            CSH.UniformFloatHyperparameter(
                'colsample_bylevel', lower=0.7, upper=1,
                default_value=1),
            CSH.CategoricalHyperparameter(
                'scale_pos_weight', choices=[1, half_balanced, balanced],
                default_value=1),
        ])
        self._cs = cs

    def next(self, nthreads):
        from catboost import CatBoostClassifier
        from sklearn.pipeline import make_pipeline
        params = dict(**self._cs.sample_configuration())
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = make_pipeline(
                CatBoostClassifier(
                    **params,
                    random_seed=6,
                    thread_count=nthreads,
                    logging_level='Silent',
                    allow_writing_files=False))
            return self.make_task(model, params)
