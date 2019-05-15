from war.core import Strategy


class RandomSearchGradientBoosting(Strategy):

    def __init__(self):
        super().__init__(name='RS Gradient Boosting',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=1000, default_value=100),
            CSH.UniformFloatHyperparameter(
                'learning_rate', lower=0.001, upper=1, log=True,
                default_value=0.1),
            CSH.CategoricalHyperparameter('max_depth', choices=[2, 3, 4])
        ])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            GradientBoostingClassifier(**params))
        return self.make_task(model, params)
