from war.core import Strategy


class RandomSearchLogisticRegressionL2(Strategy):

    def __init__(self):
        super().__init__(name='RS LR L2',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        a = CSH.UniformFloatHyperparameter(
            'C', lower=0.001, upper=100, default_value=0.1, log=True)
        b = CSH.CategoricalHyperparameter('penalty', choices=['l2'])
        cs.add_hyperparameters([a, b])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, Imputer
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            LogisticRegression(**params))
        return self.make_task(model, params)


class RandomSearchLogisticRegressionL1(Strategy):

    def __init__(self):
        super().__init__(name='RS LR L1',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformFloatHyperparameter(
                'C', lower=0.001, upper=100, default_value=0.1, log=True),
            CSH.CategoricalHyperparameter('penalty', choices=['l1'])
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, Imputer
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            LogisticRegression(**params))
        return self.make_task(model, params)
