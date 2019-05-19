from war.core import Strategy


class RandomSearchAdaBoost(Strategy):

    def __init__(self):
        super().__init__(name='RS AdaBoost', parallel_fit_bounds=(1, 1))
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=50, upper=1000, default_value=50),
            CSH.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=5, default_value=1),
            CSH.UniformFloatHyperparameter(
                'learning_rate', lower=0.001, upper=1, log=True,
                default_value=1),
        ])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        from sklearn.tree import DecisionTreeClassifier
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            AdaBoostClassifier(
                base_estimator=
                    DecisionTreeClassifier(max_depth=params['max_depth']),
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                random_state=6))
        return self.make_task(model, params)
