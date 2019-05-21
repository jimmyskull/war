from war.core import Strategy


class RandomSearchBaggingTree(Strategy):

    def __init__(self):
        super().__init__(name='RS Bagging Tree')
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=1000, default_value=10),
            CSH.UniformIntegerHyperparameter(
                'max_depth', lower=1, upper=7, default_value=1),
            CSH.UniformFloatHyperparameter(
                'max_samples', lower=0.5, upper=1,
                default_value=1),
            CSH.UniformFloatHyperparameter(
                'max_features', lower=0.5, upper=1,
                default_value=1),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.ensemble import BaggingClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        from sklearn.tree import DecisionTreeClassifier
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            BaggingClassifier(
                base_estimator=(
                    DecisionTreeClassifier(max_depth=params['max_depth'])),
                n_estimators=params['n_estimators'],
                max_samples=params['max_samples'],
                max_features=params['max_features'],
                random_state=6,
                n_jobs=nthreads))
        return self.make_task(model, params)
