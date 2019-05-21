from war.core import Strategy


class RandomSearchExtraTrees(Strategy):

    def __init__(self):
        super().__init__(name='RS Extra Trees')
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'n_estimators', lower=10, upper=1000, default_value=100),
            CSH.CategoricalHyperparameter(
                'max_depth', choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
            CSH.CategoricalHyperparameter(
                'min_samples_split', choices=[2, 5, 10, 16, 32, 64]),
            CSH.CategoricalHyperparameter(
                'min_samples_leaf', choices=[1, 2, 5, 10, 16, 32, 64]),
            CSH.CategoricalHyperparameter(
                'criterion', choices=['entropy', 'gini']),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            ExtraTreesClassifier(
                **params,
                random_state=6,
                class_weight='balanced',
                n_jobs=nthreads))
        return self.make_task(model, params)
