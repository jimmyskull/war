from war.core import Strategy


class PCAGaussianMixture(Strategy):

    def __init__(self):
        super().__init__(name='PCA + Gaussian Mix',
                         parallel_fit_bounds=(1, 1))

    def init(self, info):
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        n_features = max(1, info['features'].shape[1] - 1)
        cs = CS.ConfigurationSpace()
        max_components = n_features
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'pca_n_components',
                lower=1, upper=max_components,
                default_value=min(2, n_features)),
            CSH.UniformIntegerHyperparameter(
                'n_components', lower=1, upper=min(2, max_components),
                default_value=1),
            CSH.UniformIntegerHyperparameter(
                'max_iter', lower=50, upper=1000, default_value=100),
            CSH.UniformFloatHyperparameter(
                'reg_covar', lower=0.0000001, upper=0.0001,
                default_value=0.0000001),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import Imputer, StandardScaler
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            PCA(n_components=params['pca_n_components']),
            GaussianMixture(
                n_components=params['n_components'],
                reg_covar=params['reg_covar'],
                max_iter=params['max_iter'],
                n_init=10,
                random_state=6,
            ))
        task = self.make_task(model, params)
        return task
