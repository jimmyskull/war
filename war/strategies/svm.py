from war.core import Strategy


class RandomSearchPCASVMLinear(Strategy):

    def __init__(self):
        super().__init__(name='RS PCA + SVM Linear',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)

    def init(self, info):
        nfeatures = info['features'].shape[1]
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        # PCA
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.CategoricalHyperparameter('n_components',
                                          choices=range(1, nfeatures+1))
        ])
        self._pca_cs = cs
        # SVM
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformFloatHyperparameter(
                'C', lower=0.001, upper=100, default_value=0.1, log=True)
        ])
        self._svm_cs = cs

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler, Imputer
        from sklearn.svm import SVC
        assert nthreads == 1
        svm_params = dict(**self._svm_cs.sample_configuration())
        pca_params = dict(**self._pca_cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            PCA(**pca_params),
            SVC(**svm_params,
                kernel='linear',
                probability=True,
                class_weight='balanced',
                max_iter=100))
        return self.make_task(model, dict(svm=svm_params, pca=pca_params))
