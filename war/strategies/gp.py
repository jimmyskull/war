from war.core import Strategy


class GridSearchGP(Strategy):

    def __init__(self):
        super().__init__(name='GS Gaussian Process')
        self._param = None
        self._pidx = 0

    def init(self, info):
        # pylint: disable=I1101
        from sklearn.model_selection import ParameterGrid
        nfeatures = info['features'].shape[1]
        param = {
            'n_components': range(1, nfeatures),
            'kernel': ['rbf', 'white', 'dot'],
        }
        self._param = list(ParameterGrid(param))

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import (RBF, WhiteKernel,
                                                      DotProduct)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer, StandardScaler
        size = len(self._param)
        while self._pidx < size:
            params = self._param[self._pidx]
            self._pidx += 1
            kernel, kernel_name = None, params['kernel']
            if kernel_name == 'rbf':
                kernel = RBF()
            elif kernel_name == 'white':
                kernel = WhiteKernel()
            elif kernel_name == 'dot':
                kernel_name = DotProduct()
            model = make_pipeline(
                Imputer(),
                StandardScaler(),
                PCA(n_components=params['n_components']),
                GaussianProcessClassifier(
                    kernel=kernel,
                    random_state=6,
                    copy_X_train=False,
                    n_jobs=nthreads))
            task = self.make_task(model, params)
            if task:
                return task
        raise StopIteration
