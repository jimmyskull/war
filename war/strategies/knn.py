from war.core import Strategy


class GridSearchPCAKNN(Strategy):

    def __init__(self):
        super().__init__(name='PCA + k-NN', parallel_fit_bounds=(1, -1))
        self._param = None
        self._pidx = 0

    def init(self, info):
        from sklearn.model_selection import ParameterGrid
        nfeatures = info['features'].shape[1]
        # pylint: disable=I1101
        param = {
            'n_components': range(1, nfeatures),
            'n_neighbors': range(1, 12),
            'lp_norm': range(1, 3),
        }
        self._param = list(ParameterGrid(param))

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer, StandardScaler
        size = len(self._param)
        while self._pidx < size:
            params = self._param[self._pidx]
            self._pidx += 1
            model = make_pipeline(
                Imputer(),
                StandardScaler(),
                PCA(n_components=params['n_components']),
                KNeighborsClassifier(
                    n_neighbors=params['n_neighbors'],
                    p=params['lp_norm'],
                    n_jobs=nthreads
                ))
            task = self.make_task(model, params)
            self._pidx += 1
            if task:
                return task
        raise StopIteration
