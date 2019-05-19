from war.core import Strategy


class LDA(Strategy):

    def __init__(self):
        super().__init__(name='LDA', warm_up=100, parallel_tasks_bounds=(1, 1),
                         parallel_fit_bounds=(1, 1), max_tasks=1)

    def next(self, nthreads):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        assert nthreads == 1
        model = make_pipeline(
            Imputer(),
            LinearDiscriminantAnalysis())
        return self.make_task(model, dict())


class PCALDA(Strategy):

    def __init__(self):
        super().__init__(name='PCA + LDA', warm_up=100,
                         parallel_fit_bounds=(1, 1))
        self.nfeatures = None
        self.curr = None

    def init(self, info):
        self.nfeatures = max(1, info['features'].shape[1] - 1)
        self.curr = 1

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        assert nthreads == 1
        while self.curr < self.nfeatures:
            model = make_pipeline(
                Imputer(),
                PCA(n_components=self.curr),
                LinearDiscriminantAnalysis())
            task = self.make_task(model, dict(n_components=self.curr))
            self.curr += 1
            if task:
                return task
        raise StopIteration
