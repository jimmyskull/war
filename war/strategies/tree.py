from war.core import Strategy


class DecisionTree(Strategy):

    def __init__(self):
        super().__init__(name='GS Tree', warm_up=50,
                         parallel_fit_bounds=(1, 1))
        self._param = None
        self._pidx = 0

    def init(self, info):
        from sklearn.model_selection import ParameterGrid
        nfeatures = info['features'].shape[1]
        # pylint: disable=I1101
        param = {
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 6),
            'min_samples_split': [2, 4, 8, 16, 32],
            'min_samples_leaf': [1, 4, 8, 16],
            'class_weight': ['balanced', None],
        }
        self._param = list(ParameterGrid(param))

    def next(self, nthreads):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        assert nthreads == 1
        size = len(self._param)
        while self._pidx < size:
            params = self._param[self._pidx]
            self._pidx += 1
            model = make_pipeline(
                Imputer(),
                DecisionTreeClassifier(
                    **params,
                    random_state=6))
            task = self.make_task(model, params)
            if task:
                return task
        raise StopIteration
