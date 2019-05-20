from sklearn.base import BaseEstimator, ClassifierMixin
import numpy

from war.core import Strategy


class PerceptronProba(BaseEstimator, ClassifierMixin):

    def __init__(self, task_params):
        from sklearn.linear_model import Perceptron
        self.task_params = task_params
        self.model = Perceptron(
            penalty=task_params['penalty'],
            alpha=task_params['alpha'],
            max_iter=task_params['max_iter'],
            eta0=task_params['eta0'],
            random_state=6,
            class_weight='balanced'
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        pred = self.model.predict(X)
        return numpy.stack([1 - pred, pred]).T


class PCAPerceptron(Strategy):

    def __init__(self):
        super().__init__(name='PCA + Perceptron', parallel_fit_bounds=(1, 1))

    def init(self, info):
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        n_features = max(1, info['features'].shape[1] - 1)
        cs = CS.ConfigurationSpace()
        max_components = n_features
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'n_components',
                lower=1, upper=max_components,
                default_value=min(2, n_features)),
            CSH.CategoricalHyperparameter(
                'penalty', choices=['l1', 'l2']),
            CSH.UniformFloatHyperparameter(
                'alpha', lower=0.0001, upper=100,
                default_value=0.0001, log=True),
            CSH.UniformIntegerHyperparameter(
                'max_iter', lower=50, upper=10000, default_value=1000),
            CSH.UniformFloatHyperparameter(
                'eta0', lower=0.1, upper=1, default_value=1),
        ])
        self._cs = cs

    def next(self, nthreads):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer, StandardScaler
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            PCA(n_components=params['n_components']),
            PerceptronProba(params))
        task = self.make_task(model, params)
        return task
