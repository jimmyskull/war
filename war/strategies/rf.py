from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from war.core import Strategy


class RandomSearchRandomForest(Strategy):

    def __init__(self):
        super().__init__(name='RS Random Forest',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=-1)
        cs = CS.ConfigurationSpace()
        a = CSH.UniformIntegerHyperparameter(
            'n_estimators', lower=10, upper=1000, default_value=100)
        b = CSH.CategoricalHyperparameter(
            'max_depth', choices=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        c = CSH.CategoricalHyperparameter(
            'min_samples_split', choices=[2, 5, 10, 16, 32, 64])
        cs.add_hyperparameters([a, b, c])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            RandomForestClassifier(**params, n_jobs=nthreads))
        return self.make_task(model, params)
