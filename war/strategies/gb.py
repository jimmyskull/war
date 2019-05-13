from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from war.core import Strategy


class RandomSearchGradientBoosting(Strategy):

    def __init__(self):
        super().__init__(name='RS Gradient Boosting',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        cs = CS.ConfigurationSpace()
        a = CSH.UniformIntegerHyperparameter(
            'n_estimators', lower=50, upper=500, default_value=100)
        b = CSH.UniformFloatHyperparameter(
            'learning_rate', lower=0.001, upper=1, default_value=0.1, log=True)
        c = CSH.CategoricalHyperparameter('max_depth', choices=[2, 3, 4])
        cs.add_hyperparameters([a, b, c])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            GradientBoostingClassifier(**params))
        return self.make_task(model, params)
