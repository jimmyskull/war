from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Imputer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from war.core import Strategy


class RandomSearchPCAMLP(Strategy):

    def __init__(self):
        super().__init__(name='RS PCA + MLP',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        cs = CS.ConfigurationSpace()
        a = CSH.UniformIntegerHyperparameter(
            'size', lower=1, upper=1000, default_value=1)
        b = CSH.UniformIntegerHyperparameter(
            'max_iter', lower=1, upper=1000, default_value=1)
        c = CSH.UniformFloatHyperparameter(
            'learning_rate', lower=0.001, upper=2, default_value=0.1, log=True)
        d = CSH.UniformFloatHyperparameter(
            'alpha', lower=0.001, upper=2, default_value=0.1, log=True)
        cs.add_hyperparameters([a, b, c, d])
        self._mlp_cs = cs

    def init(self, info):
        nfeatures = info['features'].shape[1]
        cs = CS.ConfigurationSpace()
        a = CSH.CategoricalHyperparameter('n_components',
                                          choices=range(1, nfeatures+1))
        cs.add_hyperparameters([a])
        self._pca_cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        mlp_params = dict(**self._mlp_cs.sample_configuration())
        pca_params = dict(**self._pca_cs.sample_configuration())
        model = make_pipeline(
            Imputer(),
            StandardScaler(),
            PCA(**pca_params),
            MLPClassifier(
                hidden_layer_sizes=(mlp_params['size'],),
                learning_rate_init=mlp_params['learning_rate'],
                alpha=mlp_params['alpha'],
                max_iter=mlp_params['max_iter'],
            ))
        return self.make_task(model, dict(mlp=mlp_params, pca=pca_params))
