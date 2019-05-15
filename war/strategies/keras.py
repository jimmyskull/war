from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from war.core import Strategy


class RandomSearchKerasMLP(Strategy):

    def __init__(self):
        super().__init__(name='RS Keras MLP',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        self._nfeatures = None
        self._cs = None

    def init(self, info):
        self._nfeatures = info['features'].shape[1]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'epochs', lower=50, upper=1000, default_value=100),
            CSH.CategoricalHyperparameter(
                'activation_0', choices=['softmax', 'relu', 'sigmoid']),
            CSH.UniformFloatHyperparameter(
                'dropout_1', lower=0.0, upper=0.9, default_value=0.5),
            CSH.UniformIntegerHyperparameter(
                'size_2', lower=2, upper=self._nfeatures * 2,
                default_value=self._nfeatures // 2 + 1),
            CSH.CategoricalHyperparameter(
                'activation_2', choices=['softmax', 'relu', 'sigmoid']),
            CSH.UniformFloatHyperparameter(
                'dropout_3', lower=0.0, upper=0.9, default_value=0.5),
            CSH.CategoricalHyperparameter(
                'activation_4', choices=['softmax', 'relu', 'sigmoid']),
        ])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        # create model
        model = Sequential()
        model.add(Dense(self._nfeatures,
                        input_dim=self._nfeatures,
                        kernel_initializer='normal',
                        activation=params['activation_0']))
        model.add(Dropout(params['dropout_1']))
        model.add(Dense(params['size_2'],
                        activation=params['activation_2']))
        model.add(Dropout(params['dropout_3']))
        model.add(Dense(1,
                        kernel_initializer='normal',
                        activation=params['activation_4']))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam')
        # Build wrapper
        model = make_pipeline(
            Imputer(),
            KerasClassifier(
                build_fn=model,
                epochs=params['epochs'],
                verbose=-1))
        return self.make_task(model, params)
