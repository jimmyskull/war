import os

from sklearn.base import BaseEstimator, ClassifierMixin

from war.core import Strategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LazyKerasBuild(BaseEstimator, ClassifierMixin):

    def __init__(self, task_params):
        super().__init__()
        self.task_params = task_params
        self._model = None

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        params = self.task_params
        # create model
        model = Sequential()
        model.add(Dense(params['nfeatures'],
                        input_dim=params['nfeatures'],
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
        return model

    def fit(self, *args, **kwargs):
        from keras import backend as K
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Imputer
        import tensorflow as tf

        nthreads = self.task_params['nthreads']
        tf.logging.set_verbosity(tf.logging.ERROR)
        config = tf.ConfigProto(
            log_device_placement=False,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count = {'CPU': nthreads}
        )

        session = tf.Session(config=config)
        K.set_session(session)


        # Build wrapper
        self._model = make_pipeline(
            Imputer(),
            KerasClassifier(
                build_fn=self.build_model,
                epochs=self.task_params['epochs'],
                verbose=0))

        self._model.fit(*args, **kwargs)

    def predict_proba(self, X):
        return self._model.predict_proba(X)


class RandomSearchKerasMLP(Strategy):

    def __init__(self):
        super().__init__(name='RS Keras MLP',
                         max_parallel_tasks=-1,
                         max_threads_per_estimator=1)
        self._nfeatures = None
        self._cs = None

    def init(self, info):
        # pylint: disable=I1101
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
        self._nfeatures = info['features'].shape[1]
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CSH.UniformIntegerHyperparameter(
                'epochs', lower=50, upper=100, default_value=100),
            CSH.CategoricalHyperparameter(
                'activation_0', choices=['relu']),
            CSH.UniformFloatHyperparameter(
                'dropout_1', lower=0.0, upper=0.7, default_value=0.5),
            CSH.UniformIntegerHyperparameter(
                'size_2', lower=2, upper=self._nfeatures,
                default_value=max(1, self._nfeatures // 2)),
            CSH.CategoricalHyperparameter(
                'activation_2', choices=['relu']),
            CSH.UniformFloatHyperparameter(
                'dropout_3', lower=0.0, upper=0.9, default_value=0.5),
            CSH.CategoricalHyperparameter(
                'activation_4', choices=['sigmoid']),
        ])
        self._cs = cs

    def next(self, nthreads):
        assert nthreads == 1
        params = dict(**self._cs.sample_configuration())
        params['nfeatures'] = self._nfeatures
        params['nthreads'] = nthreads
        model = LazyKerasBuild(params)
        return self.make_task(model, params)
