import tensorflow as tf


class Model:
    def __init__(self, input_shape, ae_input_shape, encoding_dim):
        self.input_shape = input_shape
        self.ae_input_shape = ae_input_shape
        self.encoding_dim = encoding_dim
        self.ae = self.__get_model()

    def __get_model(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.ae_input_shape),
            tf.keras.layers.Dense(
                units=128,
                kernel_initializer='he_uniform',
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(
                units=self.encoding_dim,
                kernel_initializer='glorot_uniform',
                activation='sigmoid',
                name='encoder_output'),
            tf.keras.layers.Dense(
                units=128,
                kernel_initializer='he_uniform',
                use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(
                units=int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2]),
                activation='sigmoid'),
            tf.keras.layers.Reshape(target_shape=self.input_shape)])
