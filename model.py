import tensorflow as tf


class Model:
    def __init__(self, input_shape, ae_input_shape, encoding_dim):
        self.input_shape = input_shape
        self.ae_input_shape = ae_input_shape
        self.encoding_dim = encoding_dim
        self.ae = self.__get_model()

    def __get_model(self):
        input_layer = tf.keras.layers.Input(shape=self.ae_input_shape)
        x = tf.keras.layers.Dense(
            units=128,
            kernel_initializer='he_uniform',
            activation='relu')(input_layer)
        x = tf.keras.layers.Dense(
            units=self.encoding_dim,
            kernel_initializer='glorot_uniform',
            activation='sigmoid',
            name='encoder_output')(x)
        x = tf.keras.layers.Dense(
            units=128,
            kernel_initializer='he_uniform',
            activation='relu')(x)
        x = tf.keras.layers.Dense(
            units=int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2]),
            activation='sigmoid')(x)
        x = tf.keras.layers.Reshape(target_shape=self.input_shape)(x)
        return tf.keras.models.Model(input_layer, x)
