import tensorflow as tf


class Model:
    def __init__(self, input_shape, ae_input_shape, encoding_dim):
        self.input_shape = input_shape
        self.ae_input_shape = ae_input_shape
        self.encoding_dim = encoding_dim
        encoder_input, encoder_output, ae_output = self.__get_model_layers()
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        self.ae = tf.keras.models.Model(encoder_input, ae_output)

    def __get_model_layers(self):
        encoder_input = tf.keras.layers.Input(shape=self.ae_input_shape)
        x = tf.keras.layers.Dense(
            units=128,
            kernel_initializer='he_uniform',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoder_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        encoder_output = tf.keras.layers.Dense(
            units=self.encoding_dim,
            activation='sigmoid')(x)
        x = tf.keras.layers.Dense(
            units=128,
            kernel_initializer='he_uniform',
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(0.001))(encoder_output)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(
            units=int(self.input_shape[0] * self.input_shape[1] * self.input_shape[2]),
            activation='sigmoid')(x)
        ae_output = tf.keras.layers.Reshape(target_shape=self.input_shape)(x)
        return encoder_input, encoder_output, ae_output
