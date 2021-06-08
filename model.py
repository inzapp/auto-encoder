import tensorflow as tf


class Model:
    def __init__(self, input_shape, encoding_dim):
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.ae = self.__get_model()

    def __get_model(self):
        input_layer = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(input_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPool2D()(x)

        x = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.GlobalAveragePooling2D(name='encoder_output')(x)

        x = tf.keras.layers.Reshape(target_shape=(1, 1, self.encoding_dim))(x)
        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.UpSampling2D()(x)
        x = tf.keras.layers.Conv2DTranspose(
            filters=16,
            kernel_size=3,
            kernel_initializer='he_uniform',
            padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            filters=self.input_shape[-1],
            kernel_size=1,
            kernel_initializer='glorot_uniform',
            activation='sigmoid')(x)
        return tf.keras.models.Model(input_layer, x)
