"""
Authors : inzapp

Github url : https://github.com/inzapp/auto-encoder

Copyright (c) 2021 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
import tensorflow as tf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Model:
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.ae = None
        self.decoder = None
        self.encoder = None
        self.latent_rows = input_shape[0] // 32
        self.latent_cols = input_shape[1] // 32

    def build(self):
        assert self.input_shape[0] % 32 == 0 and self.input_shape[1] % 32 == 0
        encoder_input, encoder_output = self.build_encoder(bn=False)
        decoder_input, decoder_output = self.build_decoder(bn=False)
        self.decoder = tf.keras.models.Model(decoder_input, decoder_output)
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        ae_output = self.decoder(encoder_output)
        self.ae = tf.keras.models.Model(encoder_input, ae_output)
        return self.encoder, self.decoder, self.ae

    def build_encoder(self, bn):
        encoder_input = tf.keras.layers.Input(shape=self.input_shape)
        x = encoder_input
        x = self.conv2d(x,  32, 3, 2, activation='relu', bn=bn)
        x = self.conv2d(x,  64, 3, 2, activation='relu', bn=bn)
        x = self.conv2d(x, 128, 3, 2, activation='relu', bn=bn)
        x = self.conv2d(x, 256, 3, 2, activation='relu', bn=bn)
        x = self.conv2d(x, 512, 3, 2, activation='relu', bn=bn)
        x = self.flatten(x)
        encoder_output = self.dense(x, self.latent_dim, activation='linear', bn=bn)
        return encoder_input, encoder_output

    def build_decoder(self, bn):
        decoder_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = decoder_input
        x = self.dense(x, self.latent_rows * self.latent_cols * 512, activation='relu', bn=bn)
        x = self.reshape(x, (self.latent_rows, self.latent_cols, 512))
        x = self.conv2d_transpose(x, 512, 3, 2, activation='relu', bn=bn)
        x = self.conv2d_transpose(x, 256, 3, 2, activation='relu', bn=bn)
        x = self.conv2d_transpose(x, 128, 3, 2, activation='relu', bn=bn)
        x = self.conv2d_transpose(x,  64, 3, 2, activation='relu', bn=bn)
        x = self.conv2d_transpose(x,  32, 3, 2, activation='relu', bn=bn)
        decoder_output = self.conv2d_transpose(x, self.input_shape[-1], 1, 1, activation='tanh', bn=bn)
        return decoder_input, decoder_output

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu'):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides, bn=False, activation='relu'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='relu'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.he_normal()

    def activation(self, x, activation):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        else:
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.decoder.summary()
        print()
        self.ae.summary()

