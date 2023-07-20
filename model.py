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
    def __init__(self, input_shape, latent_dim, denoising_model):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.denoising_model = denoising_model
        self.ae = None
        self.encoder = None
        if self.denoising_model:
            self.channels = [16, 32, 64]
        else:
            self.channels = [16, 32, 64, 128, 256, 512]
        scale = pow(2, len(self.channels) - 1)
        self.latent_rows = input_shape[0] // scale
        self.latent_cols = input_shape[1] // scale

    def build(self, bn=False):
        if self.denoising_model:
            assert self.input_shape[0] % 4 == 0 and self.input_shape[1] % 4 == 0, 'input_rows, input_cols must be multiple of 4 when training denoising model'
        else:
            assert self.input_shape[0] % 32 == 0 and self.input_shape[1] % 32 == 0, 'input_rows, input_cols must be multiple of 32'
        encoder_input = tf.keras.layers.Input(shape=self.input_shape, name='encoder_input')
        x = encoder_input
        features = []
        for i, channel in enumerate(self.channels):
            if i == len(self.channels) - 1:
                x = self.conv2d(x, channel, 3, 1, bn=bn)
            else:
                if self.denoising_model:
                    x = self.conv2d(x, channel, 3, 1, bn=bn)
                    features.append(x)
                    x = self.max_pool(x)
                else:
                    x = self.conv2d(x, channel, 3, 2, bn=bn)
        x = x if self.denoising_model else self.encoding_layer(x, self.latent_dim)
        encoder_output = x

        if self.denoising_model:
            features = list(reversed(features))
        if not self.denoising_model:
            x = self.dense(x, self.latent_rows * self.latent_cols * self.channels[-1], bn=bn)
            x = self.reshape(x, (self.latent_rows, self.latent_cols, self.channels[-1]))
        for i, channel in enumerate(list(reversed(self.channels))[1:]):
            if self.denoising_model:
                x = self.upsampling(x)
                x = self.conv2d(x, channel, 3, 1, bn=bn)
                x = self.add([x, features[i]])
            else:
                x = self.conv2d_transpose(x, channel, 3, 2, bn=bn)
        x = self.decoding_layer(x)
        decoder_output = x
        self.encoder = tf.keras.models.Model(encoder_input, encoder_output)
        self.ae = tf.keras.models.Model(encoder_input, decoder_output)
        return self.ae, self.encoder

    def encoding_layer(self, x, latent_dim):
        return self.dense(self.flatten(x), latent_dim, bn=False, activation='linear', name='encoder_output')

    def decoding_layer(self, x):
        return self.conv2d(x, self.input_shape[-1], 1, 1, bn=False, activation='sigmoid', name='decoder_output')

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            activation='linear' if bn else activation,
            kernel_initializer=self.kernel_initializer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation) if bn else x

    def conv2d_transpose(self, x, filters, kernel_size, strides, bn=False, activation='relu'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            kernel_size=kernel_size,
            use_bias=not bn,
            activation='linear' if bn else activation,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation) if bn else x

    def dense(self, x, units, bn=False, activation='relu', name=None):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            activation='linear' if bn else activation,
            kernel_initializer=self.kernel_initializer(),
            name=name)(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation) if bn else x

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization()(x)

    def kernel_initializer(self):
        return tf.keras.initializers.glorot_normal()

    def activation(self, x, activation):
        return tf.keras.layers.Activation(activation=activation)(x) if activation != 'linear' else x

    def max_pool(self, x):
        return tf.keras.layers.MaxPool2D()(x)

    def upsampling(self, x):
        return tf.keras.layers.UpSampling2D()(x)

    def add(self, layers):
        return tf.keras.layers.Add()(layers)

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        if not self.denoising_model:
            self.encoder.summary()
            print()
        self.ae.summary()

