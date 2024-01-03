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
import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from time import time
from model import Model
from generator import DataGenerator
from lr_scheduler import LRScheduler


class AutoEncoder:
    def __init__(self,
                 input_rows,
                 input_cols,
                 input_channels,
                 train_image_path,
                 validation_image_path,
                 lr,
                 warm_up,
                 batch_size,
                 latent_dim,
                 iterations,
                 save_interval,
                 training_view,
                 checkpoint_path='checkpoint'):
        assert save_interval >= 1000
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.input_shape = (input_rows, input_cols, input_channels)
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval
        self.iterations = iterations
        self.training_view = training_view
        self.checkpoint_path = checkpoint_path
        self.live_view_previous_time = time()

        self.train_image_paths = None
        self.validation_image_paths = None
        self.encoder = None
        self.ae = None

    def init(self, model_path):
        if not self.is_valid_path(self.train_image_path):
            print(f'train image path is not valid : {self.train_image_path}')
            exit(0)

        if not self.is_valid_path(self.validation_image_path):
            print(f'validation image path is not valid : {self.validation_image_path}')
            exit(0)

        self.train_image_paths = self.init_image_paths(self.train_image_path)
        if len(self.train_image_paths) <= self.batch_size:
            print(f'image count({len(self.train_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)

        self.validation_image_paths = self.init_image_paths(self.validation_image_path)
        if len(self.validation_image_paths) <= self.batch_size:
            print(f'image count({len(self.validation_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)

        if model_path != '':
            self.ae, self.encoder, self.input_shape = self.load_model(model_path)
        else:
            self.ae, self.encoder = Model(input_shape=self.input_shape, latent_dim=self.latent_dim).build()
        self.data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size)

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def load_model(self, model_path):
        if not (os.path.exists(model_path) and os.path.isfile(model_path)):
            print(f'file not found : {model_path}')
            exit(0)
        model = tf.keras.models.load_model(model_path, compile=False)
        input_shape = model.input_shape[1:]

        ae = None
        encoder = None
        try:
            encoder_output_layer = model.get_layer(name='encoder_output').output
            encoder = tf.keras.models.Model(model.input, encoder_output_layer)
        except ValueError:
            print(f'no encoder output layer found. encoder extraction skip')
        try:
            decoder_output_layer = model.get_layer(name='decoder_output').output
            ae = tf.keras.models.Model(model.input, decoder_output_layer)
        except ValueError:
            print(f'no decoder output layer found in model {model_path}')
            exit(0)
        return ae, encoder, input_shape

    @tf.function
    def compute_gradient(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(x - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def predict(self, img, input_image_concat=True):
        x = DataGenerator.preprocess(img).reshape((1,) + self.input_shape)
        y = np.array(self.graph_forward(self.ae, x))[0]
        img_reconstructed = self.postprocess(y)
        if input_image_concat:
            img_reconstructed = np.concatenate((img, img_reconstructed), axis=1)
        return img_reconstructed

    def predict_images(self, model_path='', image_path='', dataset='validation'):
        self.init(model_path=model_path)
        image_paths = []
        if image_path != '':
            if not os.path.exists(image_path):
                print(f'image path not found : {image_path}')
                return
            if os.path.isdir(image_path):
                image_paths = glob(f'{image_path}/*.jpg')
            else:
                image_paths = [image_path]
        else:
            assert dataset in ['train', 'validation']
            if dataset == 'train':
                image_paths = self.train_image_paths
            else:
                image_paths = self.validation_image_paths

        if len(image_paths) == 0:
            print(f'no images found')
            return

        for path in image_paths:
            img = self.data_generator.load_image(path)
            img_reconstructed = self.predict(img)
            cv2.imshow('img_reconstructed', img_reconstructed)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)

    def postprocess(self, y):
        img = np.asarray(np.clip((y * 255.0), 0.0, 255.0)).astype('uint8').reshape(self.input_shape)
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def train(self, model_path=''):
        self.init(model_path=model_path)
        if self.encoder is not None:
            self.encoder.summary()
            print()
        self.ae.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        os.makedirs(self.checkpoint_path, exist_ok=True)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        while True:
            for x in self.data_generator:
                lr_scheduler.update(optimizer, iteration_count)
                loss = self.compute_gradient(self.ae, optimizer, x)
                iteration_count += 1
                print(f'\r[iteration_count : {iteration_count:6d}] loss : {loss:>8.4f}', end='')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % self.save_interval == 0:
                    self.ae.save(f'{self.checkpoint_path}/ae_{iteration_count}_iter.h5', include_optimizer=False)
                    self.encoder.save(f'{self.checkpoint_path}/encoder_{iteration_count}_iter.h5', include_optimizer=False)
                if iteration_count == self.iterations:
                    print('\ntrain end successfully')
                    return

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            img_path = np.random.choice(self.validation_image_paths)
            img = self.data_generator.load_image(img_path)
            img_reconstructed = self.predict(img)
            cv2.imshow('training view', img_reconstructed)
            cv2.waitKey(1)

