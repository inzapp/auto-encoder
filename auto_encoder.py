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
from lr_scheduler import LRScheduler
from generator import DataGenerator


class AutoEncoder:
    def __init__(self,
                 train_image_path,
                 validation_image_path,
                 input_shape,
                 lr,
                 warm_up,
                 batch_size,
                 latent_dim,
                 iterations,
                 save_interval,
                 strided_model,
                 denoising_model,
                 training_view,
                 unet,
                 checkpoint_path='checkpoint'):
        assert input_shape[2] in [1, 3]
        assert save_interval >= 1000
        self.input_shape = input_shape
        self.lr = lr
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval
        self.iterations = iterations
        self.training_view = training_view
        self.denoising_model = denoising_model
        self.unet = unet
        self.checkpoint_path = checkpoint_path
        self.live_view_previous_time = time()

        if not self.is_valid_path(train_image_path):
            print(f'train image path is not valid : {train_image_path}')
            exit(0)
        if not self.is_valid_path(validation_image_path):
            print(f'validation image path is not valid : {validation_image_path}')
            exit(0)

        self.train_image_paths = self.init_image_paths(train_image_path)
        if len(self.train_image_paths) <= self.batch_size:
            print(f'image count({len(train_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)
        self.validation_image_paths = self.init_image_paths(validation_image_path)
        if len(self.validation_image_paths) <= self.batch_size:
            print(f'image count({len(validation_image_path)}) is lower than batch size({self.batch_size})')
            exit(0)

        self.model = Model(
            input_shape=input_shape,
            latent_dim=self.latent_dim,
            strided_model=strided_model,
            denoising_model=denoising_model,
            unet=unet)
        self.encoder, self.ae = self.model.build()
        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            input_shape=input_shape,
            batch_size=batch_size,
            denoising_model=denoising_model)

    def is_valid_path(self, path):
        return os.path.exists(path) and os.path.isdir(path)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    def fit(self):
        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        os.makedirs(self.checkpoint_path, exist_ok=True)
        optimizer = tf.keras.optimizers.RMSprop(lr=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=self.warm_up, policy='step')
        while True:
            for batch_x, batch_y in self.train_data_generator:
                lr_scheduler.update(optimizer, iteration_count)
                loss = self.compute_gradient(self.ae, optimizer, batch_x, batch_y)
                iteration_count += 1
                print(f'\r[iteration_count : {iteration_count:6d}] loss : {loss:>8.4f}', end='')
                if self.training_view:
                    self.training_view_function()
                if iteration_count % self.save_interval == 0:
                    if self.denoising_model:
                        self.ae.save(f'{self.checkpoint_path}/denoising_ae_{iteration_count}_iter.h5', include_optimizer=False)
                    else:
                        self.encoder.save(f'{self.checkpoint_path}/encoder_{iteration_count}_iter.h5', include_optimizer=False)
                if iteration_count == self.iterations:
                    print('\ntrain end successfully')
                    return

    @tf.function
    def graph_forward(self, model, x):
        return model(x, training=False)

    def resize(self, img, size):
        if img.shape[1] > size[0] or img.shape[0] > size[1]:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        else:
            return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 0.5:
            self.live_view_previous_time = cur_time
            img_path = np.random.choice(self.validation_image_paths)
            img, img_noise = self.train_data_generator.load_image(img_path)
            if self.denoising_model:
                img = img_noise
            input_shape = self.ae.input_shape[1:]
            x = DataGenerator.normalize(img).reshape((1,) + input_shape)
            decoded_image = DataGenerator.denormalize(self.graph_forward(self.ae, x))
            view_img = np.concatenate((img.reshape(input_shape), decoded_image.reshape(input_shape)), axis=1)
            cv2.imshow('training view', view_img)
            cv2.waitKey(1)

