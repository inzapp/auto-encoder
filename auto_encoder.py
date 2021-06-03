import os
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob
from random import randrange
from time import time

import cv2
import numpy as np
import tensorflow as tf

from generator import AutoEncoderDataGenerator
from model import Model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class LearningRateSchedulingCallback(tf.keras.callbacks.Callback):
    def __init__(self, lr, epochs):
        self.lr = lr
        self.epochs = epochs
        self.decay_step = epochs / 10
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if epoch != 1 and (epoch - 1) % self.decay_step == 0:
            self.lr *= 0.5
            tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)


class AutoEncoder:
    def __init__(
            self,
            train_image_path,
            input_shape,
            encoding_dim,
            lr,
            epochs,
            batch_size,
            validation_image_path=''):
        self.pool = ThreadPoolExecutor(8)
        self.train_image_paths = glob(rf'{train_image_path}/*.jpg')
        self.input_shape = input_shape
        self.validation_image_paths = []
        if validation_image_path != '':
            self.validation_image_paths = glob(rf'{validation_image_path}/*.jpg')
        self.encoding_dim = encoding_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_type = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        self.model = Model(
            input_shape=self.input_shape,
            encoding_dim=self.encoding_dim)
        self.train_data_generator = AutoEncoderDataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            encoding_dim=self.encoding_dim,
            batch_size=self.batch_size,
            img_type=self.img_type)
        self.validation_data_generator = AutoEncoderDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            encoding_dim=self.encoding_dim,
            batch_size=self.batch_size,
            img_type=self.img_type)

    def fit(self):

        @tf.function
        def predict_on_graph(model, x):
            return model(x, training=False)

        def training_view(batch, logs):
            global live_view_previous_time
            cur_time = time()
            if cur_time - live_view_previous_time > 0.5:
                live_view_previous_time = cur_time
                __x = cv2.imread(self.train_image_paths[randrange(0, len(self.train_image_paths))], self.img_type)
                __x = cv2.resize(__x, (self.input_shape[1], self.input_shape[0]))
                __x = np.asarray(__x).reshape((1,) + self.input_shape) / 255.0
                __y = predict_on_graph(self.model.ae, __x)
                __x = np.clip(__x * 255.0, 0, 255).astype('uint8').reshape(self.input_shape)
                __y = np.clip(__y * 255.0, 0, 255).astype('uint8').reshape(self.input_shape)
                __x = cv2.resize(__x, (128, 128), interpolation=cv2.INTER_LINEAR)
                __y = cv2.resize(__y, (128, 128), interpolation=cv2.INTER_LINEAR)
                cv2.imshow('ae', np.concatenate((__x, __y), axis=1))
                cv2.waitKey(1)

        self.model.ae.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss=tf.keras.losses.MeanSquaredError())
        self.model.ae.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples')

        if not (os.path.exists('checkpoints') and os.path.isdir('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)
        callbacks = [
            LearningRateSchedulingCallback(self.lr, self.epochs),
            tf.keras.callbacks.LambdaCallback(on_batch_end=training_view),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/ae_epoch_{epoch}_loss_{loss:.4f}_val_loss_{val_loss:.4f}.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True)]

        if len(self.validation_image_paths) > 0:
            print(f'validate on {len(self.validation_image_paths)} samples')
            self.model.ae.fit(
                x=self.train_data_generator,
                validation_data=self.validation_data_generator,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        else:
            self.model.ae.fit(
                x=self.train_data_generator,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        cv2.destroyAllWindows()

    @staticmethod
    def extract_encoder_from_ae(pretrained_ae_path):
        pretrained_ae = tf.keras.models.load_model(pretrained_ae_path, compile=False)
        encoder = tf.keras.models.Sequential()
        for layer in pretrained_ae.layers:
            encoder.add(layer)
            if layer.name == 'encoder_output':
                break
        ae_file_name = pretrained_ae_path.replace('\\', '/').split('/')[-1]
        encoder.save(f'encoder_{ae_file_name[3:]}')
