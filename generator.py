from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow as tf


class AutoEncoderDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, input_shape, encoding_dim, batch_size, img_type):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.encoding_dim = encoding_dim
        self.batch_size = batch_size
        self.img_type = img_type
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size
        fs = []
        for i in range(start_index, start_index + self.batch_size):
            cur_img_path = self.image_paths[self.random_indexes[i]]
            fs.append(self.pool.submit(self.__load_image, cur_img_path))

        for f in fs:
            x = f.result()
            x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)
            batch_y.append(x)
        return np.asarray(batch_x), np.asarray(batch_y)

    def __load_image(self, image_path):
        return cv2.imread(image_path, self.img_type)

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)
