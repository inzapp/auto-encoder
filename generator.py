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
import cv2
import numpy as np
import albumentations as A

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 image_paths,
                 input_shape,
                 batch_size,
                 dtype='float32'):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(image_paths)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
            A.GaussianBlur(p=0.5, blur_limit=(5, 5))
        ])

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        batch_x = []
        for f in fs:
            img = f.result()
            batch_x.append(self.preprocess(np.asarray(img).reshape(self.input_shape)))
        batch_x = np.asarray(batch_x).astype(self.dtype)
        return batch_x

    @staticmethod
    def preprocess(img):
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.clip(np.asarray(img).astype('float32') / 255.0, 0.0, 1.0)

    @staticmethod
    def get_z_vector(size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def resize(self, img, size):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if size[0] > img_width or size[1] > img_height:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        return cv2.resize(img, size, interpolation=interpolation)

    def load_image(self, image_path):
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if self.input_shape[-1] == 1 else cv2.IMREAD_COLOR)
        img = self.resize(img, (self.input_shape[1], self.input_shape[0]))
        img = np.asarray(img).reshape(self.input_shape)
        return img

