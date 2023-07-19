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
import argparse

from auto_encoder import AutoEncoder


if __name__ == '__main__':
    auto_encoder = AutoEncoder(
        input_shape=(32, 32, 1),
        train_image_path=r'/train_data/imagenet/train',
        validation_image_path=r'/train_data/imagenet/validation',
        lr=0.001,
        warm_up=0.5,
        batch_size=8,
        latent_dim=32,
        iterations=100000,
        save_interval=5000,
        denoising_model=False,
        training_view=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='', help='pretrained model path')
    parser.add_argument('--predict', action='store_true', help='evaluate using train or validation dataset')
    parser.add_argument('--dataset', type=str, default='validation', help='dataset for evaluate, train or validation available')
    parser.add_argument('--path', type=str, default='', help='image or video path for evaluate')
    args = parser.parse_args()
    if args.predict:
        auto_encoder.predict_images(model_path=args.model, image_path=args.path, dataset=args.dataset)
    else:
        auto_encoder.train(model_path=args.model)

