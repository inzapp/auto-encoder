from auto_encoder import AutoEncoder

if __name__ == '__main__':
    AutoEncoder(
        train_image_path=r'.',
        validation_image_path=r'.',
        input_shape=(64, 64, 1),
        encoding_dim=32,
        lr=1e-3,
        epochs=100,
        batch_size=32).fit()
