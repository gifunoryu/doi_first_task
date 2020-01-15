import argparse

import numpy as np

import tensorflow as tf
keras = tf.keras

from datasets import directory_dataset
import transforms
from utils.config import yaml2attrdict

def main():
    # load configures
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path of config file')
    args = parser.parse_args()

    config = yaml2attrdict(args.config)

    # define preprocess
    train_transforms = transforms.Compose([
        transforms.read_img,
        transforms.Resize(cfg.size),
        tf.image.random_flip_left_right,
        transforms.Normalize(0, 255)
        ])
    test_transforms = transforms.Compose([
        transforms.read_img,
        transforms.Resize(cfg.size),
        transforms.Normalize(0, 255)
        ])

    # load dataset
    (train_dataset, val_dataset, test_dataset), \
            (train_image_count, val_image_count, test_image_count) = \
            directory_dataset(
                    cfg.data_dir,
                    train_transform, test_transform,
                    split = (8, 1, 1)
                    )

    # define datasets
    train_dataset = train_dataset.shuffle(buffer_size=train_image_count)
    train_dataset = train_dataset.repeat().batch(cfg.batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=val_image_count)
    val_dataset = val_dataset.repeat().batch(cfg.batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=test_image_count)
    test_dataset = test_dataset.repeat().batch(cfg.batch_size)

    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top = False, weights = 'imagenet')
    base_model.trainable = False
    print('*' * 40)
    print('Visualize the Architecture of the Base Model')
    print('*' * 40)
    base_model.summary() # visualize the model architecture

    # create whole model
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(1)
    model = keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
        ])

    # train the model
    model.compile(
            optimizer = keras.optimizers.RMSprop(lr=base_learning_rate),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
            )
    print('*' * 40)
    print('Visualize the Whole Model')
    print('*' * 40)
    model.summary()

    history = model.fit(
            train_batches,
            epochs = initial_epochs,
            validation_data = validation_batches
            )
    model.save('../data/dog_cat/saved_model.h5')

if __name__ == '__main__':
    main()
