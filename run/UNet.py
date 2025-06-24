##
## EPITECH PROJECT, 2025
## Unet
## File description:
## Unet
##

import numpy as np
from keras import Input, layers, models
import keras
import tensorflow as tf
from data import paired_data_generator, resize_image
import cv2

def unet_model(input_size=(180, 320, 3)):
    inputs = Input(input_size)

    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    conv1 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(pool2)
    conv3 = layers.Conv2D(256, 3, activation="relu", padding="same")(conv3)

    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    up4 = layers.Conv2D(128, 2, activation="relu", padding="same")(up4)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(merge4)
    conv4 = layers.Conv2D(128, 3, activation="relu", padding="same")(conv4)

    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    up5 = layers.Conv2D(64, 2, activation="relu", padding="same")(up5)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(merge5)
    conv5 = layers.Conv2D(64, 3, activation="relu", padding="same")(conv5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


class UNetDetector:
    def __init__(self, input_size=(180, 320, 3)):
        self.input_size = input_size
        self.model = unet_model(input_size)
        def dice_coef(y_true, y_pred, smooth=1):
            y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
            y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
            intersection = tf.reduce_sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (
                tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
            )

        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", dice_coef]
        )

    def train(self, input_paths, mask_paths, epochs=20, batch_size=4):
        steps_per_epoch = int(np.ceil(len(input_paths) / batch_size))
        generator = paired_data_generator(input_paths, mask_paths, self.input_size[:2], batch_size)

        self.model.fit(
            generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1
        )


    def predict(self, image):
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        img = resize_image(image, (self.input_size[0], self.input_size[1]))
        img = np.expand_dims(img / 255.0, axis=0)

        pred = self.model.predict(img, verbose=0)[0, :, :, 0]
        return (pred > 0.5).astype(np.uint8) * 255

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
