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
from data import paired_data_generator, resize_image, img_height, img_width
import cv2

from postProcess import PostProcessor

def conv_block(x, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def unet_model(input_size=(img_width, img_height, 3)):
    inputs = Input(input_size)

    conv1 = conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256, dropout_rate=0.3)

    up4 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv3)
    merge4 = layers.concatenate([conv2, up4], axis=3)
    conv4 = conv_block(merge4, 128)

    up5 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv4)
    merge5 = layers.concatenate([conv1, up5], axis=3)
    conv5 = conv_block(merge5, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(conv5)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


class UNetDetector:
    def __init__(self, input_size=(img_height, img_width, 3)):
        self.input_size = input_size
        self.model = unet_model(input_size)
        self.post_processor = PostProcessor()
        
        def dice_coef(y_true, y_pred, smooth=1):
            y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
            y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
            )
            
        def dice_loss(y_true, y_pred):
            smooth = 1
            y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
            y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
            intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
            return 1 - (2. * intersection + smooth) / (
                tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
            )

        def focal_loss(y_true, y_pred, gamma=2., alpha=.25):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
            cross_entropy = -y_true * tf.keras.backend.log(y_pred) - (1 - y_true) * tf.keras.backend.log(1 - y_pred)
            weight = alpha * tf.keras.backend.pow(1 - y_pred, gamma)
            return tf.keras.backend.mean(weight * cross_entropy)

        def total_loss(y_true, y_pred):
            return focal_loss(y_true=y_true, y_pred=y_pred) + dice_loss(y_true, y_pred)

        self.model.compile(
            optimizer="adam", loss=total_loss, metrics=["accuracy", dice_coef]
        )

    def train(self, input_paths, mask_paths, epochs=20, batch_size=16):
        steps_per_epoch = int(np.ceil(len(input_paths) / batch_size))
        generator = paired_data_generator(input_paths, mask_paths, self.input_size[:2], batch_size)

        self.model.fit(
            generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=1
        )

    def predict(self, image):
        original_shape = image.shape[:2]

        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        img = resize_image(image, (self.input_size[0], self.input_size[1]))
        img = np.expand_dims(img / 255.0, axis=0)

        pred = self.model(img, training=False).numpy()[0, :, :, 0]

        pred = np.tanh(5 * (pred - 0.5)) * 0.5 + 0.5
        
        # Utilisation de cv2.resize au lieu de tf.image.resize pour compatibilité
        pred_resized = cv2.resize(pred, (original_shape[1], original_shape[0]))

        mask = (pred_resized > 0.5).astype(np.uint8) * 255
        mask = self.post_processor.process(mask)
        return mask
        
    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)
