##
## EPITECH PROJECT, 2025
## MLP
## File description:
## MLP
##

import keras
from keras import layers
import tensorflow as tf

class MLP(keras.Model):
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        self.input_layer = layers.InputLayer(input_shape=(input_size,))
        
        self.hidden_layers = []
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(layers.Dense(
                hidden_size, 
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(1e-4)
            ))
            self.hidden_layers.append(layers.Dropout(0.2))
        
        self.speed_output = layers.Dense(1, activation='sigmoid', name='speed')
        self.steering_output = layers.Dense(1, activation='tanh', name='steering')

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        
        speed = self.speed_output(x)
        steering = self.steering_output(x)
        
        return tf.concat([speed, steering], axis=1)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

