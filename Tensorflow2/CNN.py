import tensorflow as tf
import pandas as pd
import numpy as np


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool1 = tf.keras.layers.AvgPool2D(
            pool_size=[2, 2],
            strides=2
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.AvgPool2D(
            pool_size=[2, 2],
            strides=2
        )

