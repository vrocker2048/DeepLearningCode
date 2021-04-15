import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.layers import *

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "dog_cat_data")
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")
train_dogs_data_dir = os.path.join(train_data_dir, "dogs/")
train_cats_data_dir = os.path.join(train_data_dir, "cats/")
valid_dogs_data_dir = os.path.join(valid_data_dir, "dogs/")
valid_cats_data_dir = os.path.join(valid_data_dir, "cats/")


def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label


train_cat_filenames = tf.constant([train_cats_data_dir + filename for filename in os.listdir(train_cats_data_dir)])
train_dog_filenames = tf.constant([train_dogs_data_dir + filename for filename in os.listdir(train_dogs_data_dir)])
train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
train_labels = tf.concat([
    tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
    tf.ones(train_dog_filenames.shape, dtype=tf.int32)],
    axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
train_dataset = train_dataset.map(
    map_func=_decode_and_resize,
    num_parallel_calls=tf.data.experimental.AUTOTUNE)
# 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
train_dataset = train_dataset.shuffle(buffer_size=23000)
train_dataset = train_dataset.batch(4)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.models.Sequential([
    Conv2D(32, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

model.fit(train_dataset, batch_size=4, epochs=5)
