import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "dog_cat_data")
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")
train_dogs_data_dir = os.path.join(train_data_dir, "dogs/")
train_cats_data_dir = os.path.join(train_data_dir, "cats/")
valid_dogs_data_dir = os.path.join(valid_data_dir, "dogs/")
valid_cats_data_dir = os.path.join(valid_data_dir, "cats/")
tfrecord_file = os.path.join(train_data_dir, "train.tfrecords")

raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}


def _parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
    return feature_dict['image'], feature_dict['label']


dataset = raw_dataset.map(_parse_example)

for image, label in dataset:
    plt.title('cat' if label == 0 else 'dog')
    plt.imshow(image.numpy())
    plt.show()
    break
