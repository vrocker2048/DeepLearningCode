import tensorflow as tf
import numpy as np
import os
import sys

cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "dog_cat_data")
train_data_dir = os.path.join(data_dir, "train")
valid_data_dir = os.path.join(data_dir, "valid")
train_dogs_data_dir = os.path.join(train_data_dir, "dogs/")
train_cats_data_dir = os.path.join(train_data_dir, "cats/")
valid_dogs_data_dir = os.path.join(valid_data_dir, "dogs/")
valid_cats_data_dir = os.path.join(valid_data_dir, "cats/")
tfrecord_file = os.path.join(train_data_dir, "train.tfrecords")

train_cat_filenames = [train_cats_data_dir + filename for filename in os.listdir(train_cats_data_dir)]
train_dog_filenames = [train_dogs_data_dir + filename for filename in os.listdir(train_dogs_data_dir)]
train_filenames = train_cat_filenames + train_dog_filenames
train_labels = [0] * len(train_cat_filenames) + [1] * len(train_dog_filenames)

with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for filename, label in zip(train_filenames, train_labels):
        print(filename)
        print(type(filename))
        image = open(filename, 'rb').read()
        feature = {
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
