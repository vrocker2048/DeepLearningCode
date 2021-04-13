import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# keras 的序列模型
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(100, activation=tf.nn.relu),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Softmax()
# ])

dataset = tf.keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = dataset.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
test_data = np.expand_dims(test_data.astype(np.float32) / 255.0, axis=-1)
print(train_data.shape)
train_label = train_label.astype(np.int32)
test_label = test_label.astype(np.int32)

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)(x)
x = tf.keras.layers.Softmax()(x)
outputs = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_crossentropy, 'acc']
)

model.fit(
    x=train_data,
    y=train_label,
    batch_size=64,
    epochs=100
)


