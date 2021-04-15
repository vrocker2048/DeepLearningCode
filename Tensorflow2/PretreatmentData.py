import tensorflow as tf
import numpy as np

# (train_data, train_label), (_, _) = tf.keras.datasets.mnist.load_data()
# train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
# mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
#
#
# def rot90(image, label):
#     image = tf.image.rot90(image)
#     return image, label
#
#
# mnist = mnist_dataset.map(rot90)

np.random.randint(0, 10, (4, 3))
lis1 = np.array([np.random.randint(0, 10, (4, 3)) for i in range(1000)])
lis2 = np.array([np.random.randint(0, 10) for i in range(1000)])
temp_data = tf.data.Dataset.from_tensor_slices((lis1, lis2)).map(lambda x, y: (x * 100, y)).shuffle(100).batch(64)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["acc"]
)

model.fit(
    temp_data,
    batch_size=64,
    epochs=5
)