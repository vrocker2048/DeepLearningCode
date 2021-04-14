import tensorflow as tf
import numpy as np

dataset = tf.keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = dataset.load_data()
train_data = np.expand_dims(train_data.astype(np.float32) / 255.0, axis=-1)
test_data = np.expand_dims(test_data.astype(np.float32) / 255.0, axis=-1)
train_label = train_label.astype(np.int32)
test_label = test_label.astype(np.int32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=[3, 3],
        padding='same',
        activation=tf.nn.relu,
        input_shape=[28, 28, 1]
    ),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.layers.Softmax())
])

check_point = tf.train.Checkpoint(myModel=model)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["acc"]
)

model.fit(
    x=train_data,
    y=train_label,
    batch_size=128,
    epochs=5
)

path = check_point.save("./check_point/model.ckpt")
print("save to %s" % path)
