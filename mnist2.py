import tensorflow as tf
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import time

# mnist = keras.datasets.mnist

# model_name = "mnist{}".format(str(time.time()))
model_name = "mnist_log_1"
tensorboard=TensorBoard(log_dir='logs/{}'.format(model_name))
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation="softmax")
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=64,callbacks=[tensorboard])
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_loss)
print(test_acc)
