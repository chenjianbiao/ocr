import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

print(tf.VERSION)
print(tf.keras.__version__)

model = tf.keras.Sequential()

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
val_data = np.random.random((1000, 32))
val_labels = np.random.random((1000, 10))
# model.fit(data, labels, batch_size=32, epochs=10,validation_data=(val_data,val_labels))
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()
model.fit(dataset, steps_per_epoch=30, epochs=10)
model.evaluate(data, labels, batch_size=32)
value = model.evaluate(dataset, steps=30)
print(value)
data2 = model.predict(data, batch_size=32)
print(data2.shape)
