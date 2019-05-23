import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

# 加载数据集，返回4个numpy数组
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.show()
train_images = train_images / 255
test_images = test_images / 255
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()
print(type(train_images))
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    # keras.layers.Dense(28,activation=tf.nn.relu,input_shape=(28,28)),
    keras.layers.Dense(28,activation='relu'),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    # keras.layers.Dense(10,activation='relu'),
    # keras.layers.Dropout(0.2),
    # keras.layers.Dense(30,activation=tf.nn.conv2d()),
    # keras.layers.Dense(30,activation=tf.nn.max_pool(value=2,ksize=2,strides=2,padding='same')),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history=model.fit(train_images, train_labels,batch_size=128,validation_split=0.2, epochs=50)

test_loss, test_acc = model.evaluate(test_images, test_labels)

y=history.history['loss']
plt.figure()
plt.title('loss')

sub1=plt.subplot(2,1,1)
sub2=plt.subplot(2,1,2)
l1=sub1.plot(history.history['loss'],label='loss')
l1=sub1.plot(history.history['val_loss'],label='val_loss')

l2=sub2.plot(history.history['acc'],label='acc')
l2=sub2.plot(history.history['val_acc'],label='val_acc')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# plt.title('accuracy')
# plt.plot(history.history['acc'],label='acc')
# plt.plot(history.history['val_acc'],label='val_acc')
# plt.legend('lower right')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.show()
print('Test accuracy', test_acc)
