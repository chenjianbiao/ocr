import  tensorflow as tf
mnist=tf.keras.datasets.mnist
import numpy as np
data=np.random.random_sample((1000,32))
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train/255.0,x_test/255.0
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                  tf.keras.layers.Dense(512,activation=tf.nn.relu),
                                  tf.keras.layers.Dropout(0.2),
                                  tf.keras.layers.Dense(10,activation=tf.nn.softmax)
                                  ])
model.compile(optimizer='adam',loss=tf.losses.sparse_softmax_cross_entropy,metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)
model.evaluate(x_test,y_test)