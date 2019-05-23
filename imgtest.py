import tensorflow as tf
import numpy as np

# 随机生成100个数据

X_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], X_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, X_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
sess.run(init)

for step in range(0, 301):
    sess.run(train)
    if step % 20 == 0:
        print(step)
        print(sess.run(W))
        print(sess.run(b))

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# sess.run(tf.global_variables_initializer())
