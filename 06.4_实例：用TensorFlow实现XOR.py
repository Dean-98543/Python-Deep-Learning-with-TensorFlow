# Author: Dean
# Email: yangmingnjau@163.com
# Date: 2020/1/4 17:52

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# tf.set_random_seed(15)
tf.set_random_seed(20)
# tf.set_random_seed(3567)

# 1. 定义输入与目标值
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0],
              [1],
              [1],
              [0]])

# 2. 定义占位符，从输入或目标中按行取数据
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# 3. 初始化权重，使权重满足正态分布
w1 = tf.Variable(tf.random_normal([2, 2]))
w2 = tf.Variable(tf.random_normal([2, 1]))

# 4. 定义偏移量
b1 = tf.Variable([0.1, 0.1])
b2 = tf.Variable(0.1)

# 5. 利用激活函数计算隐含层的输出值
h = tf.nn.relu(tf.matmul(x, w1) + b1)

# 6. 计算输出层的值
out = tf.matmul(h, w2) + b2

# 7. 定义代价函数
loss = tf.reduce_mean(tf.square(out - y))

# 8. 利用Adam自适应优化算法
train = tf.train.AdamOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(4000):
        sess.run(train, feed_dict={x: X, y: Y})
        loss_ = sess.run(loss, feed_dict={x: X, y: Y})
        if i % 200 == 0:
            print("step: %d, loss: %.3f" % (i, loss_))
    print("X: %r" % X)
    print("pred: %r" % sess.run(out, feed_dict={x: X}))

'''
无法复现实验结果，即使设置了随机种子，也无法得到类似的结果
在设置随机数种子# tf.set_random_seed(15)时，可以得到：
pred: array([[0.],
       [1.],
       [1.],
       [0.]], dtype=float32)
但此结果和例子中的结果相差比较大，无法解释如此准确的结果
2020-01-08 21:53
'''

































