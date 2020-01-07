# Author: Dean
# Email: yangmingnjau@163.com
# Date: 2020/1/7 15:40

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.__version__)

# 1. 用constant表示输入
a = tf.constant(5)
b = tf.constant(3)
e = tf.add(a, b)
d = tf.multiply(a, b)
with tf.Session() as sess:
    output1 = sess.run(e)
    output2 = sess.run(d)
    print(output1)      # 8
    print(output2)      # 15

# 2. 用variable表示输入
a = tf.Variable(5)
b = tf.Variable(3)
c = tf.multiply(a, b)
d = tf.add(a, b)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(c))      # 15
    print(sess.run(d))      # 8

# 3. 输入用向量表示
a = tf.Variable([4, 5, 6])
b = tf.reduce_prod(a)       # tf.reduce_prod()计算一个张量的各个维度上元素的乘积
c = tf.reduce_sum(a)        # tf.reduce_sum()计算一个张量的各个维度上元素的总和
d = tf.reduce_mean(a)       # tf.reduce_mean()计算一个张量的各个维度上元素的平均值
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(b))      # 120
    print(sess.run(c))      # 15
    print(sess.run(d))      # 5

# 4. 用placeholder表示输入
a = tf.placeholder(tf.int8, shape=[None])
b = tf.reduce_prod(a)
c = tf.reduce_mean(a)
d = tf.reduce_sum(a)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(b, feed_dict={a: [3, 4, 5]}))        # 60
    print(sess.run(c, feed_dict={a: [3, 4, 5]}))        # 4
    print(sess.run(d, feed_dict={a: [4, 4, 5]}))        # 13
# feed_dict用于覆盖图中的Tensor值，这需要一个Python字典对象作为输入，字典中的key值将被value值覆盖




























