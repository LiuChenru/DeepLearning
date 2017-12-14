# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:54:07 2017

@author: Administrator
"""

import tensorflow as tf
import os
os.chdir("G:/LiuChenru/JianGuo Yun/Code/DeepLearning/")



g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v",[2,1],
                        initializer=tf.zeros_initializer(tf.float32))
    a1 = tf.Variable(1.0,tf.float32)
    a2 = tf.Variable(2.0,tf.float32)
    result = tf.add(a1,a2)
    add1 = tf.assign_add(a1,1)
    a_mat1 = tf.Variable([[2.0,1.0]],tf.float32)
    a_mat2 = tf.Variable([[1.0],[2.0]],tf.float32)

    
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v",[1],
                        initializer=tf.ones_initializer(tf.float32))
    a1 = tf.Variable(1.0,tf.float32)
    add1 = tf.assign_add(a1,1)
    

    
with tf.Session(graph=g1) as sess:
    (sess.run(tf.global_variables_initializer()))
    print(a1.eval())
    sess.run(add1)
    print(a1.eval())
    print(sess.run(result))
    print(sess.run(tf.matmul(a_mat1,a_mat2)))

with tf.Session(graph=g2) as sess:
    (sess.run(tf.global_variables_initializer()))
    print(a1.eval())
    print(tf.GraphKeys.GLOBAL_VARIABLES())


g3 = tf.Graph()
with g3.as_default():
    weights = tf.Variable(tf.truncated_normal([3,3],mean=3,stddev=1.0))
    weights2 = tf.Variable(weights.initialized_value()*2)
    wei_mat = tf.matmul(weights,weights2)

with tf.Session(graph=g3) as sess:
    writer = tf.summary.FileWriter("tempDF/",graph = g3)
    sess.run(tf.global_variables_initializer())
    print(sess.run(weights))
    print(sess.run(weights2))
    print(sess.run(wei_mat))


from numpy.random import RandomState
rdm = RandomState(1)
dataset_size = 128

X = rdm.rand(dataset_size,2)
Y = [[int((x1+x2)<1) for (x1,x2) in X]]
Y = np.array(Y).reshape(128,1)


batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape = [None,2],name="x-input")
y_ = tf.placeholder(tf.float32,shape = [None,1],name="y-input")

a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

cross_entropy = -tf.reduce_mean(
        y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))
    
    steps = 100
    for i in range(steps):
        start = (i*batch_size)
        end = min(start+batch_size,dataset_size)
        
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%10 == 0:
            total_cross_entropy = sess.run(
                    cross_entropy,feed_dict={x:X,y_:Y})
            print(i,total_cross_entropy)


v1 = tf.Variable(tf.random_normal([100,1]),tf.float32)
v2 = tf.Variable(tf.random_normal([100,1]),tf.float32)

z1 = tf.greater(v1,v2)
z2 = 3*(v1-v2)

loss = tf.reduce_mean(
        tf.where(tf.greater(v1,v2),3*(v1-v2),2*(v2-v1)))

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(loss))



global_step = tf.Variable(0)

learning_rate = tf.train.exponential_decay(
        0.1,global_step,100,0.96)

learning_step = tf.train.GradientDescentOptimizer(learning_rate)\
.minimize(cross_entropy,global_step)

tf.contrib.layers.l2_regularizer


import numpy as np
num_sample = 10000
num_variable = 10
A
X = np.random.randn(num_sample,num_variable).astype(np.float32)
Y = np.random.randint(2,size=(num_sample,1)).astype(np.float32)

num_nodes = [15,10,7,3,1]
num_nodes = [15,1]
num_hidden_layer = len(num_nodes)

y_ = tf.placeholder(tf.float32,[None,1])
x = tf.placeholder(tf.float32,[None,num_variable])
cur_layer = x
in_dimension = num_variable
for i in np.arange(num_hidden_layer):
    out_dimension = num_nodes[i]
    weights = tf.Variable(tf.random_normal([in_dimension,out_dimension]),tf.float32)
    b = tf.Variable(tf.random_normal([1,out_dimension]),tf.float32)
    cur_layer = tf.nn.relu(tf.matmul(cur_layer,weights)+b)
    in_dimension = out_dimension
    
loss = tf.reduce_mean(tf.square(y_-cur_layer))
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96)
learning_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(loss,feed_dict={x:X,y_:Y})
    for i in np.arange(100):
        cur_loss,_ = sess.run([loss,learning_step],feed_dict={x:X,y_:Y})


v1 = tf.Variable(0.0,tf.float32)
step = tf.Variable(0,trainable = False)
ema = tf.train.ExponentialMovingAverage(0.99,step)
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1,ema.average(v1)]))
    
    sess.run(tf.assign(v1,10))
    sess.run(maintain_average_op)
    print(sess.run([v1,ema.average(v1)]))

    sess.run(tf.assign(step,10000))
    sess.run(tf.assign(v1,10))
    sess.run(maintain_average_op)
    print([v1,ema.average(v1)])









