# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 15:12:56 2017

@author: Administrator
"""
logs_dir = './graphs'
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3) 
x = tf.add(a,b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter(logs_dir,sess.graph)
    print(sess.run(x))
    
    
writer.close()

a = tf.ones([2,3],tf.int32)
b = tf.zeros([3,2],tf.int16) 
c = tf.fill([3,2],8)  
#x = tf.add(c,b) ## error message which mean we can not add two matrix with different int type
x = tf.matmul(a,c)

with tf.Session() as sess:
    print(sess.run([c,x]))
    
tf.range()


tf.truncated_normal