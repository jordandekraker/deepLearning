#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for classifying MNIST images. 
The goal of this architecture is to have sensory processing, memory, and motor 
actions be fully integrated even in a very basic small network, similar to a
primitive organism (inspired by the ontogeny/phylogeny structure of the 'neural 
tube'). Added complexity can then be added via additional units, layers, or 
hidden unit capacities.

@author: jordandekraker
"""

from __future__ import print_function
import fixeye_saccade as fe
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph input
X = tf.placeholder("float", [200]) #200 inputs
Y = tf.placeholder("float", [10]) #10 output calsses

# Define weights
weights = tf.Variable(tf.random_normal([200, 110])) # 200 input units; 110 outputs
biases = tf.Variable(tf.random_normal([110])) #110 outputs
acc_MNIST = tf.Variable(tf.random_normal([10])) # 10 MNIST outputs
fix = np.zeros(100) 
fix[44] = 1; # initial fixation
loss = tf.Variable(tf.random_normal([10])) # loss value for each 
memstate = tf.zeros([1,200]) # ititial hidden layer state

def RNN(x, weights, biases):
    lstm_cell = rnn.BasicLSTMCell(200) # 200 hidden units
    outputs, newmemstate = lstm_cell(tf.reshape(x,[1,200]),memstate)
    # Linear activation, using rnn inner loop last output
    RNNout = tf.matmul(outputs, weights) + biases
    return RNNout, newmemstate

RNNout,newmemstate = RNN(X, weights, biases)
memstate = newmemstate
logits = RNNout[0:9]
prediction = tf.nn.softmax(logits)
fix = np.zeros(100)
fix[tf.argmax(RNNout[10:110])] = 1

# Define loss and optimizer for MNIST
loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_MNIST = optimizer_MNIST.minimize(loss_MNIST)
# define loss and optimizer for new FIX
loss_FIX = tf.reduce_mean(acc_MNIST)
optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_FIX = optimizer_FIX.minimize(loss_FIX)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction), tf.argmax(Y))
accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
# Run the initializer
sess.run(init)
n=0
# Start training
for iters in range(10000): #10000 iterations
    MNISTimg, MNISTcat = mnist.train.next_batch(iters)
    while n<10:
        n+=1
        # Apply fisheye filter and reshape data
        FEimg = fe.fixeye(MNISTimg,fix)
        # Run optimization op (backprop)
        sess.run(train_MNIST, feed_dict={X: np.concatenate((FEimg,fix)), 
                                         Y: MNISTcat})
        loss[n], acc_MNIST[n] = sess.run([loss_MNIST,accuracy_MNIST], 
            feed_dict={X: np.concatenate((FEimg,fix)), Y: MNISTcat})
    sess.run(train_FIX, {X: np.concatenate((FEimg,fix))})
    
    
    if iters % 200 == 0 or iters == 0:
        # Calculate batch loss and accuracy
        print("Iteration " + str(iters) + ", loss_MNIST= " + \
              "{:.4f}".format(loss_MNIST) + "fixation at " + \
              "{:.4f}".format(tf.argmax(fix))) # finds the fixation

# sess.close()

print("Optimization Finished!")
# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len]
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", \
    sess.run(accuracy_MNIST, feed_dict={X: test_data, Y: test_label}))