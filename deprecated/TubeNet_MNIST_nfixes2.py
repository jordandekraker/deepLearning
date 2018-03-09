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
import numpy as np
import tensorflow as tf
import fixeye_saccade as fe
import matplotlib.pyplot as plt


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph input
X = tf.placeholder("float", [1,10,200]) #200 inputs
Y = tf.placeholder("float", [1,10]) #10 output calsses

# initialize hard coded variables                 
fix = np.zeros([1,10,100]) 
fix[0,0,55] = 1; # initial fixation location
loss = np.zeros([10]) # loss value for each 
acc_MNIST = np.zeros([10]) # 10 MNIST outputs
newfixind = np.zeros([10]) # 10 MNIST outputs
FEimg = (np.zeros([1,10,100])) # fix image size

# initialize tensorflow model trainable variables
memstate = (tf.Variable(tf.random_normal([1, 200])),)*2 # ititial hidden layer c_state and m_state
memstateseq = (tf.Variable(tf.random_normal([1, 200])),)*2 # ititial hidden layer c_state and m_state
topweights = tf.Variable(tf.random_normal([200, 110])) # 200 input units; 110 outputs
topbiases = tf.Variable(tf.random_normal([110])) #110 outputs
with tf.variable_scope("rnn"):
    with tf.variable_scope("basic_lstm_cell"):
        weights = tf.get_variable('weights', [400, 800], dtype=tf.float32, 
                                  initializer=tf.random_normal_initializer())
        biases = tf.get_variable('biases', [800, ],  dtype=tf.float32, 
                                 initializer=tf.random_normal_initializer())

#build the network
lstm_cell = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True, reuse=True)
oseq, memstateseq = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
outseq = tf.matmul(oseq[-1], topweights) + topbiases
logits = outseq[0,:10] #only part of the output we care abnout

# Define loss and optimizer for MNIST
loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)
grads_MNIST = optimizer_MNIST.compute_gradients(loss_MNIST)
train_MNIST = optimizer_MNIST.minimize(loss_MNIST)

# define loss and optimizer for new FIX
#loss_FIX = tf.reduce_mean(acc_MNIST)
#optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#train_FIX = optimizer_FIX.minimize(loss_FIX, grad_loss = grads_MNIST)

# Evaluate model with MNIST (with test logits, for dropout to be disabled)
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction), tf.argmax(Y))
accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
# Run the initializer
sess.run(init)
# Start training
for iters in range(100): # iterations
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    MNISTimg = MNISTimg+0.001 # otherwise can lead to cropping problems in fixeye
    for n in range(10):
        # get new series of fixations bet feeding forward for n fixations
        FEimg[0,n,:] = fe.fixeye(MNISTimg,fix[0,n,:].reshape([1,100]))
        
        t = np.concatenate((FEimg[0,n,:].reshape([1,100]),fix[0,n,:].reshape([1,100])),1)
        
        with tf.variable_scope("rnn"):
            osingle, newmemstate = lstm_cell(tf.convert_to_tensor(t,dtype=tf.float32), memstate)
        outsingle = tf.matmul(osingle, topweights) + topbiases
        memstate = newmemstate
        newfixind[n] = sess.run(tf.argmax(outsingle[0,10:]))
        m=n+1
        if n==9:
            m=0
        fix[0,m,:] = np.zeros([1,100])
        fix[0,m,newfixind[n].astype(int)] = 1
        
    # now train using data from those n fixations
    sess.run(train_MNIST, feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    #    sess.run(train_FIX, feed_dict={X: np.concatenate((FEimg,fix),1),
    #                                   Y: })

    
    
    if iters % 20 == 0 or iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        loss, acc_MNIST = sess.run([loss_MNIST,accuracy_MNIST],
                feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    
        print("Iteration " + str(iters) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(loss)) + ", Accuracy " + \
              "{:.4f}".format(np.mean(acc_MNIST)) + ", fixation sequence was:")
        print(newfixind.astype(int))

# sess.close()

print("Optimization Finished!")


# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len]
test_label = mnist.test.labels[:test_len]
finalaccuracy = np.zeros(test_len)
testx = np.zeros([1,784])
testy = np.zeros([1,10])
for t in range(test_len):
    n=0
    while n<10:
        # Apply fisheye filter and reshape data
        testx[0,:] = test_data[n,:]
        testy[0,:] = test_label[n,:]
        FEimg = fe.fixeye(testx,fix)
        # Return values of interest
        loss, acc_MNIST, nfixind = sess.run([loss_MNIST,accuracy_MNIST, 
            newfixind], feed_dict={X: np.concatenate((FEimg,fix),1), Y: testy})
        # Return new fixation
        fix = np.zeros([1,100])
        fix[0,nfixind[n].astype(int)] = 1
        n+=1
    finalaccuracy[t] = acc_MNIST[n-1]
print("Testing Accuracy:", "{:.4f}".format(np.mean(finalaccuracy)), 
      ", final fixation sequence was:")
print(nfixind.astype(int))