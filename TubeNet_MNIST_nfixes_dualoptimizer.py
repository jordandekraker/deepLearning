#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for classifying MNIST images. 
The goal of this architecture is to have sensory, memory, and motor processing 
be fully integrated at the most basic level possible, similar to a primitive 
organism (inspired by the ontogeny/phylogeny structure of the 'neural tube'). 
Added complexity can then be added via additional units, layers, or additional
units or layers nested in the LSTM hidden layer.

@author: jordandekraker
"""

nfixes = 5
iters = 10000
disp_n_iters = 100

import numpy as np
import tensorflow as tf
import fixeye_saccade as fe
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph setup
X = tf.placeholder("float", [1,784]) #size of an MNIST image
Y = tf.placeholder("float", [1,10]) #number of possible classes

# initialize tensorflow trainable variables
topweights = tf.Variable(tf.random_normal([200, 110])) # set to 110 for passing new fixes
topbiases = tf.Variable(tf.random_normal([110])) # set to 110 for passing new fixes
with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        weights = tf.get_variable('kernel',[400, 800])
        biases = tf.get_variable('bias',[800]) 
        # NOTE: these two variables are what tf.contrib.rnn.BasicLSTMCell would create by default
lstm_cell = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True, reuse=True)
memstate = (tf.Variable(tf.random_normal([1, 200])),)*2 # ititial hidden layer c_state and m_state 

def runsinglefix(fixwithimg,memstate):
    with tf.variable_scope('rnn') as scope:
        scope.reuse_variables()
        osingle, newmemstate = lstm_cell(tf.reshape(fixwithimg,[1,200]), memstate)
    outsingle = tf.matmul(osingle, topweights) + topbiases # linear activation function
    memdiff_m = tf.reduce_mean(tf.square(newmemstate[1]-memstate[1]),1)
    newfixind = tf.cast(tf.argmax(outsingle[0,10:]),tf.int64)
    
    # train for fixes that increase memdiff_m
    Q = outsingle[0,10:]
    Qtarget = Q + tf.multiply(Q,
          tf.sparse_tensor_to_dense(tf.SparseTensor([[newfixind]],memdiff_m,[100])))
    loss_FIX = tf.reduce_sum(tf.square(Qtarget - Q))
    optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.001) 
    optimizer_FIX.minimize(loss_FIX,var_list=[topweights,topbiases])
    return outsingle, newmemstate, newfixind

# define the model
def runnfixes(image,memstate):
    fix_list = []
    fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([[55]],[1.0],[100]))) #starting fixation
    fix = tf.reshape(fix_list,[-1,100])
    # get new series of fixations by feeding forward for n fixations
    for n in range(nfixes):
        if n < nfixes-1: #dont do this for last fixation
            FEimg = tf.cast(tf.py_func(fe.fixeye,[image, fix[n,:]],'double'),tf.float32) #fixated image at fix
            outsingle, newmemstate, newfixind = runsinglefix(tf.concat((FEimg,fix[n,:]),0), memstate)
            memstate = newmemstate

            #regroup for next fix
            fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([[newfixind]],[1.0],[100])))
            fix = tf.reshape(fix_list,[-1,100])
    return fix, memstate, outsingle
seqfix,memstate,finalout = runnfixes(X,memstate)


# Define loss and optimizer for MNIST
logits = tf.reshape(finalout[0,:10],[1,10]) # MNIST categories
loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_MNIST = optimizer_MNIST.minimize(loss_MNIST)

# Evaluate model with MNIST (with test logits, for dropout to be disabled)
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")

# Start training
# initialize outer np variables
lossperiter = np.zeros(iters)
accuracyperiter = np.zeros(iters)
for i in range(iters): # iterations
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    lossperiter[i],accuracyperiter[i],finalfix = sess.run(
            [loss_MNIST,accuracy_MNIST,seqfix], 
            feed_dict={X: MNISTimg, Y: MNISTcat})
    if i % disp_n_iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        print("Iteration " + str(i) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(lossperiter[i-disp_n_iters:i])) + ", Accuracy= " + \
              "{:.4f}".format(np.mean(accuracyperiter[i-disp_n_iters:i])) + ", fixation sequence was:")
        print(np.where(finalfix[:,:]==1)[1])
    sess.run(train_MNIST, feed_dict={X: MNISTimg, Y: MNISTcat})
    
print("Optimization Finished!")
#saver.save(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#train_writer = tf.summary.FileWriter('tmp/TubeNet_MNIST_nfixes_singleoptimizer',sess.graph)

# Calculate accuracy for 128 mnist test images
test_len = 128
finallossperiter = np.zeros(test_len)
finalaccuracyperiter = np.zeros(test_len)
for i in range(test_len):
    MNISTimg, MNISTcat = mnist.test.next_batch(1)
    finallossperiter[i], finalaccuracyperiter[i], outerfixes = sess.run(
            [loss_MNIST,accuracy_MNIST,seqfix],
            feed_dict={X: MNISTimg, Y: MNISTcat})
             
print("Test mean loss_MNIST= " + "{:.4f}".format(np.mean(finallossperiter)) \
      + ", Accuracy= " + "{:.4f}".format(np.mean(finalaccuracyperiter)) \
      + ", fixation sequence was:")
print(np.where(outerfixes[:,:]==1)[1])

meanaccperiter = np.zeros(int(iters/disp_n_iters))
for n in range(int(iters/disp_n_iters)):
    meanaccperiter[n] = np.mean(accuracyperiter[n*disp_n_iters : (n+1)*disp_n_iters])
plt.plot(meanaccperiter)

sess.close()