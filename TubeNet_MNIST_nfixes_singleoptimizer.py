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
iters = 1000
disp_n_iters = 100

import numpy as np
import tensorflow as tf
import fixeye_saccade as fe

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph input
sess = tf.Session()
X = tf.placeholder("float", [1,784]) #200 inputs
Y = tf.placeholder("float", [1,10]) #10 output calsses

# initialize tensorflow model trainable variables
topweights = tf.Variable(tf.random_normal([200, 110])) # set to 110 for passing new fixes
topbiases = tf.Variable(tf.random_normal([110])) # set to 110 for passing new fixes
memstate = (tf.Variable(tf.random_normal([1, 200])),)*2 # ititial hidden layer c_state and m_state
#seqfixwithimg = tf.Variable(tf.random_normal([1, nfixes, 200]))
lstm_cell = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True, reuse=None)
tf.nn.dynamic_rnn(lstm_cell, tf.Variable(tf.random_normal([1, nfixes, 200])), dtype=tf.float32)

    
# run individual passes
def runnfixes(MNISTimg, memstate):
    nfixwithimg_list = []
    fix_list = []
    fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([[44]],[1.0],[100])))
    fix = tf.reshape(fix_list,[-1,100])
    for n in range(nfixes-1):
        # get new series of fixations by feeding forward for n fixations
        FEimg = tf.py_func(fe.fixeye,(MNISTimg,fix[n,:]),'float') 
        nfixwithimg_list.append(tf.concat((FEimg,fix[n,:]),0))
        nfixwithimg = tf.reshape(nfixwithimg_list,[-1,200])
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            osingle, newmemstate = lstm_cell(tf.reshape(nfixwithimg[n,:], [1,200]), memstate)
        outsingle = tf.matmul(osingle, topweights) + topbiases
        memstate = newmemstate
        newfixind = tf.cast(tf.argmax(outsingle[0,10:]),tf.int64)
        fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([[newfixind]],[1],[100])))
        fix = tf.reshape(nfixwithimg_list,[-1,100])
    return tf.reshape(nfixwithimg,[1,nfixes,200]), fix, memstate
seqfixwithimg, seqfix, memstate = runnfixes(X, memstate)

# full sequence pass (for training with unfolding over time)
with tf.variable_scope('rnn') as scope:
    scope.reuse_variables()
    oseq, memstateseq = tf.nn.dynamic_rnn(lstm_cell, seqfixwithimg, dtype=tf.float32)
outseq = tf.matmul(oseq[:,-1,:], topweights) + topbiases # keep only last in time output
logits = tf.reshape(outseq[0,:10],[1,10]) # MNIST categories
                   
# Define loss and optimizer for MNIST
loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_MNIST = optimizer_MNIST.minimize(loss_MNIST)

# Evaluate model with MNIST (with test logits, for dropout to be disabled)
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction), tf.argmax(Y))
accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess.run(init)
#saver.restore(sess, "/tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")

# Start training
# initialize outer np variables
lossperiter = np.zeros(iters)
accuracyperiter = np.zeros(iters)
for i in range(iters): # iterations
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    lossperiter[i], accuracyperiter[i] = sess.run([loss_MNIST,accuracy_MNIST],
        feed_dict={X: MNISTimg, Y: MNISTcat})
    if i % disp_n_iters == 0 or i == 0:
        # Calculate seq loss and accuracy and see fixations used
        print("Iteration " + str(i) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(lossperiter[i-disp_n_iters:i])) + ", Accuracy " + \
              "{:.4f}".format(np.mean(accuracyperiter[i-disp_n_iters:i])) + ", fixation sequence was:")
#        print(np.where(outerfixes[:,:]==1)[1])
        
    sess.run(train_MNIST, feed_dict={X: MNISTimg, Y: MNISTcat})
    
print("Optimization Finished!")
#saver = tf.train.Saver()
#saver.save(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#train_writer = tf.summary.FileWriter('tmp/TubeNet_MNIST_nfixes_singleoptimizer',sess.graph)

# Calculate accuracy for 128 mnist test images
test_len = 128
finallossperiter = np.zeros(test_len)
finalaccuracyperiter = np.zeros(test_len)
finalfixes = np.zeros([10,test_len])
for i in range(test_len):
    MNISTimg, MNISTcat = mnist.test.next_batch(1)
    finallossperiter[i], finalaccuracyperiter[i] = sess.run([loss_MNIST,accuracy_MNIST],
        feed_dict={X: MNISTimg, Y: MNISTcat})
             
print(", loss_MNIST= " + "{:.4f}".format(np.mean(finallossperiter)) \
      + ", Accuracy " + "{:.4f}".format(np.mean(finalaccuracyperiter)) \
      + ", fixation sequence was:")
#        print(np.where(fix[:,:]==1)[1])


# sess.close()