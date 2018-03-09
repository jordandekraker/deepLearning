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

#from __future__ import print_function
import numpy as np
import tensorflow as tf
import fixeye_saccade as fe

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph input
x = tf.placeholder("float", [1,200]) #200 inputs
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
[pweights, pbiases, ptopweights, ptopbiases] = [weights, biases, topweights, topbiases]
lstm_cell = tf.contrib.rnn.BasicLSTMCell(200, state_is_tuple=True, reuse=True)

# single forward pass (for generating the fixation sequence)
with tf.variable_scope("rnn"):
    osingle, newmemstate = lstm_cell(tf.convert_to_tensor(x,dtype=tf.float32), memstate)
outsingle = tf.matmul(osingle, topweights) + topbiases
memstate = newmemstate
nfi = tf.argmax(outsingle[0,10:])

# full sequence (for training with unfolding over time)
oseq, memstateseq = tf.nn.dynamic_rnn(lstm_cell, X, dtype=tf.float32)
outseq = tf.matmul(oseq[-1], topweights) + topbiases
logits = outseq[9,:10] #only part of the output we care about (first 10 units 
#of the output from the last sequence)

               
# Define loss and optimizer for MNIST
loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)
grads_MNIST = optimizer_MNIST.compute_gradients(loss_MNIST)
train_MNIST = optimizer_MNIST.minimize(loss_MNIST)

# define loss and optimizer for new FIX
loss_FIX = tf.reduce_mean(acc_MNIST)
optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.001)
def Sub(a,b):
    return tf.subtract(a,b)
@tf.RegisterGradient("Sub")
def _sub_grad(unused_op, grad):
  return grad, tf.negative(grad)
gradient_FIX = tf.gradients([weights, biases, topweights, topbiases],)
train_FIX = optimizer_FIX.minimize(loss_FIX, gradient_FIX)

# Evaluate model with MNIST (with test logits, for dropout to be disabled)
prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(prediction), tf.argmax(Y))
accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
# Run the initializer
sess.run(init)
#saver.restore(sess, "/tmp/TubeNet_MNIST_nfixes_dualoptimizer.py.ckpt")

# Start training
for iters in range(100000): # iterations
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    for n in range(10):
        # get new series of fixations by feeding forward for n fixations
        FEimg[0,n,:] = fe.fixeye(MNISTimg,fix[0,n,:].reshape([1,100]))
        singlein = np.concatenate((FEimg[0,n,:].reshape([1,100]),fix[0,n,:].reshape([1,100])),1)
        newfixind[n] = np.asarray(sess.run([nfi],feed_dict={x:singlein}))
        m=n+1
        if n==9:
            m=0
        fix[0,m,:] = np.zeros([1,100])
        fix[0,m,newfixind[n].astype(int)] = 1
    # get current weights and biases
    [pweights, pbiases, ptopweights, ptopbiases] = sess.run([weights, biases, topweights, topbiases])
    # now train using data from those n fixations
    sess.run(train_MNIST, feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    #now get new accuracy and loss
    loss, acc_MNIST = sess.run([loss_MNIST,accuracy_MNIST],
            feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    sess.run(train_FIX, feed_dict={X: np.concatenate((FEimg,fix),1),
                                   Y: acc_MNIST})

    # benchmarking
    if iters % 2000 == 0 or iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        loss, acc_MNIST = sess.run([loss_MNIST,accuracy_MNIST],
                feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    
        print("Iteration " + str(iters) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(loss)) + ", Accuracy " + \
              "{:.4f}".format(np.mean(acc_MNIST)) + ", fixation sequence was:")
        print(newfixind.astype(int))

#sess.close()
print("Optimization Finished!")
saver = tf.train.Saver()
saver.save(sess, 'tmp/TubeNet_MNIST_nfixes_dualoptimizer.ckpt')


# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len]
test_label = mnist.test.labels[:test_len]
finalaccuracy = np.zeros(test_len)
finalloss = np.zeros(test_len)
finalfixes = np.zeros([10,test_len])

for tes in range(test_len):
    MNISTimg, MNISTcat = mnist.test.next_batch(1)
    MNISTimg = MNISTimg+0.001 # otherwise can lead to cropping problems in fixeye

    for n in range(10):
        # get new series of fixations by feeding forward for n fixations
        FEimg[0,n,:] = fe.fixeye(MNISTimg,fix[0,n,:].reshape([1,100]))
        singlein = np.concatenate((FEimg[0,n,:].reshape([1,100]),fix[0,n,:].reshape([1,100])),1)
        newfixind[n] = np.asarray(sess.run([nfi],feed_dict={x:singlein}))
        m=n+1
        if n==9:
            m=0
        fix[0,m,:] = np.zeros([1,100])
        fix[0,m,newfixind[n].astype(int)] = 1
        
    # now test using data from those n fixations
    loss, acc_MNIST = sess.run([loss_MNIST,accuracy_MNIST],
            feed_dict={X: np.concatenate((FEimg,fix),2), Y: MNISTcat})
    finalaccuracy[tes] = acc_MNIST
    finalloss[tes] = loss
    finalfixes[:,tes] = newfixind.astype(int)
             
print("Testing Accuracy:", "{:.4f}".format(np.mean(finalaccuracy)), 
      ", final fixation sequence was:")
print(finalfixes[:,test_len-1].astype(int))
