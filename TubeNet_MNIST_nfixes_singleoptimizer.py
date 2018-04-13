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
#nfixes = 5
#iters = 100000
#disp_n_iters = 1000
#test_len = 1000

inputshape = 25
fixshape = 100

import numpy as np
import tensorflow as tf
import fixeye_saccade as fe
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# tf Graph setup
X = tf.placeholder("float32", [1,784]) #size of an MNIST image
Y = tf.placeholder("float64", [1,10]) #number of possible classes

# initialize tensorflow trainable variables
bottomweights1 = tf.Variable(tf.random_normal([inputshape+fixshape, inputshape+fixshape])) 
bottombiases1 = tf.Variable(tf.random_normal([inputshape+fixshape])) 
topweights = tf.Variable(tf.random_normal([inputshape+fixshape, 10+fixshape])) 
topbiases = tf.Variable(tf.random_normal([10+fixshape])) 
with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        weights = tf.get_variable('kernel',[(inputshape+fixshape)*2, (inputshape+fixshape)*4])
        biases = tf.get_variable('bias',[(inputshape+fixshape)*4]) 
        # NOTE: these two variables are what tf.contrib.rnn.BasicLSTMCell would create by default
lstm_cell = tf.contrib.rnn.BasicLSTMCell(inputshape+fixshape, state_is_tuple=True, reuse=True)
memstate = (tf.Variable(tf.random_normal([1, inputshape+fixshape])),)*2 # ititial hidden layer c_state and m_state 
mdiff = tf.zeros([nfixes-1])

e=0.1 #this annoys the crap out of me but has to be here for runnfixes to stop complaining

# define the model
def runnfixes(image,memstate,meanmemdiff):
    fixind = [55] # initial fixation pt
    fix_list = []
    fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],[1.0],[fixshape]))) 
    fix = tf.reshape(fix_list,[-1,fixshape])
    memdiff_m = []
    
    # get new series of fixations by feeding forward for n fixations
    for n in range(nfixes-1):  
          
        #get fisheye image at fix, and concatenate fix
        FEimg = tf.cast(tf.py_func(fe.fixeye,[image, fix[n,:]],[tf.float64]),tf.float32)
        fixwithimg = tf.reshape(tf.concat((FEimg[0,:],fix[n,]),0),[1,inputshape+fixshape])
        
        # feed new image and its fix forward
        l1 = tf.tanh(tf.matmul(fixwithimg, bottomweights1) + bottombiases1) 
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            l2, newmemstate = lstm_cell(tf.reshape(l1,[1,inputshape+fixshape]), memstate)
        memdiff_m.append(tf.reduce_mean(tf.square(newmemstate[1]-memstate[1]),1))
        l3 = tf.matmul(l2, topweights) + topbiases # linear activation function?
        Q = tf.reshape(l3[0,10:fixshape+10],[fixshape])
        
        # train for fixes that increase memdiff_m (regularize for mean memdiff_m)
#        Qchange = tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],
#            memdiff_m[n]-meanmemdiff[n],[fixshape]))
#        Qtarget = Q + tf.multiply(Q,Qchange)
#        loss_FIX = tf.squared_difference(Qtarget,Q)
#        optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
#        optimizer_FIX.minimize(loss_FIX)#,var_list=[topweights,topbiases])
        
        # sometimes try random fixation (e decreases over time)
        fixind = [tf.argmax(Q)]
        if np.random.rand(1) < e:
            fixind = tf.random_uniform([1],0,fixshape,dtype=tf.int64)
        
        #regroup for next fixation
        fix_list.append(tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],[1.0],[fixshape])))
        fix = tf.reshape(fix_list,[-1,fixshape])
        memstate = newmemstate
        
    # track memdiff over time to get relative goodness of fixation
    memdiff_m = tf.reshape(memdiff_m,[nfixes-1])        
    meanmemdiff = meanmemdiff*0.99 + memdiff_m+0.01 # running average with momentum
    return fix, memstate, l3, meanmemdiff,n
seqfix,newmemstate,finalout,mdiff,nn = runnfixes(X,memstate,mdiff)

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
#writer = tf.train.write_graph(tf.get_default_graph(),'tmp/tensorboard','dualoptimizer')

# Start training
# initialize outer np variables
lossperiter = np.zeros(iters)
accuracyperiter = np.zeros(iters)
for i in range(iters): # iterations
    e = 1./((i/50) + 10)
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    lossperiter[i],accuracyperiter[i],finalfix = sess.run(
            [loss_MNIST,accuracy_MNIST,seqfix], 
            feed_dict={X: MNISTimg, Y: MNISTcat})
    if i % disp_n_iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        print("Iteration " + str(i) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(lossperiter[i-disp_n_iters:i])) + ", Accuracy= " + \
              "{:.4f}".format(np.mean(accuracyperiter[i-disp_n_iters:i])) + 
              ", fixation sequence was:")
        print(np.where(finalfix[:,:]==1)[1])
    sess.run(train_MNIST, feed_dict={X: MNISTimg, Y: MNISTcat})
    
print("Optimization Finished!")
#saver.save(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#train_writer = tf.summary.FileWriter('tmp/TubeNet_MNIST_nfixes_singleoptimizer',sess.graph)

# Calculate accuracy for 128 mnist test images
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
#plt.plot(meanaccperiter)

sess.close()