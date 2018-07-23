#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:29:24 2018

Creates TubeNet architecture for predicting new image from image+xy motion

@author: jordandekraker
"""

import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import fixeye_saccade as fe

image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#fig, ax = plt.subplots()
#ax.imshow(image)

sensesize = 1024
motorsize = 3
NNwidth = 2*(sensesize+motorsize)

# initialize tensorflow trainable variables
# S=sensory, H=hippocampal, M=motor
# w=weights, b=biases, a=activations
S1w = tf.Variable(tf.random_normal([sensesize+motorsize, NNwidth])) 
S1b = tf.Variable(tf.random_normal([NNwidth])) 
S1a = tf.Variable(tf.constant(0.0,shape=[1,NNwidth])) 

S2w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
S2b = tf.Variable(tf.random_normal([NNwidth])) 
S2a = tf.Variable(tf.constant(0.0,shape=[1,NNwidth])) 

M2w = tf.Variable(tf.random_normal([NNwidth, NNwidth])) 
M2b = tf.Variable(tf.random_normal([NNwidth])) 
M2a = tf.Variable(tf.constant(0.0,shape=[1,NNwidth])) 

M1w = tf.Variable(tf.random_normal([NNwidth, motorsize])) 
M1b = tf.Variable(tf.random_normal([motorsize])) 
M1a = tf.Variable(tf.constant(0.0,shape=[1,motorsize])) 

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) 

# initial fixation
fix = tf.placeholder("float32", [3]) #size of an MNIST image
X = tf.placeholder("float32", [image.shape[0],image.shape[1]]) #size of an MNIST image

# define the model
def tubenet(X,fix):
    #get fisheye image at fix, and concatenate fix
    FEimg = tf.py_func(fe.fixeye,[X, fix],[tf.float64])
    FEimg = tf.cast(FEimg,tf.float32)
    S = tf.reshape(tf.concat((FEimg[0,:],fix[:]),0),[1,sensesize+motorsize])
    
    # feed new image and its fix forward
    nS1a = tf.tanh(tf.matmul(S, S1w) + S1b) 
    nS2a = tf.tanh(tf.matmul(nS1a, S2w) + S2b) 
    nM2a = tf.matmul(nS2a, M2w) + M2b # linear activation function?
    nM1a = tf.matmul(nM2a, M1w) + M1b # linear activation function?
    
    # optimize weights
    loss = tf.squared_difference(nS1a,S1a)
    optimizer.minimize(loss,var_list=[S1w,S1b])
    loss = tf.squared_difference(nS2a,S2a)
    optimizer.minimize(loss,var_list=[S2w,S2b])
    loss = tf.squared_difference(nM2a,M2a)
    optimizer.minimize(loss,var_list=[M2w,M2b])
    loss = tf.squared_difference(nM1a,M1a)
    optimizer.minimize(loss,var_list=[M1w,M1b])
    
    # assign old values new
    tf.assign(S1a,nS1a)
    tf.assign(S2a,nS2a)
    tf.assign(M2a,nM2a)
    tf.assign(M1a,nM1a)
    
    fix = fix+M1a
    return S,M1a
S,M1a = tubenet(X,fix)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver()
#saver.restore(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#writer = tf.train.write_graph(tf.get_default_graph(),'tmp/tensorboard','dualoptimizer')

# Start training


S,M1a = sess.run([S,M1a],feed_dict={X:image,fix:[0.5, 0.5, 0.9]})

sess.close()