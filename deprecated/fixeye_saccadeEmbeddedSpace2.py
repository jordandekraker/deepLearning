#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:20:21 2018

@author: jordandekraker
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
from fixeye_saccade import fixeye,fixembed

image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
image = image.astype(float)
outsz = [100,100]
outsz1d = outsz[0]*outsz[1]

#generate training data
training_steps = 1000;
testing_steps = 10;
trainimgs = np.zeros([outsz1d*2,training_steps])
testimgs = np.zeros([outsz1d*2,testing_steps])
fix = np.array([0.5,0.5])
for n in range(0,training_steps+testing_steps):
    img1 = fixeye(image,fix)
    fix = np.random.normal(0.5,0.1,2)
    fix[fix>1] = 1
    fix[fix<0] = 0
    img2 = fixembed(fix)
    #fig, ax = plt.subplots()
    #ax.imshow(img2)
    if n < training_steps:
        trainimgs[:,n] = np.concatenate((img1, img2)) #remember, this gives the current view and the location of the NEXT fix
    else:
        testimgs[:,n-training_steps] = np.concatenate((img1, img2))

n=5
fig, ax = plt.subplots()
ax.imshow(np.reshape(testimgs[0:outsz1d,n],outsz))
fig, ax = plt.subplots()
ax.imshow(np.reshape(testimgs[outsz1d:outsz1d*2,n],outsz))



# build LSTM

# Training Parameters
learning_rate = 0.001
batch_size = 1
display_step = 100

# Network Parameters
num_input = outsz1d*2 # MNIST data input (img shape: 28*28)
timesteps = 1
num_hidden = 1000 # hidden layer num of features
num_output = outsz1d

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

LSTMoutput = RNN(X,weights,biases)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(Y,LSTMoutput)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for n in range(timesteps, training_steps-1):
        batch_x = trainimgs[0:outsz1d*2,n-timesteps:n]
        #batch_x = batch_x.T
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_y = trainimgs[0:outsz1d,n+1]
        batch_y = batch_y.reshape((batch_size, num_output))
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if n % display_step == 0 or n <= 10:
            # Calculate batch loss
            loss = sess.run([loss_op], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Training iteration " + str(n) + ", MSerror= " + str(loss))

    print("Optimization Finished!")
    
    for n in range(1,testing_steps):
        xtest = testimgs[:,n]
        xtest = xtest.reshape((1, 1, num_input))
        ytest = sess.run(LSTMoutput, feed_dict={X: xtest})
        fig, ax = plt.subplots()
        ax.imshow(np.reshape(testimgs[outsz1d:outsz1d*2,n],outsz))
        fig, ax = plt.subplots()
        ax.imshow(np.reshape(ytest[0,:],outsz))
