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
try:
    loaded
except NameError:
    maxfixes = 50
    minfixes = 5
    iters = 50000
    disp_n_iters = int(iters/10)
    test_len = 1000
    
    inputshape = 25
    fixshape = 100
    memshape = 100
    outshape = 10
    
    import numpy as np
    import tensorflow as tf
    import fixeye_saccade as fe
    import matplotlib.pyplot as plt
    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    loaded = True
else:
    print('inputs already defined')

# tf Graph setup
X = tf.placeholder("float32", [1,784]) #size of an MNIST image
Y = tf.placeholder("float64", [1,10]) #number of possible classes
e = tf.placeholder("float64", [1])

# initialize tensorflow trainable variables
# S=sensory, H=hippocampal, M=motor
# w=weights, b=biases, a=activations
S1 = tf.Variable(tf.random_normal([inputshape+fixshape, inputshape+fixshape])) 
S1b = tf.Variable(tf.random_normal([inputshape+fixshape])) 
with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
    with tf.variable_scope('basic_lstm_cell'):
        weights = tf.get_variable('kernel',[(inputshape+fixshape+memshape), (memshape)*4])
        biases = tf.get_variable('bias',[(memshape)*4]) 
        # NOTE: these two variables are what tf.contrib.rnn.BasicLSTMCell would create by default
H1 = tf.contrib.rnn.BasicLSTMCell(memshape, state_is_tuple=True, reuse=True)
H2w = tf.Variable(tf.random_normal([memshape,outshape])) 
H2b = tf.Variable(tf.random_normal([outshape])) 
M1w = tf.Variable(tf.random_normal([outshape+fixshape, fixshape])) 
M1b = tf.Variable(tf.random_normal([fixshape])) 

H1m_old_m = tf.Variable(tf.random_normal([1, memshape])) # ititial hidden layer c_state and m_state 
H1m_old_c = tf.Variable(tf.random_normal([1, memshape])) # ititial hidden layer c_state and m_state 
RollingAverage = tf.Variable(tf.zeros([maxfixes]))
fix_list = tf.Variable(tf.zeros([fixshape]))

optimizer_FIX = tf.train.GradientDescentOptimizer(learning_rate=0.1) 
optimizer_MNIST = tf.train.GradientDescentOptimizer(learning_rate=0.001)



# define the model
def tubenet(X,Y,e):
    fixind = [55] # initial fixation pt
    fix_list.assign(tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],[1.0],[fixshape]))) 
    H2a=tf.constant([0.0],shape=[1,10])
    recordfixes = tf.constant([0],tf.int64);
    certainty = tf.constant([0.0])
    n=0
        
    # get new series of fixations by feeding forward for n fixations (in tf.while_loop)
    def cond(recordfixes,n,certainty,H2a):
        certainty_check = tf.cond((certainty[n]<0.9),lambda:True,lambda:False)
        return tf.cond(((n<minfixes) | (n<maxfixes)) & certainty_check,lambda:True,lambda:False)
    
    def explore(recordfixes,n,certainty,H2a):
        #get fisheye image at fix, and concatenate fix
        FEimg = tf.cast(tf.py_func(fe.fixeye,[X, fix_list],[tf.float64]),tf.float32)
        fixwithimg = tf.reshape(tf.concat((FEimg[0,:],fix_list),0),[1,inputshape+fixshape])
        
        # feed new image and its fix forward
        S1a = tf.tanh(tf.matmul(fixwithimg, S1) + S1b) 
        with tf.variable_scope('rnn') as scope:
            scope.reuse_variables()
            H1a, H1m = H1(S1a, (H1m_old_m,H1m_old_c))
        H2a = tf.tanh(tf.matmul(H1a, H2w) + H2b)
        # concatenate fix after LSTM layer
        H2aWithfix = tf.reshape(tf.concat((H2a[0,:],fix_list),0),[1,-1])
        M1a = tf.matmul(H2aWithfix, M1w) + M1b # linear activation function?
        
        # measure certainty to determine whether to keep exploring
        certainty = tf.concat((certainty,[tf.reduce_max(tf.nn.softmax(H2a))]),0)

        # train for fixes that increase the change in memory 
        Qsignal = tf.reduce_mean(tf.square(H1m[1]-H1m_old_c),1) # mean mem diff
        Q = tf.reshape(M1a[0,:],[fixshape])
        Qchange = Qsignal-RollingAverage[n] #difference from rollingaverage
        Qtarget = Q + tf.multiply(Q,Qchange)
        loss_FIX = tf.squared_difference(Qtarget,Q)
        optimizer_FIX.minimize(loss_FIX,var_list=[M1w,M1b])
        # update rolling average
        RollingAverage[n].assign(RollingAverage[n]*0.99 + Qsignal[n]*0.01)
        
        # sometimes try random fixation (e decreases over time)
        fixind = tf.cond(tf.random_uniform([1],0,1,dtype=tf.float64)[0] < e[0], 
                         lambda: tf.random_uniform([1],0,fixshape,dtype=tf.int64),
                         lambda: tf.argmax(Q))
        
        #regroup for next fixation
        fix_list.assign(tf.sparse_tensor_to_dense(tf.SparseTensor([fixind],[1.0],[fixshape])))
        recordfixes = tf.concat((recordfixes,[fixind]),0)
        H1m_old_m.assign(H1m[0])
        H1m_old_c.assign(H1m[1])
        
        n += 1
        return recordfixes,n,certainty,H2a
    recordfixes,n,certainty,H2a = tf.while_loop(cond,explore,(recordfixes,n,certainty,H2a),
        shape_invariants=[tf.TensorShape([None]),0,tf.TensorShape([None]),tf.TensorShape([1,10])])
    
    # Define loss and optimizer for MNIST
    logits = tf.reshape(H2a[0,:outshape],[1,10]) # MNIST categories
    loss_MNIST = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits, labels=Y))
    optimizer_MNIST.minimize(loss_MNIST)
    
    # Evaluate model with MNIST (with test logits, for dropout to be disabled)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
    accuracy_MNIST = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    return recordfixes,loss_MNIST,accuracy_MNIST
fix,loss_MNIST,accuracy_MNIST = tubenet(X,Y,e)



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
    ee = 1./((i/50) + 10)
    MNISTimg, MNISTcat = mnist.train.next_batch(1)
    tt1,tt2,lossperiter[i],accuracyperiter[i],finalfix = sess.run(
            [t1,t2,loss_MNIST,accuracy_MNIST,fix], 
            feed_dict={X: MNISTimg, Y: MNISTcat, e: [ee]})
    
    # ongoing display
    if i % disp_n_iters == 0:
        # Calculate seq loss and accuracy and see fixations used
        print("Iteration " + str(i) + ", loss_MNIST= " + \
              "{:.4f}".format(np.mean(lossperiter[i-disp_n_iters:i])) + ", Accuracy= " + \
              "{:.4f}".format(np.mean(accuracyperiter[i-disp_n_iters:i])) + 
              ", fixation sequence was:")
        print(np.where(finalfix[:,:]==1)[1])
    
print("Optimization Finished!")
#saver.save(sess, "tmp/TubeNet_MNIST_nfixes_singleoptimizer.ckpt")
#train_writer = tf.summary.FileWriter('tmp/TubeNet_MNIST_nfixes_singleoptimizer',sess.graph)

meanaccperiter = np.zeros(int(iters/disp_n_iters))
for n in range(int(iters/disp_n_iters)):
    meanaccperiter[n] = np.mean(accuracyperiter[n*disp_n_iters : (n+1)*disp_n_iters])
plt.plot(meanaccperiter)
plt.show()


# Calculate accuracy for 128 mnist test images
finallossperiter = np.zeros(test_len)
finalaccuracyperiter = np.zeros(test_len)
for i in range(test_len):
    MNISTimg, MNISTcat = mnist.test.next_batch(1)
    finallossperiter[i], finalaccuracyperiter[i], outerfixes = sess.run(
            [loss_MNIST,accuracy_MNIST,fix],
            feed_dict={X: MNISTimg, Y: MNISTcat, e: [ee]})
             
print("Test mean loss_MNIST= " + "{:.4f}".format(np.mean(finallossperiter)) \
      + ", Accuracy= " + "{:.4f}".format(np.mean(finalaccuracyperiter)) \
      + ", fixation sequence was:")
print(np.where(outerfixes[:,:]==1)[1])

sess.close()