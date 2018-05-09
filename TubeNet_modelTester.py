#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:06:26 2018

tests MNISTperformance on single vs dual optimizer models

@author: jordandekraker
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt

nfixes = 5
iters = 100000 #10k training images seen 10 times each
tests = 20
disp_n_iters = int(iters/10)
pltwindow = int(iters/100)
test_len = 1000 #1k test images

scripts = ['TubeNet_MNIST_S1H1M1.py',
           'TubeNet_MNIST_S1H1M1_shared.py']

allaccperiter = np.zeros([iters,tests,len(scripts)])
alltestacc = np.zeros([tests,len(scripts)])
for ntests in range(tests):
    for s in range(len(scripts)):
        runfile('/kohlerlab/Jordan/deepLearning/'+scripts[s])
        allaccperiter[:,ntests,s] = accuracyperiter
        alltestacc[ntests,s] = np.mean(finalaccuracyperiter)
        print(scripts[s])

trainingplt = np.zeros([int(iters/pltwindow),len(scripts)])
meanallaccperiter = np.zeros([iters,len(scripts)])
for s in range(len(scripts)):
    print(scripts[s] + ": mean test accuracy =" + "{:.4f}".format(np.mean(alltestacc[:,s])))    
    #plot accuracy over the course of training
    meanallaccperiter[:,s] = np.mean(allaccperiter[:,:,s],1)
    for aw in range(0,iters,pltwindow):
        trainingplt[int(aw/pltwindow),s] = np.mean(meanallaccperiter[aw:aw+pltwindow,s])
        
for s in range(len(scripts)):
    plt.plot(trainingplt[:,s], label=scripts[s])
plt.xlabel("training iteration (*" + str(pltwindow) + ")")
plt.ylabel("MNIST accuracy")
plt.legend()
axes = plt.gca()
#axes.set_ylim([0.7,1.0])

plt.bar(range(s+1),np.mean(alltestacc,0))
plt.errorbar(range(s+1),np.mean(alltestacc,0),np.std(alltestacc,0), linestyle='none')
plt.xticks(range(s+1), scripts, rotation='vertical')
axes = plt.gca()
axes.set_ylim([np.min(alltestacc),np.max(alltestacc)])
