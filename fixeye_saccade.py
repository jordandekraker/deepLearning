#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:18:01 2017

@author: jordandekraker
"""
import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt


#image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#image = image.astype(float)
#fix = [.3, .3];

        
# fisheye filter
def fixeye(image,fix):
    #params
    distCoeff = np.array([[0.01], [0.01], [0], [0]])
    fixation = np.multiply(fix,image.shape)
    cam = np.eye(3)
    cam[0,2] = fixation[0]  # define center x
    cam[1,2] = fixation[1] # define center y
    cam[0,0] = 5.        # define focal length x
    cam[1,1] = 5.        # define focal length y
    outsz = [100,100];
         #run
    dst = cv2.undistort(image,cam,distCoeff)
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T

    dst = scipy.misc.imresize(dst,outsz)
    dst = dst - np.mean(dst)
    dst = dst / np.std(dst)
    dst = np.reshape(dst,[outsz[0]*outsz[1]]) #make 1D
    return (dst)
    

# dst = fixeye(image,fix)
#fig, ax = plt.subplots()
#ax.imshow(dst)





