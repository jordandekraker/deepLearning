#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:18:01 2017

@author: jordandekraker
"""
import cv2
import numpy as np
import scipy.misc
#import matplotlib.pyplot as plt


#image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#image = image.astype(float)
        
# fisheye filter
def fixeye(image,fix):
    image = image+0.001 # otherwise can lead to cropping problems in fixeye
    outsz = [10,10]
    
    # make sure all inputs are square matrices
    image = np.reshape(image,[np.sqrt(image.size).astype(int),np.sqrt(image.size).astype(int)])
    fix = np.reshape(fix,[np.sqrt(fix.size).astype(int),np.sqrt(fix.size).astype(int)])
    #convert fix to coordinates
    distCoeff = np.array([[0.01], [0.01], [0], [0]])
    fix = np.where(fix==1)
    x = fix[0]
    y = fix[1]
    fix = np.asarray([x[0],y[0]])
    fix = np.round(fix/outsz*image.shape) # scale fixation values to fraction of image size!
    
    # set up fisheye parameters
    cam = np.eye(3)
    cam[0,2] = fix[0]  # define center x
    cam[1,2] = fix[1] # define center y
    cam[0,0] = 5.        # define focal length x
    cam[1,1] = 5.        # define focal length y
    #run fisheye
    dst = cv2.undistort(image,cam,distCoeff)
    
    #crop
    dst = dst[~np.all(dst==0,axis=0)]
    dst = dst.T
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T

    # resize and normalize
    dst = scipy.misc.imresize(dst,outsz)
    dst = dst - np.mean(dst)
    dst = dst / np.std(dst)
    dst = np.reshape(dst,[outsz[0]*outsz[1]]) #make 1D
    return (dst)


# dst = fixeye(image,fix)
#fig, ax = plt.subplots()
#ax.imshow(dst)