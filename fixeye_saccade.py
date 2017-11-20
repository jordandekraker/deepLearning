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

# fisheye filter
def fixeye(image,fix):
    distCoeff = np.array([[0.01], [0.01], [0], [0]])
    fix[0] = fix[0] * image.shape[0]
    fix[1] = fix[1] * image.shape[1]
    cam = np.eye(3)
    cam[0,2] = fix[0]  # define center x
    cam[1,2] = fix[1] # define center y
    cam[0,0] = 5.        # define focal length x
    cam[1,1] = 5.        # define focal length y
    dst = cv2.undistort(image,cam,distCoeff)
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T
    dst = dst - np.mean(dst)
    dst = dst - np.std(dst)
    dst = scipy.misc.imresize(dst,sz)
    return (dst)
    
image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
image = image.astype(float)

fix = [.3, .3];
sz = [100,100];
szn = sz[0]*sz[1];

dst = fixeye(image,fix)
fig, ax = plt.subplots()
ax.imshow(dst)





