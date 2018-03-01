#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:18:01 2017

@author: jordandekraker
"""
import cv2
import numpy as np
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


#image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
#image = image.astype(float)


        
# fisheye filter
def fixeye(image,fix):
    outsz = [10,10]
    #make square
    if np.any(np.isin(image.shape,1)):
        image = np.reshape(image,[np.sqrt(image.size).astype(int),np.sqrt(image.size).astype(int)])
    if np.any(np.isin(fix.shape,1)):
        fix = np.reshape(fix,[np.sqrt(fix.size).astype(int),np.sqrt(fix.size).astype(int)])
    #params
    distCoeff = np.array([[0.01], [0.01], [0], [0]])
    fix = np.where(fix==1)
    x = fix[0]
    y = fix[1]
    fix = np.asarray([x[0],y[0]])
    
    fix = np.round(fix/outsz*image.shape) # scale fixation to image size!
    cam = np.eye(3)
    cam[0,2] = fix[0]  # define center x
    cam[1,2] = fix[1] # define center y
    cam[0,0] = 5.        # define focal length x
    cam[1,1] = 5.        # define focal length y
         #run
    dst = cv2.undistort(image,cam,distCoeff)
    
    #crop?
    dst = dst[~np.all(dst==0,axis=0)]
    dst = dst.T
    dst = dst[~np.all(dst==0,axis=1)]
    dst = dst.T

    dst = scipy.misc.imresize(dst,outsz)
#    dst = dst - np.mean(dst)
#    dst = dst / np.std(dst)
    dst = np.reshape(dst,[1,outsz[0]*outsz[1]]) #make 1D
    return (dst)

def fixembed(fix):
    fix = np.multiply(fix,outsz)
    fix = fix.astype(np.int64)
    img2 = np.zeros(outsz)
    img2[fix[1],fix[0]] = 1
#    img2 = gaussian_filter(img2,50, mode='constant', cval=0.0) # sigma chosen arbitrarily.. related to focal length?
#    img2 = img2 - np.mean(img2)
#    img2 = img2 / np.std(img2)
    img2 = np.reshape(img2,[outsz[0]*outsz[1]]) #make 1D
    return (img2)


# dst = fixeye(image,fix)
#fig, ax = plt.subplots()
#ax.imshow(dst)





