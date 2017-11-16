#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:18:01 2017

@author: jordandekraker
"""
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
image = image +1
fig, ax = plt.subplots()
ax.imshow(image)


# fisheye filter
distCoeff = np.zeros((4,1),np.float64)
distCoeff[0,0] = 0.01;
distCoeff[1,0] = 0.01;
distCoeff[2,0] = 0.0;
distCoeff[3,0] = 0.0;
cam = np.eye(3,dtype=np.float32)
cam[0,2] = 100  # define center x
cam[1,2] = 100 # define center y
cam[0,0] = 5.        # define focal length x
cam[1,1] = 5.        # define focal length y
dst = cv2.undistort(image,cam,distCoeff)
dst = dst[~np.all(dst==0,axis=1)]
# dst = dst[~np.all(dst==0,axis=0)]
# dst2 = dst[range(min(a),max(a)), range(min(b),max(b))]



fig, ax = plt.subplots()
ax.imshow(dst)

