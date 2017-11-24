#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:20:54 2017

@author: jordandekraker
"""
import cv2
import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from fixeye_saccade import fixeye

image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
image = image.astype(float)

# initial
fix = [0.5, 0.5] # starting pt
img1 = np.concatenate(([[fixeye(image,fix)],[fix]]), axis=1)
img1 = img1

model = Sequential()
model.add(Dense(1000, activation='tanh', input_shape=(None,img1.shape[1])))
model.add(Dense(10002, activation='tanh'))
model.compile(optimizer='rmsprop',
              loss='mean_squared_error')

for iters in range(10):
    fix = np.random.uniform(0.1, 0.9, 2)
    img2 = fixeye(image,fix)
    model.fit(img1, img2, epochs=1, batch_size=1)  # starts training
    img1 = img2
         
    