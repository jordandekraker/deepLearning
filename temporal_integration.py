#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:20:54 2017

@author: jordandekraker
"""
import cv2
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from fixeye_saccade import fixeye

image = cv2.imread("./SmileyFace8bitGray.png",cv2.IMREAD_GRAYSCALE)
image = image.astype(float)

# initial
fix = [0.5, 0.5] # starting pt
img1 = fixeye(image,fix)
sz = img1.shape
sz2 = np.add(sz,2)
inputs = Input(shape=sz2)
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='relu')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='SGD',
              loss='mean_squared_error')

for iters in range(10):
    fix = np.random.uniform(0.1, 0.9, 2)
    img2 = fixeye(image,fix)
    model.fit(np.concatenate((img1, fix)), img2)  # starts training
    img1 = img2
         
    