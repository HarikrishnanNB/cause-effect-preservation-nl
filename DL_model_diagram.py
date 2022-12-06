#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:38:56 2022

@author: harikrishnan
"""

### DL Model diagram

# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
# from Codes import chaosnet, k_cross_validation
# import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split
# from ETC import CCC
import keras
from keras.models import Sequential
from keras.layers import Dense


from keras import callbacks

from keras.utils.vis_utils import plot_model




    
   
model = Sequential()
model.add(Dense(5000, input_dim=2000, activation='sigmoid'))
model.add(Dense(500, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(2, activation='softmax'))
# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)  