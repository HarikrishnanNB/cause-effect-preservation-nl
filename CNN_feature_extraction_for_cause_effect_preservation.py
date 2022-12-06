#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022


In this code, we train with UNI-DIR-COUP-TENT (b1 = 0.65 , b2 = 0.47) and test with 

1) UNI-DIR-COUP-TENT-DATA-2 (b1 = 0.6, b2 = 0.4)
2) UNI-DIR-COUP-TENT-DATA-3 (b1 = 0.1, b2 = 0.3)
3) UNI-DIR-COUP-TENT-DATA-4 (b1 = 0.49, b2 = 0.52)

"""



import os
import numpy as np
# import scipy
from scipy import io
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
# from sklearn import datasets
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
# from Codes import chaosnet, k_cross_validation
from sklearn.model_selection import train_test_split
# from ETC import CCC

from keras.models import Sequential
from keras.layers import Dense, Convolution1D, Dropout, Flatten, MaxPooling1D

from scipy import io


PATH = os.getcwd()


DATA_NAME_TRAIN = 'AR-1'
DATA_NAME_TEST = 'AR-1'
VAR = 1000
LEN_VAL = 2000

# CNN architecture details
input_dim = 2000
out_dim = 2
batch_size_val = 32
epochs_val = 30


COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Sync_Error = np.zeros(len(COUP_COEFF1))
F1_score_Result_array = np.zeros(len(COUP_COEFF1))







ROW = -1
for COUP_COEFF in COUP_COEFF1:
    
    ROW = ROW+1
    RESULT_PATH = PATH + '/CNN_FEATURES/'  + DATA_NAME_TRAIN + '/' + str(COUP_COEFF) +'/'
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    
    RESULT_PATH_TRAIN = PATH + '/DATA/'  + DATA_NAME_TRAIN + '/' + str(COUP_COEFF) +'/'
    

    #### Loading Traindata

    Y_independent_data = io.loadmat(RESULT_PATH_TRAIN + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH_TRAIN + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH_TRAIN + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH_TRAIN + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_independent_data['class_0_indep_raw_data'][0:VAR, 0:LEN_VAL]
    class_0_label = Y_independent_label['class_0_indep_raw_data_label'][0:VAR, 0:LEN_VAL]
    class_1_data = X_dependent_data['class_1_dep_raw_data'][0:VAR, 0:LEN_VAL]
    class_1_label = X_dependent_label['class_1_dep_raw_data_label'][0:VAR, 0:LEN_VAL]

    
  
   

    # model.add(Dense(65, activation='relu'))
   

    model = Sequential()
    layer_1 = model.add(Convolution1D(32, 3, activation="relu",input_shape=(input_dim, 1), name="layer_1"))
    layer_2 = model.add(MaxPooling1D(2, name="layer_2"))
    layer_3 = model.add(Flatten(name="layer_3"))
    layer_4 = model.add(Dense(32, activation="relu", name="layer_4"))
    # model.add(Dense(65, activation='relu'))
    layer_5 = model.add(Dense(out_dim, activation='softmax', name="layer_5"))

    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #checkpointer = callbacks.ModelCheckpoint(filepath=TRAIN_DATA_PATH + "checkpoint.hdf5", verbose=1, monitor='val_accuracy', mode='max', save_best_only=True)
    model.load_weights('CNN/logs/'+ DATA_NAME_TRAIN +'/' +str(COUP_COEFF)+'/'+'check-points'+'/'+'checkpoint.hdf5')
    
    
    from keras import backend as K

    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[3].output])
    layer_output_class_0 = get_3rd_layer_output([class_0_data.reshape(class_0_data.shape[0], class_0_data.shape[1], 1)])[0]
    layer_output_class_1 = get_3rd_layer_output([class_1_data.reshape(class_1_data.shape[0], class_1_data.shape[1], 1)])[0]
    
        # Saving the Firing Time and Firing Rate
    io.savemat(RESULT_PATH + 'class_0_indep_cnn_features.mat', {'class_0_indep_cnn_features': layer_output_class_0 })
    io.savemat(RESULT_PATH + 'class_1_dep_cnn_features.mat', {'class_1_dep_cnn_features': layer_output_class_1})
    
 