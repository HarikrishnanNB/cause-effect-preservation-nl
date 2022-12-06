#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 16:57:37 2021

@author: harikrishnan
"""
import os
import numpy as np
import scipy
from scipy import io




from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix as cm
from sklearn import datasets

from Codes import chaosnet, k_cross_validation
import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split

PATH = os.getcwd()
DATA_NAME = 'AR-1'
COUP_COEFF = 0.4
RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'



Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')

X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')

class_0_data = Y_independent_data['class_0_indep_raw_data']
class_0_label = Y_independent_label['class_0_indep_raw_data_label']
class_1_data = X_dependent_data['class_1_dep_raw_data']
class_1_label = X_dependent_label['class_1_dep_raw_data_label']

total_data = np.concatenate((class_0_data, class_1_data))
total_label = np.concatenate((class_0_label, class_1_label))


INITIAL_NEURAL_ACTIVITY = np.arange(0.71, 0.81, 0.01)
DISCRIMINATION_THRESHOLD = [0.499]#np.arange(0.1, 0.99, 0.01)
EPSILON = [0.171]#np.arange(0.16, 0.18,0.001)

traindata, testdata, trainlabel, testlabel = train_test_split(total_data, total_label, test_size=0.2, random_state=42)




    
FOLD_NO = 5 
Perf_Metric, Q, B, EPS, EPSILON = k_cross_validation(FOLD_NO, traindata, trainlabel, testdata, testlabel, INITIAL_NEURAL_ACTIVITY, DISCRIMINATION_THRESHOLD, EPSILON, DATA_NAME, COUP_COEFF )
