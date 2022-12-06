#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022


This is the testing code for AR process using ChaosNet DATA_NAME = AR-1.
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
import ChaosFEX.feature_extractor as CFX
from sklearn.model_selection import train_test_split
# from ETC import CCC



PATH = os.getcwd()


DATA_NAME = 'AR-1'

VAR = 1000
LEN_VAL = 2000

COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
Sync_Error = np.zeros(len(COUP_COEFF1))
F1_score_Result_array = np.zeros(len(COUP_COEFF1))

ROW = -1
for COUP_COEFF in COUP_COEFF1:
    
    ROW = ROW+1

    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'



    Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_independent_data['class_0_indep_raw_data'][0:VAR, 0:LEN_VAL]
    class_0_label = Y_independent_label['class_0_indep_raw_data_label'][0:VAR, 0:LEN_VAL]
    class_1_data = X_dependent_data['class_1_dep_raw_data'][0:VAR, 0:LEN_VAL]
    class_1_label = X_dependent_label['class_1_dep_raw_data_label'][0:VAR, 0:LEN_VAL]

    total_data = np.concatenate((class_0_data, class_1_data))
    total_label = np.concatenate((class_0_label, class_1_label))
    
    
    Sync_Error[ROW] = np.mean(np.mean((class_0_data - class_1_data)**2,1))
    
    traindata, testdata, trainlabel, testlabel = train_test_split(total_data, total_label, test_size=0.2, random_state=42)
    
    print(traindata.shape[0])
    print(testdata.shape[0])
    INA = 0.78
    DT = 0.499
    EPSILON_1 = 0.171
    # ChaosFEX feature Extraction
    feat_mat_traindata = CFX.transform(traindata, INA, 10000, EPSILON_1, DT)
    feat_mat_testdata = CFX.transform(testdata, INA, 10000, EPSILON_1, DT)
    
    
    
    
    from sklearn.metrics.pairwise import cosine_similarity
    NUM_FEATURES = feat_mat_traindata.shape[1]
    NUM_CLASSES = len(np.unique(trainlabel))
    mean_each_class = np.zeros((NUM_CLASSES, NUM_FEATURES))
    
    for label in range(0, NUM_CLASSES):
        
        mean_each_class[label, :] = np.mean(feat_mat_traindata[(trainlabel == label)[:,0], :], axis=0)
        
    predicted_label = np.argmax(cosine_similarity(feat_mat_testdata, mean_each_class), axis = 1)
    
    
    ACC = accuracy_score(testlabel, predicted_label)*100
    RECALL = recall_score(testlabel, predicted_label, average="macro")
    PRECISION = precision_score(testlabel, predicted_label, average="macro")
    F1SCORE = f1_score(testlabel, predicted_label, average="macro")
    print("ChaosFEX - F1-score for COUP_COEFF =  ",COUP_COEFF,"is", F1SCORE)
    
    F1_score_Result_array[ROW] = F1SCORE 
    
    
    # plt.figure(figsize=(15,10))
    # plt.plot(class_0_data[0,0:100], '-*k', markersize = 10, label = 'Cause')
    # plt.plot(class_1_data[0,0:100], '-*r', markersize = 10, label = 'Effect')
    
    # plt.xticks(fontsize=30)
    # plt.yticks(fontsize=30)
    # plt.grid(True)
    # plt.xlabel('Time', fontsize=30)
    # plt.ylabel('Amplitude', fontsize=30)
    # plt.ylim(0, 1.1)
    # plt.legend(fontsize = 30)
    # plt.tight_layout()
    # plt.savefig(RESULT_PATH+"/PLOT-"+str(COUP_COEFF)+".jpg", format='jpg', dpi=300)
    # # plt.savefig(RESULT_PATH+"/Chaosnet-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
    # plt.show()

RESULT_PATH_FINAL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 


 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

    
# Final Result Plot
plt.figure(figsize=(15,10))
plt.plot(COUP_COEFF1,F1_score_Result_array, '-*k', markersize = 10, label = "F1-Score")
plt.plot(COUP_COEFF1,Sync_Error, '-or', markersize = 10, label = "MSE")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('F1-score/MSE', fontsize=30)
plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/Chaosnet-"+DATA_NAME+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/Chaosnet-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
plt.show()


np.save(RESULT_PATH_FINAL +'F1-score.npy', F1_score_Result_array)

np.save(RESULT_PATH_FINAL +'sync_error.npy', Sync_Error)





