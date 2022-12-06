#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 11:49:09 2022

@author: harikrishnan
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
from ETC import CCC



PATH = os.getcwd()


DATA_NAME = 'UNI-DIR-COUP-TENT'
TRIALS = 50
VAR = 1000
LEN_VAL = 2000

COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
M_causes_S = np.zeros(len(COUP_COEFF1))
S_causes_M = np.zeros(len(COUP_COEFF1))


ROW = -1
for COUP_COEFF in COUP_COEFF1:
    
    ROW = ROW+1

    RESULT_PATH = PATH + '/DL_FEATURES/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'



    Y_independent_data = io.loadmat(RESULT_PATH + 'class_0_indep_dl_features.mat')
    # Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'class_1_dep_dl_features.mat' )
    # X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_independent_data['class_0_indep_dl_features'][0:VAR, 0:LEN_VAL]
    # class_0_label = Y_independent_label['class_0_indep_raw_data_label'][0:VAR, 0:LEN_VAL]
    class_1_data = X_dependent_data['class_1_dep_dl_features'][0:VAR, 0:LEN_VAL]
    # class_1_label = X_dependent_label['class_1_dep_raw_data_label'][0:VAR, 0:LEN_VAL]

    total_data = np.concatenate((class_0_data, class_1_data))
    # total_label = np.concatenate((class_0_label, class_1_label))
    
    
    
    print("Coupling-Coefficient", COUP_COEFF)
 
    # INA = 0.56
    # DT = 0.499
    # EPSILON_1 = 0.171
    # # ChaosFEX feature Extraction
    # feat_mat_class_0_data = CFX.transform(class_0_data, INA, 10000, EPSILON_1, DT)[:, 4000:6000]
    # feat_mat_class_1_data = CFX.transform(class_1_data, INA, 10000, EPSILON_1, DT)[:, 4000:6000]
    
    M_S =[]
    S_M = []
    for data_instance_no in range(0, TRIALS):
        
        ccc_M_S = CCC.compute(class_1_data[data_instance_no, :], class_0_data[data_instance_no, :], LEN_past=15, ADD_meas=10, STEP_size=2, n_partitions=2)
        ccc_S_M = CCC.compute(class_0_data[data_instance_no, :], class_1_data[data_instance_no, :], LEN_past=15, ADD_meas=10, STEP_size=2, n_partitions=2)
        M_S.append(ccc_M_S)
        S_M.append(ccc_S_M)
    
    M_causes_S[ROW] = np.mean(M_S)
    S_causes_M[ROW] = np.mean(S_M)
    
RESULT_PATH_FINAL = PATH + '/' +'DL-RESULTS' + '/'+ DATA_NAME + '/' 


 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

plt.figure(figsize=(15,10))
plt.plot(COUP_COEFF1, M_causes_S, '-*k', markersize = 10, label = "M causes S (DL)")
plt.plot(COUP_COEFF1, S_causes_M, '-or', markersize = 10, label = "S causes M (DL)")
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('CCC', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/DL-"+DATA_NAME+"-raw_data-CCC-2.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/DL-"+DATA_NAME+"-raw_data-CCC.eps", format='eps', dpi=300)
plt.show()

# Saving Results

np.save(RESULT_PATH_FINAL +'DL_CCC_feature_data_M_causes_S.npy', M_causes_S)
np.save(RESULT_PATH_FINAL +'DL_CCC_feature_data_S_causes_M.npy', S_causes_M)


# L=20, w= 10, 15, delta=0, bin=2
