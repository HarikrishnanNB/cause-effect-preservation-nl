
'''
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022

In this code, the chaosfex features for AR-1 master slave system are extracted.
ChaosFEX hyperparameters
------------------------
q = 0.78
b = 0.499
epsilon = 0.17
'''



import os
import numpy as np
from scipy import io
import ChaosFEX.feature_extractor as CFX

PATH = os.getcwd()
DATA_NAME = 'AR-1'
COUP_COEFF1 =  np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
for COUP_COEFF in COUP_COEFF1:
    print(COUP_COEFF)
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'
    # Loading the normalized Raw Data
    Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_independent_data['class_0_indep_raw_data']
    class_0_label = Y_independent_label['class_0_indep_raw_data_label']
    class_1_data = X_dependent_data['class_1_dep_raw_data']
    class_1_label = X_dependent_label['class_1_dep_raw_data_label']
        
    # total_data = np.concatenate((class_0_data, class_1_data))
    # total_label = np.concatenate((class_0_label, class_1_label))
    INA = 0.78
    DT = 0.499
    EPSILON_1 = 0.17
    
    # ChaosFEX feature Extraction
    feat_mat_class_0 = CFX.transform(class_0_data, INA, 10000, EPSILON_1, DT)
    feat_mat_class_1 = CFX.transform(class_1_data, INA, 10000, EPSILON_1, DT)
    
    
    
    
    # Saving the Firing Time and Firing Rate
    io.savemat(RESULT_PATH + 'class_0_indep_firing_time.mat', {'class_0_indep_firing_time': feat_mat_class_0[:, 4000:6000]})
    io.savemat(RESULT_PATH + 'class_1_dep_firing_time.mat', {'class_1_dep_firing_time': feat_mat_class_1[:, 4000:6000]})
    
    io.savemat(RESULT_PATH + 'class_0_indep_firing_rate.mat', {'class_0_indep_firing_rate': feat_mat_class_0[:, 0:2000]})
    io.savemat(RESULT_PATH + 'class_1_dep_firing_rate.mat', {'class_1_dep_firing_rate': feat_mat_class_1[:, 0:2000]})
    
