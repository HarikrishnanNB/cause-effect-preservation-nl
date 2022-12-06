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
DATA_NAME = 'Bidir_AR-1'
COUP_COEFF1 = np.array([0.4])
for COUP_COEFF in COUP_COEFF1:
    print(COUP_COEFF)
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COUP_COEFF) +'/'
    # Loading the normalized Raw Data
    Y_dependent_data = io.loadmat(RESULT_PATH + 'Y_dependent_data_class_0.mat')
    Y_dependent_label = io.loadmat(RESULT_PATH + 'Y_dependent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_dependent_data['class_0_dep_raw_data']
    class_0_label = Y_dependent_label['class_0_dep_raw_data_label']
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
    io.savemat(RESULT_PATH + 'class_0_dep_firing_time.mat', {'class_0_dep_firing_time': feat_mat_class_0[:, 4000:6000]})
    io.savemat(RESULT_PATH + 'class_1_dep_firing_time.mat', {'class_1_dep_firing_time': feat_mat_class_1[:, 4000:6000]})
    
    io.savemat(RESULT_PATH + 'class_0_dep_firing_rate.mat', {'class_0_dep_firing_rate': feat_mat_class_0[:, 0:2000]})
    io.savemat(RESULT_PATH + 'class_1_dep_firing_rate.mat', {'class_1_dep_firing_rate': feat_mat_class_1[:, 0:2000]})
    

