#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022


Data generation code: Coupled master-slave for chaotic skew tent system.
1) UNI-DIR-COUP-TENT (b1 = 0.65, b2 = 0.47)
2) UNI-DIR-COUP-TENT-DATA-2 (b1 = 0.6, b2 = 0.4)
3) UNI-DIR-COUP-TENT-DATA-3 (b1 = 0.1, b2 = 0.3)
4) UNI-DIR-COUP-TENT-DATA-4 (b1 = 0.49, b2 = 0.52)



"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Codes import skew_tent

DATA_NAME = 'UNI-DIR-COUP-TENT-DATA-2'
LENGTH = 2500
TRANSIENT_LENGTH = 500
SAMPLES_PER_CLASS = 1000
B1 = 0.6
B2 = 0.4

COUP_COEFF = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


for COEFF in COUP_COEFF:

    class_0_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_0_label = np.zeros((SAMPLES_PER_CLASS,1))
    
    class_1_data = np.zeros((SAMPLES_PER_CLASS, LENGTH - TRANSIENT_LENGTH))
    class_1_label = np.ones((SAMPLES_PER_CLASS,1))
    
    for NUM_TRIALS in range(0, SAMPLES_PER_CLASS):
        x_in = np.random.rand(1)
        y_in = np.random.rand(1)
        master_timeseries = np.zeros(LENGTH)
        slave_timeseries = np.zeros(LENGTH)
            
        master_timeseries[0] = x_in
        slave_timeseries[0] = y_in
    
        for num_instance in range(1, LENGTH):
            master_timeseries[num_instance] = skew_tent(master_timeseries[num_instance - 1], B1)
            slave_timeseries[num_instance] = (1-COEFF) * skew_tent(slave_timeseries[num_instance - 1], B2) + COEFF * skew_tent(master_timeseries[num_instance - 1], B1)
        
        class_0_data[NUM_TRIALS, :] = master_timeseries[TRANSIENT_LENGTH:]
        class_1_data[NUM_TRIALS, :] = slave_timeseries[TRANSIENT_LENGTH:]
    
    PATH = os.getcwd()
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' + str(COEFF) +'/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    
    
    from scipy import io
    
    
    io.savemat(RESULT_PATH + 'Y_independent_data_class_0.mat', {'class_0_indep_raw_data': class_0_data})
    io.savemat(RESULT_PATH + 'Y_independent_label_class_0.mat', {'class_0_indep_raw_data_label': class_0_label})
    
    
    io.savemat(RESULT_PATH + 'X_dependent_data_class_1.mat', {'class_1_dep_raw_data': class_1_data})
    io.savemat(RESULT_PATH + 'X_dependent_label_class_1.mat', {'class_1_dep_raw_data_label': class_1_label})
    
    
        