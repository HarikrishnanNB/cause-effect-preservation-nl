#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022

Plotting the attractor dynamics for a coupling coefficient of 0.4.

1) UNI-DIR-COUP-TENT-DATA-2 (b1 = 0.6, b2 = 0.4)
2) UNI-DIR-COUP-TENT-DATA-3 (b1 = 0.1, b2 = 0.3)
3) UNI-DIR-COUP-TENT-DATA-4 (b1 = 0.49, b2 = 0.52)
4) UNI-DIR-COUP-LOGISTIC-MAP-DATA-1 (b1 = 4, b2 = 3.82)
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
PATH = os.getcwd()

DATA_NAME_TEST = 'UNI-DIR-COUP-TENT'
VAR = 1000
LEN_VAL = 2000

COUP_COEFF = 0.4





RESULT_PATH_TEST = PATH + '/DATA/'  + DATA_NAME_TEST + '/' + str(COUP_COEFF) +'/'


  
### Loading Testdata


Y_independent_data_test = io.loadmat(RESULT_PATH_TEST + 'Y_independent_data_class_0.mat')
Y_independent_label_test = io.loadmat(RESULT_PATH_TEST + 'Y_independent_label_class_0.mat')

X_dependent_data_test = io.loadmat(RESULT_PATH_TEST + 'X_dependent_data_class_1.mat' )
X_dependent_label_test = io.loadmat(RESULT_PATH_TEST + 'X_dependent_label_class_1.mat')

class_0_data_test = Y_independent_data_test['class_0_indep_raw_data'][0:VAR, 0:LEN_VAL]
class_0_label_test = Y_independent_label_test['class_0_indep_raw_data_label'][0:VAR, 0:LEN_VAL]
class_1_data_test = X_dependent_data_test['class_1_dep_raw_data'][0:VAR, 0:LEN_VAL]
class_1_label_test = X_dependent_label_test['class_1_dep_raw_data_label'][0:VAR, 0:LEN_VAL]


RESULT_PATH_FINAL = PATH + '/' +'ATTRACTOR-RESULTS' + '/'+ DATA_NAME_TEST + '/' 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

 


plt.figure(figsize=(15,10))
plt.plot(class_0_data_test[0, 0:-1],class_0_data_test[0, 1:], '*k', markersize = 10, label = "Master")
plt.plot(class_1_data_test[0, 0:-1],class_1_data_test[0, 1:], 'or', markersize = 10, label = "Slave")
# plt.plot(COUP_COEFF1,Sync_Error, '-or', markersize = 10, label = "Synchronization Error")
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel(r'$M_n$, $S_n$', fontsize=40)
plt.ylabel(r'$M_{n+1}$, $S_{n+1}$', fontsize=40)
plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/Attractor"+DATA_NAME_TEST+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/Attractor"+DATA_NAME_TEST+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
plt.show()




