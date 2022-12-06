#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022


This is the code to plot manifold learning results using ChaosNet and DL.

2) UNI-DIR-COUP-TENT-DATA-2 (b1 = 0.6, b2 = 0.4)
3) UNI-DIR-COUP-TENT-DATA-3 (b1 = 0.1, b2 = 0.3)
4) UNI-DIR-COUP-TENT-DATA-4 (b1 = 0.49, b2 = 0.52)
5) UNI-DIR-COUP-LOGISTIC-MAP-DATA-1 (A1 = 4.0, A2 = 3.82)



"""
import os
import numpy as np
import matplotlib.pyplot as plt





PATH = os.getcwd()
DATA_NAME = 'UNI-DIR-COUP-LOGISTIC-MAP-DATA-1'
RESULT_PATH_NL = PATH + '/' +'MANIFOLD-LEARNING-RESULTS' + '/'+ DATA_NAME + '/' 
F1_SCORE_NL = np.load(RESULT_PATH_NL +'F1-score.npy')

Synchronization_Error = np.load(RESULT_PATH_NL +'sync_error.npy')

RESULT_PATH_DL = PATH + '/' +'MANIFOLD-LEARNING-RESULTS-DL' + '/'+ DATA_NAME + '/' 
F1_SCORE_DL =  np.load(RESULT_PATH_DL +'F1-score.npy')

Coupling_Coeff = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


RESULT_PATH_FINAL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 


 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, F1_SCORE_NL, '-*k', markersize = 10, label = "ChaosNet F1-Score")
plt.plot(Coupling_Coeff, F1_SCORE_DL, '-sb', markersize = 10, label = "DL F1-Score")
# plt.plot(Coupling_Coeff,Synchronization_Error, '-or', markersize = 10, label = "Synchronization Error")
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('F1-score', fontsize=40)
plt.ylim(-0.1, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/MAnifold_Learning_Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/Manifold_Learning_Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
plt.show()


'''
#### Causal PreservationNice.


M_causes_S = np.load(RESULT_PATH_NL +'NL_CCC_firing_time_M_causes_S.npy')
S_causes_M = np.load(RESULT_PATH_NL +'NL_CCC_firing_time_S_causes_M.npy')

plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S, '-*k', markersize = 10, label = "M causes S")
plt.plot(Coupling_Coeff, S_causes_M, '-or', markersize = 10, label = "S causes M")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('CCC', fontsize=30)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/NL-"+DATA_NAME+"-CCC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/NL-"+DATA_NAME+"-CCC.eps", format='eps', dpi=300)
plt.show()
'''