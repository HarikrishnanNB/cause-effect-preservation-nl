#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:42:22 2022

@author: harikrishnan
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import io



PATH = os.getcwd()
DATA_NAME = 'AR-1'
RESULT_PATH_NL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 

# Loading the normalized Raw Data
GC_ChaosFEX = io.loadmat(RESULT_PATH_NL+ 'GC_res_AR_fir_time.mat')
GC_rawdata = io.loadmat(RESULT_PATH_NL + 'GC_res_AR_raw_data.mat')
GC_DL = io.loadmat(RESULT_PATH_NL + 'GC_res_AR_DL_features.mat')
GC_CNN = io.loadmat(RESULT_PATH_NL+'GC_res_momax20_AR_CNN_features')
GC_LSTM = io.loadmat(RESULT_PATH_NL+'GC_res_momax20_AR_LSTM_features')



M_causes_S_ChaosFEX = GC_ChaosFEX['F_mean_right']
S_causes_M_ChaosFEX = GC_ChaosFEX['F_mean_opp']
        




M_causes_S_RD = GC_rawdata['F_mean_right']
S_causes_M_RD = GC_rawdata['F_mean_opp']


M_causes_S_DL = GC_DL['F_mean_right']
S_causes_M_DL = GC_DL['F_mean_opp']



M_causes_S_CNN = GC_CNN['F_mean_right']
S_causes_M_CNN = GC_CNN['F_mean_opp']

M_causes_S_LSTM = GC_LSTM['F_mean_right']
S_causes_M_LSTM = GC_LSTM['F_mean_opp']
        

Coupling_Coeff = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S_ChaosFEX[0,:], '-*k', markersize = 10, label = "M causes S (Firing Time)")
plt.plot(Coupling_Coeff, S_causes_M_ChaosFEX[0,:], '-or', markersize = 10, label = "S causes M (Firing Time)")

# plt.plot(Coupling_Coeff, M_causes_S_RD[0,:], '--*k', markersize = 10, label = "M causes S rawdata")
# plt.plot(Coupling_Coeff, S_causes_M_RD[0,:], '--or', markersize = 10, label = "S causes M rawdata")

plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('GC (F-statistic)', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_NL+"/NL-"+DATA_NAME+"-GC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_NL+"/NL-"+DATA_NAME+"-GC.eps", format='eps', dpi=300)
plt.show()


plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S_DL[0,:], '-*k', markersize = 10, label = "M causes S (DL)")
plt.plot(Coupling_Coeff, S_causes_M_DL[0,:], '-or', markersize = 10, label = "S causes M (DL)")

# plt.plot(Coupling_Coeff, M_causes_S_RD[0,:], '--*k', markersize = 10, label = "M causes S rawdata")
# plt.plot(Coupling_Coeff, S_causes_M_RD[0,:], '--or', markersize = 10, label = "S causes M rawdata")

plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('GC (F-statistic)', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_NL+"/DL-"+DATA_NAME+"-GC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_NL+"/DL-"+DATA_NAME+"-GC.eps", format='eps', dpi=300)
plt.show()

# CNN

plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S_CNN[0,:], '-*k', markersize = 10, label = "M causes S (CNN)")
plt.plot(Coupling_Coeff, S_causes_M_CNN[0,:], '-or', markersize = 10, label = "S causes M (CNN)")

# plt.plot(Coupling_Coeff, M_causes_S_RD[0,:], '--*k', markersize = 10, label = "M causes S rawdata")
# plt.plot(Coupling_Coeff, S_causes_M_RD[0,:], '--or', markersize = 10, label = "S causes M rawdata")

plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('GC (F-statistic)', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_NL+"/CNN-"+DATA_NAME+"-GC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_NL+"/CNN-"+DATA_NAME+"-GC.eps", format='eps', dpi=300)
plt.show()

# LSTM


plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S_LSTM[0,:], '-*k', markersize = 10, label = "M causes S (LSTM)")
plt.plot(Coupling_Coeff, S_causes_M_LSTM[0,:], '-or', markersize = 10, label = "S causes M (LSTM)")

# plt.plot(Coupling_Coeff, M_causes_S_RD[0,:], '--*k', markersize = 10, label = "M causes S rawdata")
# plt.plot(Coupling_Coeff, S_causes_M_RD[0,:], '--or', markersize = 10, label = "S causes M rawdata")

plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('GC (F-statistic)', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_NL+"/LSTM-"+DATA_NAME+"-GC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_NL+"/LSTM-"+DATA_NAME+"-GC.eps", format='eps', dpi=300)
plt.show()
