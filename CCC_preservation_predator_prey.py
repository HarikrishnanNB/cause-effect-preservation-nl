#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Harikirshnan NB
Dtd: 27 January 2022

Didinium eats Paramecium, Check if there is a higher causal influence from Didinium (Data- C10:C71) to Paramecium (Data- B:10 - B:71) 
than in the opposite direction. Remove the first 9 values as transients.

For Raw data, CCC parameters: L = 40, w = 15, δ = 4, B = 8

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ETC import CCC


PATH = os.getcwd()
DATA_NAME = 'PREY_PREDATOR'
data = np.array(pd.read_csv("DATA/"+DATA_NAME+"/prey_predator_final.csv", header=None))

didinium = data[9:,1]
paramecium = data[9:,0]

didinium_norm = (didinium -  np.min(didinium))/(np.max(didinium)-np.min(didinium))
paramecium_norm = (paramecium -  np.min(paramecium))/(np.max(paramecium)-np.min(paramecium))


M_causes_S = np.zeros(1)
S_causes_M = np.zeros(1)
ccc_M_S = CCC.compute(paramecium_norm, didinium_norm, LEN_past=40, ADD_meas=15, STEP_size=4, n_partitions=8)
ccc_S_M = CCC.compute(didinium_norm, paramecium_norm, LEN_past=40, ADD_meas=15, STEP_size=4, n_partitions=8)


M_causes_S[0] = ccc_M_S 
S_causes_M[0] = ccc_S_M

RESULT_PATH_FINAL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 

print("Causality in the correct direction for raw data = ", M_causes_S[0])
 
print("Causality in the opposite direction for raw data = ", S_causes_M[0])

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)


# Saving Results

np.save(RESULT_PATH_FINAL +'CCC_prey_predator_raw_data_M_causes_S.npy', M_causes_S)
np.save(RESULT_PATH_FINAL +'CCC_prey_predator_raw_data_M_S_causes_M.npy', S_causes_M)

##########################################################################################


M_causes_S_chaosfex = np.zeros(1)
S_causes_M_chaosfex = np.zeros(1)


import ChaosFEX.feature_extractor as CFX


TOTAL_DATA = np.zeros((2, len(didinium)))
TOTAL_DATA[0,:] = didinium_norm
TOTAL_DATA[1,:] = paramecium_norm


INA = 0.56
DT = 0.499
EPSILON_1 = 0.1

# ChaosFEX feature Extraction
chaosfex_data = CFX.transform(TOTAL_DATA, INA, 10000, EPSILON_1, DT)[:, 2*62: 2*62+62]


ccc_M_S_chaosfex = CCC.compute(chaosfex_data[1, :], chaosfex_data[0, :], LEN_past=40, ADD_meas=15, STEP_size=4, n_partitions=4)
ccc_S_M_chaosfex = CCC.compute(chaosfex_data[0, :], chaosfex_data[1, :], LEN_past=40, ADD_meas=15, STEP_size=4, n_partitions=4)


M_causes_S_chaosfex[0] = ccc_M_S_chaosfex 
S_causes_M_chaosfex[0] = ccc_S_M_chaosfex

RESULT_PATH_FINAL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 

print("Causality in the correct direction for Chaosfex data = ", M_causes_S_chaosfex[0])
 
print("Causality in the opposite direction for Chaosfex data = ", S_causes_M_chaosfex[0])

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)


# Saving Results

np.save(RESULT_PATH_FINAL +'CCC_prey_predator_Chaosfex_M_causes_S.npy', M_causes_S_chaosfex)
np.save(RESULT_PATH_FINAL +'CCC_prey_predator_Chaosfex_M_S_causes_M.npy', S_causes_M_chaosfex)

from scipy import io

io.savemat(RESULT_PATH_FINAL + 'class_0_indep_firing_time.mat', {'class_0_indep_firing_time': chaosfex_data[0, :]})
io.savemat(RESULT_PATH_FINAL + 'class_1_dep_firing_time.mat', {'class_1_dep_firing_time': chaosfex_data[1, :]})

plt.figure(figsize=(15,10))
plt.plot(didinium, '-*k', markersize = 10, label = "Predator (Didinium)")
plt.plot(paramecium, '-or', markersize = 10, label = "Prey (Paramecium)")
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Time (days)', fontsize=40)
plt.ylabel('Abundance (#/ml)', fontsize=40)
plt.ylim(0, 400)
plt.legend(loc='upper left', fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/"+DATA_NAME+"-plot.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/"+DATA_NAME+"-plot.eps", format='eps', dpi=300)
plt.show()