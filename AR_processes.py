#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:46:28 2021

@author: harikrishnan
"""


import numpy as np
import matplotlib.pyplot as plt

a = 0.9
b = 0.8
NOISE_INTENSITY = 0.01

X = []#np.array([],dtype=float)
Y = []#np.array([],dtype=float)

for COUP_COEFF in np.arange(0.8, 0.9, 0.1):
    X.append(0.1)
    Y.append(0.2)
    
    for INDEX in range(1, 2000):
        X.append(a*X[ INDEX - 1 ] + COUP_COEFF*Y[ INDEX - 1 ] + NOISE_INTENSITY * np.random.randn(1))
        Y.append(b*Y[INDEX-1] + NOISE_INTENSITY * np.random.randn(1))
        
        
plt.figure(figsize=(15,10))
plt.plot(X,'-k', markersize = 10, label='Effect')
plt.plot(Y,'-r', markersize = 10, label='Cause')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Time', fontsize=30)
plt.ylabel('Amplitude', fontsize=30)
#plt.ylim(0, 1)
plt.legend(fontsize = 30)
plt.tight_layout()
# plt.savefig(RESULT_PATH+"/Chaosnet-"+DATA_NAME+"-SR_plot.jpg", format='jpg', dpi=200)
plt.show()