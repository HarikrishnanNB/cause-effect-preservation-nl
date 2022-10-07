
"""
Author: Harikrishnan NB
Email ID: harikrishnannb@07@gmail.com
Date: 22 Jan 2022

Plotting the NL and DL result for DATA_NAME = UNI-DIR-COUP-TENT
Plotting the CCC vs coupling coefficient for chaos feature extracted firing time.
"""
import os
import numpy as np
import matplotlib.pyplot as plt



PATH = os.getcwd()
DATA_NAME = 'AR-1'
RESULT_PATH_NL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 
F1_SCORE_NL = np.load(RESULT_PATH_NL +'F1-score.npy')

Synchronization_Error = np.load(RESULT_PATH_NL +'sync_error.npy')

RESULT_PATH_DL = PATH + '/' +'DL-RESULTS' + '/'+ DATA_NAME + '/' 
F1_SCORE_DL =  np.load(RESULT_PATH_DL +'F1-score.npy')



RESULT_PATH_CNN = PATH + '/' +'CNN-RESULTS' + '/'+ DATA_NAME + '/' 
F1_SCORE_CNN =  np.load(RESULT_PATH_CNN +'F1-score.npy')


RESULT_PATH_LSTM = PATH + '/' +'LSTM-RESULTS' + '/'+ DATA_NAME + '/' 
F1_SCORE_LSTM =  np.load(RESULT_PATH_LSTM +'F1-score.npy')

Coupling_Coeff = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


RESULT_PATH_FINAL = PATH + '/' +'NL-RESULTS' + '/'+ DATA_NAME + '/' 


 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)


if DATA_NAME  == 'AR-1':
    Coupling_Coeff = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    plt.figure(figsize=(18,10))
    plt.plot(Coupling_Coeff, F1_SCORE_NL, '-*k', markersize = 10, label = "ChaosNet F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_DL, '-sb', markersize = 10, label = "DL F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_CNN, '--k', markersize = 12, label = "CNN F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_LSTM, '-^y', markersize = 12, label = "LSTM F1-Score")
    plt.plot(Coupling_Coeff,Synchronization_Error, '-or', markersize = 10, label = "MSE")
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(True)
    plt.xlabel('Coupling Coefficient', fontsize=40)
    plt.ylabel('F1-score, MSE', fontsize=40)
    plt.ylim(-0.1, 1.1)
    # plt.legend(bbox_to_anchor=(0.9,0.1), fontsize = 25)
    plt.legend(loc="lower right", fontsize=25)
    plt.tight_layout()
    plt.savefig(RESULT_PATH_FINAL+"/Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
    plt.savefig(RESULT_PATH_FINAL+"/Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
    plt.show()
    
else:

    plt.figure(figsize=(18,10))
    plt.plot(Coupling_Coeff, F1_SCORE_NL, '-*k', markersize = 14, label = "ChaosNet F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_DL, '-sb', markersize = 12, label = "DL F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_CNN, '--k', markersize = 12, label = "CNN F1-Score")
    plt.plot(Coupling_Coeff, F1_SCORE_LSTM, '-^y', markersize = 12, label = "LSTM F1-Score")
    plt.plot(Coupling_Coeff,Synchronization_Error, '-or', markersize = 10, label = "Synchronization Error")
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(True)
    plt.xlabel('Coupling Coefficient', fontsize=40)
    plt.ylabel('F1-score, Synchronization Error', fontsize=40)
    plt.ylim(-0.1, 1.1)
    # plt.legend(bbox_to_anchor=(0.635,0.3), fontsize = 25)
    plt.legend(fontsize = 25)
    plt.tight_layout()
    plt.savefig(RESULT_PATH_FINAL+"/Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.jpg", format='jpg', dpi=300)
    plt.savefig(RESULT_PATH_FINAL+"/Chaosnet_DL-"+DATA_NAME+"-F1_score_vs_coup_coeff.eps", format='eps', dpi=300)
    plt.show()

#### Causal Preservation


M_causes_S = np.load(RESULT_PATH_NL +'NL_CCC_firing_time_M_causes_S.npy')
S_causes_M = np.load(RESULT_PATH_NL +'NL_CCC_firing_time_S_causes_M.npy')

plt.figure(figsize=(15,10))
plt.plot(Coupling_Coeff, M_causes_S, '-*k', markersize = 10, label = "M causes S (Firing Time)")
plt.plot(Coupling_Coeff, S_causes_M, '-or', markersize = 10, label = "S causes M (Firing Time)")
plt.xticks(fontsize=45)
plt.yticks(fontsize=45)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=40)
plt.ylabel('CCC', fontsize=40)
# plt.ylim(0, 1.1)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"/NL-"+DATA_NAME+"-CCC.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"/NL-"+DATA_NAME+"-CCC.eps", format='eps', dpi=300)
plt.show()
