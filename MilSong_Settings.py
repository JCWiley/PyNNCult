# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:58:32 2018

@author: John
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:29:20 2018

@author: John
"""
import os
import glob
import hdf5_getters

from keras import optimizers
from mnist import MNIST
import numpy as np
#%%

#represent data using mfcc
#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
#the sum of the NumTraining and NumTesting values must equal NumVectors
DataFilePath = 'C:\Users\John\Stuff\NN Cultivator\MilSong\MillionSongSubset'

ManagerOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongManagerResult.txt"
NNOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongNNResult.txt"
TrueOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongTrueResult.txt"

ext='.h5'

#%%
#set data variables
NUM_INPUTS = 784
NUM_OUTPUTS = 10

#stratify data.
#Must sum to the total number of vectors
HOLDOUT = 0
TRAINING = 60000
VALIDATION = 10000
TOTAL = HOLDOUT + TRAINING + VALIDATION

#statistics variables
NUMTRIALS = 1

#set nn worker variables
WRKR_NUM_WORKERS =30
WRKR_LOSS = 'mean_squared_error'
WRKR_MET = ['accuracy']
WRKR_EPOCHS = 500
WRKR_BATCH_SIZE = 1
WRKR_ACTIVATION = 'sigmoid'
WRKR_LEARNING_RATE = .1
WRKR_MOMENTUM = .4
WRKR_LYR_1 = 500
WRKR_LYR_2 = 8 #not used
WRKR_LYR_3 = NUM_OUTPUTS
WRKR_OPT = optimizers.SGD(lr=WRKR_LEARNING_RATE, decay=0, momentum=WRKR_MOMENTUM, nesterov=False)
        
#set lvq variables
LVQ_LEARNING_RATE = .3
LVQ_PREDICTED_NUM_CATAGORIES = 10
        
#set nn master variables
MSTR_LOSS = 'mean_squared_error'
MSTR_MET = ['accuracy']
MSTR_EPOCHS = 500
MSTR_BATCH_SIZE = 1
MSTR_ACTIVATION = 'sigmoid'
MSTR_LEARNING_RATE = .1
MSTR_MOMENTUM = .4
MSTR_LYR_1 = 500
MSTR_LYR_2 = 8 #not used
MSTR_LYR_3 = LVQ_PREDICTED_NUM_CATAGORIES
MSTR_OPT = optimizers.SGD(lr=MSTR_LEARNING_RATE, decay=0, momentum=MSTR_MOMENTUM, nesterov=False)
        
#set nn variables
NN_LOSS = 'mean_squared_error'
NN_MET = ['accuracy']
NN_EPOCHS = 500
NN_BATCH_SIZE = 1
NN_ACTIVATION = 'sigmoid'
NN_LEARNING_RATE = .1
NN_MOMENTUM = .4
NN_LYR_1 = 500
NN_LYR_2 = 8 #not used
NN_LYR_3 = NUM_OUTPUTS
NN_OPT = optimizers.SGD(lr=NN_LEARNING_RATE, decay=0, momentum=NN_MOMENTUM, nesterov=False)

#%%
titles = []
for root, dirs, files in os.walk(basedir):
    files = glob.glob(os.path.join(root,'*'+ext))
    for f in files:
        h5 = hdf5_getters.open_h5_file_read(f)
        print(hdf5_getters.get_title(h5))
        titles.append( hdf5_getters.get_title(h5) )
        h5.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        