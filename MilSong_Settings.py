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
DataFilePath = r"C:\Users\John\Stuff\NN Cultivator\MilSong\MillionSongSubset"

ManagerOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongManagerResult.txt"
NNOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongNNResult.txt"
MoEOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongMoEResult.txt
TrueOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MilSongTrueResult.txt"

ext='.h5'

#%%
#set data variables
NUM_INPUTS = 4
NUM_OUTPUTS = 13

#stratify data.
#Must sum to the total number of vectors
HOLDOUT = 0
TRAINING = 7000
VALIDATION = 3000
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
WRKR_LYR_1 = 50
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
MSTR_LYR_1 = 50
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
NN_LYR_1 = 50
NN_LYR_2 = 8 #not used
NN_LYR_3 = NUM_OUTPUTS
NN_OPT = optimizers.SGD(lr=NN_LEARNING_RATE, decay=0, momentum=NN_MOMENTUM, nesterov=False)

#set MoE variables
MoE_I = 10
MoE_Lamda = .1
MoE_K = 4
MoE_Lazy = .99
MoE_Type = 'classification'

#%%
dataset = np.zeros((TOTAL,NUM_INPUTS+NUM_OUTPUTS))

titles = []
for root, dirs, files in os.walk(DataFilePath):
    files = glob.glob(os.path.join(root,'*'+ext))
    index = 0
    for f in files:
        h5 = hdf5_getters.open_h5_file_read(f)
        if hdf5_getters.get_num_songs(h5) == 1:
            Fade = hdf5_getters.get_end_of_fade_in(h5)
            Loudness = hdf5_getters.get_loudness(h5)
            Time_Sig = hdf5_getters.get_time_signature(h5)
            Key = hdf5_getters.get_key(h5)
            Year = hdf5_getters.get_year(h5)
            if(Year != 0):
                dataset[index,0] = Fade
                dataset[index,1] = Loudness
                dataset[index,2] = Time_Sig
                dataset[index,3] = Key
                if(Year < 1900):
                    Decade = 0
                elif(Year < 1910):
                    Decade = 1
                elif(Year < 1920):
                    Decade = 2
                elif(Year < 1930):
                    Decade = 3
                elif(Year < 1940):
                    Decade = 4
                elif(Year < 1950):
                    Decade = 5
                elif(Year < 1960):
                    Decade = 6
                elif(Year < 1970):
                    Decade = 7
                elif(Year < 1980):
                    Decade = 8
                elif(Year < 1990):
                    Decade = 9
                elif(Year < 2000):
                    Decade = 10
                elif(Year < 2010):
                    Decade = 11
                elif(Year < 2020):
                    Decade = 12
                else:
                    Decade = 0
                dataset[index,Decade+4] = 1
                index = index+1
        #print(index)
        h5.close()
        
print(dataset.shape)
print("Milsong Load Complete")
        
        
        
        
        
        
        
        
        
        
        
        
        