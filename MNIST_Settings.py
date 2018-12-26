# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:29:20 2018

@author: John
"""

from keras import optimizers
from mnist import MNIST
import numpy as np

#represent data using mfcc
#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
#the sum of the NumTraining and NumTesting values must equal NumVectors
DataFilePath = 'C:\\Users\\John\\Stuff\\NN Cultivator\\MNIST Data'

ManagerOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MNISTManagerResult.txt"
NNOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MNISTNNResult.txt"
MoEOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MNISTMoEResult.txt
TrueOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\MNISTTrueResult.txt"

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
        
#%% Import data from file
mndata = MNIST('C:\\Users\\John\\Stuff\\NN Cultivator\\MNIST Data')
Tr_Images,Tr_Labels = mndata.load_training()
Te_Images,Te_Labels = mndata.load_testing()
Images = Tr_Images + Te_Images
Labels = Tr_Labels + Te_Labels
print(np.shape(Tr_Images))
print(np.shape(Te_Images))
print(np.shape(Tr_Labels))
print(np.shape(Te_Labels))
print(np.shape(Images))

print(np.unique(Tr_Labels))

One_Hot_Labels = np.zeros((TOTAL,NUM_OUTPUTS))
One_Hot_Labels[np.arange(TOTAL),Labels] = 1

print(np.shape(One_Hot_Labels))

dataset = np.zeros((TOTAL,NUM_INPUTS+NUM_OUTPUTS))

np.concatenate((Images,One_Hot_Labels),axis=1,out=dataset)

#for x in range(0,np.shape(Images)[0]):
#    dataset[x] = Images[x] + One_Hot_Labels[x]
    
print(np.shape(dataset))
print("MNIST Load Complete")
#dataset = np.loadtxt(DataFileName, delimiter=DataFileDelimiter)
#Data_Length = np.size(dataset,0)