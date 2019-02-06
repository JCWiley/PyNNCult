# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:29:20 2018

@author: John
"""

from keras import optimizers

#represent data using mfcc
#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
#the sum of the NumTraining and NumTesting values must equal NumVectors
DataFileName = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\iris.txt"
#DataFileName = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\iris0.txt"
DataFileDelimiter = " "
ManagerOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\irisCultivatorResult.txt"
NNOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\irisNNResult.txt"
MoEOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\irisMoEResult.txt"
TrueOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\irisTrueResult.txt"
WorkerCompositeOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\WorkerCompositeNNResult"
InputOutputFile = r"C:\Users\John\Stuff\NN Cultivator\OutputData\irisTrueInput.txt"

#%%
#set data variables
NUM_INPUTS = 4
NUM_OUTPUTS = 3

#statistics variables
NUMTRIALS = 5

#stratify data.
#Must sum to the total number of vectors
TOTAL = 150
VALIDATION = TOTAL//NUMTRIALS
TRAINING = TOTAL-VALIDATION


#set nn worker variables
WRKR_NUM_WORKERS =30
WRKR_LOSS = 'mean_squared_error'
WRKR_MET = ['accuracy']
WRKR_EPOCHS = 50
WRKR_BATCH_SIZE = 30
WRKR_ACTIVATION = 'sigmoid'
WRKR_LEARNING_RATE = .1
WRKR_MOMENTUM = .1
WRKR_LYR_1 = 10
WRKR_LYR_2 = 8 #not used
WRKR_LYR_3 = NUM_OUTPUTS
WRKR_OPT = optimizers.SGD(lr=WRKR_LEARNING_RATE, decay=0, momentum=WRKR_MOMENTUM, nesterov=False)
        
#set lvq variables
LVQ_LEARNING_RATE = .3
LVQ_PREDICTED_NUM_CATAGORIES = 10
        
#set nn master variables
MSTR_LOSS = 'mean_squared_error'
MSTR_MET = ['accuracy']
MSTR_EPOCHS = 50
MSTR_BATCH_SIZE = 30
MSTR_ACTIVATION = 'sigmoid'
MSTR_LEARNING_RATE = .1
MSTR_MOMENTUM = .1
MSTR_LYR_1 = 10
MSTR_LYR_2 = 8 #not used
MSTR_LYR_3 = LVQ_PREDICTED_NUM_CATAGORIES
MSTR_OPT = optimizers.SGD(lr=MSTR_LEARNING_RATE, decay=0, momentum=MSTR_MOMENTUM, nesterov=False)
        
#set nn variables
NN_LOSS = 'mean_squared_error'
NN_MET = ['accuracy']
NN_EPOCHS = 50
NN_BATCH_SIZE = 30
NN_ACTIVATION = 'sigmoid'
NN_LEARNING_RATE = .1
NN_MOMENTUM = .1
NN_LYR_1 = 10
NN_LYR_2 = 8 #not used
NN_LYR_3 = NUM_OUTPUTS
NN_OPT = optimizers.SGD(lr=NN_LEARNING_RATE, decay=0, momentum=NN_MOMENTUM, nesterov=False)

#set MoE variables
MOE_LOSS = 'mean_squared_error'
MOE_MET = ['accuracy']
MOE_EPOCHS = 50
MOE_BATCH_SIZE = 30
MOE_EXPRT_ACTIVATION = 'linear'
MOE_GATE_ACTIVATION ='softmax'
MOE_NUM_EXPERTS = 30
MOE_LEARNING_RATE = .1
MOE_MOMENTUM = .1
MOE_OPT = optimizers.SGD(lr=MOE_LEARNING_RATE, decay=0, momentum=MOE_MOMENTUM, nesterov=False)
MOE_DEBUG = 0
        

#%% Import data from file
dataset = np.loadtxt(DataFileName, delimiter=DataFileDelimiter)
Data_Length = np.size(dataset,0)

print(dataset.shape)
print("Iris Load Complete")