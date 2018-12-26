# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#imports
import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import theano
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense

exec(open("LVQ.py").read())
exec(open("Tools.py").read())
exec(open("Cultivator.py").read())
#-------------------------------
#set system variables
PRINT_DEBUG = 1
DEBUG = 0

DataFileName = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\iris.txt"
DataFileDelimiter = " "
ManagerOutputFile = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\irisManagerResult.txt"
NNOutputFile = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\irisNNResult.txt"


#set data variables
NUM_INPUTS = 4
NUM_OUTPUTS = 3

#stratify data (ie take percentages of output catagories)
HOLDOUT = .1
VALIDATION = .3
TRAINING = .6

#represent data using mfcc
#http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
NUM_TRAINING = 120 #the sum of the NumTraining and NumTesting values must equal NumVectors
NUM_TESTING = 30
NUM_VECTORS = 150
NUM_TRIALS = 2

#set nn worker variables
NUM_WORKERS = 30
WRKR_LOSS = 'binary_crossentropy'
WRKR_OPT = 'adam'
WRKR_MET = ['accuracy']
WRKR_EPOCHS = 150
WRKR_BATCH_SIZE = 10
WRKR_ACTIVATION = 'sigmoid'
WRKR_LEARNING_RATE = .7 #not currently used
WRKR_MOMENTUM = .1 #not currently used
WRKR_LYR_1 = 10
WRKR_LYR_2 = 8
WRKR_LYR_3 = NUM_OUTPUTS
#set lvq variables
LVQ_LEARNING_RATE = .8
LVQ_INPUT_COUNT = ((NUM_INPUTS*WRKR_LYR_1)+WRKR_LYR_1)+((WRKR_LYR_1*WRKR_LYR_2)+WRKR_LYR_2)+((WRKR_LYR_2*WRKR_LYR_3)+WRKR_LYR_3)
LVQ_PREDICTED_NUM_CATAGORIES = 5
LVQ_MOMENTUM = .2

#set nn master variables
MSTR_LOSS = 'binary_crossentropy'
MSTR_OPT = 'adam'
MSTR_MET = ['accuracy']
MSTR_EPOCHS = 150
MSTR_BATCH_SIZE = 10
MSTR_ACTIVATION = 'sigmoid'
MSTR_LEARNING_RATE = .7 #not currently used
MSTR_MOMENTUM = .1 #not currently used
MSTR_LYR_1 = 10
MSTR_LYR_2 = 8
MSTR_LYR_3 = LVQ_PREDICTED_NUM_CATAGORIES
#%%
#import data
dataset = np.loadtxt(DataFileName, delimiter=DataFileDelimiter)
#format data
# split into input (X) and output (Y) variables
X = dataset[:,0:NUM_INPUTS]
Y = dataset[:,NUM_INPUTS:NUM_INPUTS+NUM_OUTPUTS]
#%%
#train nn workers
Worker_List = np.empty(NUM_WORKERS,dtype=object)
Model_Accuracy = np.empty(NUM_WORKERS,dtype=float)

for i in range(0,NUM_WORKERS):
    Model = Sequential()
    Model.add(Dense(WRKR_LYR_1,input_dim=NUM_INPUTS,activation=WRKR_ACTIVATION))
    Model.add(Dense(WRKR_LYR_2,activation=WRKR_ACTIVATION))
    Model.add(Dense(WRKR_LYR_3,activation=WRKR_ACTIVATION))
    Model.compile(loss=WRKR_LOSS,optimizer=WRKR_OPT,metrics=WRKR_MET)
    Model.fit(X,Y,epochs=WRKR_EPOCHS,batch_size=WRKR_BATCH_SIZE,verbose=DEBUG)
    Model_Accuracy[i] = Model.evaluate(x=X,y=Y,verbose = DEBUG,batch_size = 128)[1]
    Worker_List[i] = Model

#%%
#convert nn workers to weight lists
    
Worker_Weights_List = np.empty(shape=(NUM_WORKERS,LVQ_INPUT_COUNT),dtype=float)#np.empty(NUM_WORKERS,dtype=object)
pos = 0
for i in range(0,NUM_WORKERS):
    pos = 0
    for item_one in Worker_List[i].get_weights():
        for item_two in item_one:
            try:
                iter(item_two)
                for item_three in item_two:
                    Worker_Weights_List[i][pos] = item_three
                    pos = pos + 1
            except TypeError:
                Worker_Weights_List[i][pos] = item_two
                pos = pos + 1
#%%
#run vbq on worker strings
LVQ = LearningVectorQuantizer(LVQ_LEARNING_RATE,LVQ_INPUT_COUNT,LVQ_PREDICTED_NUM_CATAGORIES,LVQ_MOMENTUM)
Classifications = np.empty(NUM_WORKERS)
for i in range(0,NUM_WORKERS):
    Classifications[i] = LVQ.Execute(Worker_Weights_List[i])
    
#%%
#find the best network for each lvq classification
print(Classifications)
print(np.unique(Classifications).size)
Keep_Index = np.empty(LVQ_PREDICTED_NUM_CATAGORIES,dtype=int)
Max = 0
Index = 0
Flag = 0
for i in range(0,LVQ_PREDICTED_NUM_CATAGORIES):
    Debug_Print(i)
    Debug_Print(np.where(Classifications == i)[0])
    for M in np.where(Classifications == i)[0]:
        if(Model_Accuracy[M] > Max):
            Max = Model_Accuracy[M]
            Keep_Index[Index] = M
            Flag = 1
    Debug_Print(Keep_Index)
    if(Flag == 1):
        Index = Index+1
        Flag = 0
    Max = 0
#%%
#remove all the networks that were not selected
Trained_Networks = np.empty(LVQ_PREDICTED_NUM_CATAGORIES,dtype=object)
for i in range(0,LVQ_PREDICTED_NUM_CATAGORIES):
    Trained_Networks[i] = Worker_List[Keep_Index[i]]

#%%

#create the training set for master
Performance_Record = np.zeros([X.shape[0],LVQ_PREDICTED_NUM_CATAGORIES],dtype=int)
for i in range(0,X.shape[0]):
    Best_Index = -1
    Best_Accuracy = -1.0
    for j in range(0,LVQ_PREDICTED_NUM_CATAGORIES):
        Accuracy = Trained_Networks[j].evaluate(x=X,y=Y,verbose = DEBUG,batch_size = 128)[1]
        if(Accuracy > Best_Accuracy):
            Debug_Print("Acc > B Acc")
            Debug_Print(Accuracy)
            Debug_Print(Best_Accuracy)
            Best_Accuracy = Accuracy
            Best_Index = j
    print(Best_Index)
    Debug_Print("Reset")
    Performance_Record[i][Best_Index] = 1
#%%
#create Master
Master = Sequential()
Master.add(Dense(MSTR_LYR_1,input_dim=NUM_INPUTS,activation=MSTR_ACTIVATION))
Master.add(Dense(MSTR_LYR_2,activation=MSTR_ACTIVATION))
Master.add(Dense(MSTR_LYR_3,activation=MSTR_ACTIVATION))
Master.compile(loss=MSTR_LOSS,optimizer=MSTR_OPT,metrics=MSTR_MET)

#%%
#train master
Master.fit(X,Performance_Record,epochs=MSTR_EPOCHS,batch_size=MSTR_BATCH_SIZE,verbose=DEBUG)