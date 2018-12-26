# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:33:28 2018

@author: John
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
import theano
import keras
import numpy as np
from keras import losses
from timeit import default_timer as timer

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

class MoE:
    def __init__(self,I_Num_Inputs,I_Num_Outputs,
        I_NUM_WORKERS,I_WRKR_LOSS,I_WRKR_MET,I_WRKR_EPOCHS,I_WRKR_BATCH_SIZE,
        I_WRKR_ACTIVATION,I_WRKR_LEARNING_RATE,I_WRKR_MOMENTUM,I_WRKR_LYR_1,
        I_WRKR_LYR_2,I_WRKR_LYR_3,I_WRKR_OPT,I_LVQ_LEARNING_RATE,I_LVQ_PREDICTED_NUM_CATAGORIES,
        I_MSTR_LOSS,I_MSTR_MET,I_MSTR_EPOCHS,I_MSTR_BATCH_SIZE,I_MSTR_ACTIVATION,I_MSTR_LEARNING_RATE,
        I_MSTR_MOMENTUM,I_MSTR_LYR_1,I_MSTR_LYR_2,I_MSTR_LYR_3,I_MSTR_OPT):

        self.NUM_INPUTS = I_Num_Inputs
        self.NUM_OUTPUTS = I_Num_Outputs
        #set nn worker variables
        self.NUM_WORKERS = I_NUM_WORKERS
        self.WRKR_LOSS = I_WRKR_LOSS
        self.WRKR_MET = I_WRKR_MET
        self.WRKR_EPOCHS = I_WRKR_EPOCHS
        self.WRKR_BATCH_SIZE = I_WRKR_BATCH_SIZE
        self.WRKR_ACTIVATION = I_WRKR_ACTIVATION
        self.WRKR_LEARNING_RATE =I_WRKR_LEARNING_RATE
        self.WRKR_MOMENTUM = I_WRKR_MOMENTUM
        self.WRKR_LYR_1 = I_WRKR_LYR_1
        self.WRKR_LYR_2 = I_WRKR_LYR_2
        self.WRKR_LYR_3 = I_WRKR_LYR_3
        self.WRKR_OPT = I_WRKR_OPT
    
        #set nn master variables
        self.MSTR_LOSS = I_MSTR_LOSS
        self.MSTR_MET = I_MSTR_MET
        self.MSTR_EPOCHS = I_MSTR_EPOCHS
        self.MSTR_BATCH_SIZE = I_MSTR_BATCH_SIZE
        self.MSTR_ACTIVATION = I_MSTR_ACTIVATION
        self.MSTR_LEARNING_RATE = I_MSTR_LEARNING_RATE
        self.MSTR_MOMENTUM = I_MSTR_MOMENTUM
        self.MSTR_LYR_1 = I_MSTR_LYR_1
        self.MSTR_LYR_2 = I_MSTR_LYR_2
        self.MSTR_LYR_3 = I_MSTR_LYR_3
        self.MSTR_OPT = I_MSTR_OPT
#%%
    def Train(self,Training_Inputs,Training_Outputs,Holdout_Inputs,Holdout_Outputs):
        self.Train_Experts(Training_Inputs,Training_Outputs)
        
        self.Train_Gating_Function(Holdout_Inputs,Holdout_Outputs)
#%%
    def Execute(self,Inputs):
        
        Master_Out = self.Master.predict_on_batch(Inputs)
        
        Result = 0
        
        for i in range(0,self.NUM_WORKERS):
            Result += Master_Out[i] * self.Worker_List[i].predict(Inputs,batch_size=None, verbose=0, steps=None)
        return Result
    
#%%
    def Get_Worker_List(self):
        return self.Trained_Networks

#%%
    def Train_Experts(self,Training_Inputs,Training_Outputs):
        #train nn workers
        self.Worker_List = np.empty(self.NUM_WORKERS,dtype=object)
        Model_Accuracy = np.empty(self.NUM_WORKERS,dtype=float)
        
        for i in range(0,self.NUM_WORKERS):
            Model = Sequential()
            Model.add(Dense(self.WRKR_LYR_1,input_dim=self.NUM_INPUTS,activation=self.WRKR_ACTIVATION))
            #Model.add(Dense(self.WRKR_LYR_2,activation=self.WRKR_ACTIVATION))
            Model.add(Dense(self.WRKR_LYR_3,activation=self.WRKR_ACTIVATION))
            Model.compile(loss=self.WRKR_LOSS,optimizer=self.WRKR_OPT,metrics=self.WRKR_MET)
            Model.fit(Training_Inputs,Training_Outputs,epochs=self.WRKR_EPOCHS,batch_size=self.WRKR_BATCH_SIZE,verbose=DEBUG)
            self.Worker_List[i] = Model

#%%
    def Train_Gating_Function(self,Training_Inputs,Training_Outputs):
        #create the training set for master
        Performance_Record = np.zeros([Training_Inputs.shape[0],self.NUM_WORKERS],dtype=int)
        for i in range(0,Training_Inputs.shape[0]): #for each training input
            Best_Index = -1
            Best_Accuracy = 1000
            for j in range(0,self.NUM_WORKERS): #for each expert
                Error = self.Worker_List[j].predict(x=np.array([Training_Inputs[i],]),verbose = DEBUG,batch_size = None)
                Accuracy = MSE(Training_Outputs[i], Error)
                #print('Accuracy',Accuracy,'   Training Element',i,'   Catagory',j)
                if(Accuracy < Best_Accuracy):
                    #print('Max Update')
                    Best_Accuracy = Accuracy
                    Best_Index = j
            Performance_Record[i][Best_Index] = 1
        #create Master
        self.Master = Sequential()
        self.Master.add(Dense(self.MSTR_LYR_1,input_dim=self.NUM_INPUTS,activation=self.MSTR_ACTIVATION))
        #self.Master.add(Dense(self.MSTR_LYR_2,activation=self.MSTR_ACTIVATION))
        self.Master.add(Dense(self.NUM_WORKERS,activation="softmax"))
        self.Master.compile(loss=self.MSTR_LOSS,optimizer=self.MSTR_OPT,metrics=self.MSTR_MET)
        
        #train master
#%%
        print("1")
        self.Master.fit(Training_Inputs,Performance_Record,epochs=self.MSTR_EPOCHS,batch_size=self.MSTR_BATCH_SIZE,verbose=DEBUG)
        print("2")
        #print("Master Training Complete")
    