# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:29:40 2018

@author: John
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import theano
import keras
import numpy as np
from keras import losses


from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

class MLP:
    def __init__(self,I_Num_Inputs,I_Num_Outputs,I_LOSS,I_MET,I_EPOCHS,I_BATCH_SIZE,I_ACTIVATION,I_LEARNING_RATE,I_MOMENTUM,I_LYR_1,I_LYR_2,I_LYR_3,I_OPT):  
        self.NUM_INPUTS = I_Num_Inputs
        self.NUM_OUTPUTS = I_Num_Outputs
        #set nn variables
        self.LOSS = I_LOSS
        self.MET = I_MET
        self.EPOCHS = I_EPOCHS
        self.BATCH_SIZE = I_BATCH_SIZE
        self.ACTIVATION = I_ACTIVATION
        self.LEARNING_RATE =I_LEARNING_RATE
        self.MOMENTUM = I_MOMENTUM
        self.LYR_1 = I_LYR_1
        self.LYR_2 = I_LYR_2
        self.LYR_3 = I_LYR_3
        self.OPT = I_OPT
#%%
    def Train(self,Training_Inputs,Training_Outputs,Holdout_Inputs,Holdout_Outputs):
        self.Model = Sequential()
        self.Model.add(Dense(self.LYR_1,input_dim=self.NUM_INPUTS,activation=self.ACTIVATION))
        #Model.add(Dense(self.LYR_2,activation=self.ACTIVATION))
        self.Model.add(Dense(self.LYR_3,activation=self.ACTIVATION))
        self.Model.compile(loss=self.LOSS,optimizer=self.OPT,metrics=self.MET)
        self.Model.fit(Training_Inputs,Training_Outputs,epochs=self.EPOCHS,batch_size=self.BATCH_SIZE,verbose=DEBUG)
#%%
    def Execute(self,Inputs):
        return self.Model.predict(Inputs,batch_size=None, verbose=0, steps=None)
