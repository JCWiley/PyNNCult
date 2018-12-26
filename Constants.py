# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 12:29:20 2018

@author: John
"""

#set data variables
NUM_INPUTS = 4
NUM_OUTPUTS = 3

#stratify data.
#Must sum to the total number of vectors
HOLDOUT = 0
TRAINING = 120
VALIDATION = 30

#statistics variables
NUMTRIALS = 10

#set nn worker variables
WRKR_NUM_WORKERS = 20
WRKR_LOSS = 'mean_squared_error'
WRKR_MET = ['accuracy']
WRKR_EPOCHS = 100
WRKR_BATCH_SIZE = 1
WRKR_ACTIVATION = 'sigmoid'
WRKR_LEARNING_RATE = .7
WRKR_MOMENTUM = .1
WRKR_LYR_1 = 4
WRKR_LYR_2 = 8
WRKR_LYR_3 = self.NUM_OUTPUTS
WRKR_OPT = optimizers.SGD(lr=self.WRKR_LEARNING_RATE, decay=0, momentum=self.WRKR_MOMENTUM, nesterov=False)
        
#set lvq variables
LVQ_LEARNING_RATE = .3
LVQ_PREDICTED_NUM_CATAGORIES = 5
LVQ_ACTUAL_NUM_CATAGORIES = -1
        
#set nn master variables
MSTR_LOSS = 'mean_squared_error'
MSTR_MET = ['accuracy']
MSTR_EPOCHS = 100
MSTR_BATCH_SIZE = 1
MSTR_ACTIVATION = 'sigmoid'
MSTR_LEARNING_RATE = .7 #not currently used
MSTR_MOMENTUM = .1 #not currently used
MSTR_LYR_1 = 4
MSTR_LYR_2 = 8
MSTR_LYR_3 = self.LVQ_PREDICTED_NUM_CATAGORIES
MSTR_OPT = optimizers.SGD(lr=self.MSTR_LEARNING_RATE, decay=0, momentum=self.MSTR_MOMENTUM, nesterov=False)
        
#set nn variables
NN_LOSS = 'mean_squared_error'
NN_MET = ['accuracy']
NN_EPOCHS = 100
NN_BATCH_SIZE = 1
NN_ACTIVATION = 'sigmoid'
NN_LEARNING_RATE = .7
NN_MOMENTUM = .1
NN_LYR_1 = 4
NN_LYR_2 = 8
NN_LYR_3 = self.NUM_OUTPUTS
NN_OPT = optimizers.SGD(lr=self.LEARNING_RATE, decay=0, momentum=self.MOMENTUM, nesterov=False)
        