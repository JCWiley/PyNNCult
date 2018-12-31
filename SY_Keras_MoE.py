# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:33:28 2018

@author: Shawn Yuan with edits by John Wiley
"""


##set MoE variables
#MOE_LOSS = 'mean_squared_error'
#MOE_MET = ['accuracy']
#MOE_EPOCHS = 500
#MOE_BATCH_SIZE = 200
#MOE_EXPRT_ACTIVATION = 'sigmoid'
#MOE_GATE_ACTIVATION ='sigmoid'
#MOE_NUM_EXPERTS = 30
#MOE_LEARNING_RATE = .1
#MOE_MOMENTUM = .4
#MOE_OPT = optimizers.SGD(lr=MOE_LEARNING_RATE, decay=0, momentum=MOE_MOMENTUM, nesterov=False)
#MOE_DEBUG = 0


import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"

from keras.models import Model
from keras.layers import Dense, Input, Reshape, merge, Lambda
from keras import backend as K

#%%        
def MoE_slice(x,expert_num):
    return x[:,:,:expert_num]
#%%
def reduce(x, axis):
    return K.sum(x, axis=2)


class MoE:
    def __init__(self,I_Num_Inputs,I_Num_Outputs,I_MOE_NUM_EXPERTS,I_MOE_EXPRT_ACTIVATION,
                 I_GATE_ACTIVATION,I_MOE_BATCH_SIZE,I_MOE_LOSS,I_MOE_METRICS,I_MOE_OPTIMIZER,
                 I_MOE_DEBUG,I_MOE_EPOCHS):

        self.NUM_INPUTS = I_Num_Inputs
        self.NUM_OUTPUTS = I_Num_Outputs
        
        #set MoE variables
        self.MOE_NUM_EXPERTS = I_MOE_NUM_EXPERTS
        self.MOE_EXPRT_ACTIVATION = I_MOE_EXPRT_ACTIVATION
        self.MOE_GATE_ACTIVATION = I_GATE_ACTIVATION
        self.MOE_BATCH_SIZE = I_MOE_BATCH_SIZE
        self.MOE_LOSS = I_MOE_LOSS
        self.MOE_METRICS = I_MOE_METRICS
        self.MOE_OPTIMIZER = I_MOE_OPTIMIZER
        self.MOE_DEBUG = I_MOE_DEBUG
        self.MOE_EPOCHS = I_MOE_EPOCHS


#%%
    def Train(self,Training_Inputs,Training_Outputs,Holdout_Inputs,Holdout_Outputs):
        input_vector = Input(shape=(self.NUM_INPUTS,))
        
        expert_num = self.MOE_NUM_EXPERTS
        
        gate = Dense(self.NUM_OUTPUTS*(self.MOE_NUM_EXPERTS+1), activation=self.MOE_GATE_ACTIVATION)(input_vector)
        gate = Reshape((self.NUM_OUTPUTS, self.MOE_NUM_EXPERTS+1))(gate)
        gate = Lambda(MoE_slice, output_shape=(self.NUM_OUTPUTS, self.MOE_NUM_EXPERTS), arguments={'expert_num': expert_num})(gate)
        
        expert = Dense(self.NUM_OUTPUTS*self.MOE_NUM_EXPERTS, activation=self.MOE_EXPRT_ACTIVATION)(input_vector)
        expert = Reshape((self.NUM_OUTPUTS, self.MOE_NUM_EXPERTS))(expert)
        
        output = merge([gate, expert], mode='mul')
        output = Lambda(reduce, output_shape=(self.NUM_OUTPUTS,), arguments={'axis': 2})(output)
        
        self.model = Model(input=input_vector, output=output)
        
        self.model.compile(loss=self.MOE_LOSS, metrics=self.MOE_METRICS, optimizer=self.MOE_OPTIMIZER)
        self.model.fit(Training_Inputs, Training_Outputs, batch_size=self.MOE_BATCH_SIZE, verbose=self.MOE_DEBUG, nb_epoch=self.MOE_EPOCHS)
        
#%%
    def Execute(self,Inputs):
        return self.model.predict(Inputs,batch_size=None, verbose=0, steps=None)
 