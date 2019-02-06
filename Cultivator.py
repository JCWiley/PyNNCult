# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:29:40 2018

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

class Cultivator:
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
        
        #set lvq variables
        self.LVQ_LEARNING_RATE = I_LVQ_LEARNING_RATE
        self.LVQ_PREDICTED_NUM_CATAGORIES = I_LVQ_PREDICTED_NUM_CATAGORIES
        self.LVQ_ACTUAL_NUM_CATAGORIES = -1
    
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
    def Train(self,Training_Inputs,Training_Outputs):
                
        #start = timer()
        
        WRKR_ACC_Tuple = self.Train_Workers(Training_Inputs,Training_Outputs)
        
        #end = timer()

        #print("time to train workers")
        #print(end - start)
        
        self.Generate_Best_Worker_List(WRKR_ACC_Tuple[0],WRKR_ACC_Tuple[1])
        
        #start = timer()
        
        self.Train_Master(Training_Inputs,Training_Outputs)
        
        #end = timer()

        #print("time to train master")
        #print(end - start)
        #print("-------------End of Training Cycle--------------")
#%%
    def Execute(self,Inputs):
        
        Master_Out = self.Master.predict_on_batch(Inputs)
        
        Winner = np.argmax(Master_Out)
        return self.Trained_Networks[Winner].predict(Inputs,batch_size=None, verbose=0, steps=None)
    
#%%
    def Get_Worker_List(self):
        return self.Trained_Networks

#%%
    def Train_Workers(self,Training_Inputs,Training_Outputs):
        #train nn workers
        Worker_List = np.empty(self.NUM_WORKERS,dtype=object)
        Model_Accuracy = np.empty(self.NUM_WORKERS,dtype=float)
        
        for i in range(0,self.NUM_WORKERS):
            Model = Sequential()
            Model.add(Dense(self.WRKR_LYR_1,input_dim=self.NUM_INPUTS,activation=self.WRKR_ACTIVATION))
            #Model.add(Dense(self.WRKR_LYR_2,activation=self.WRKR_ACTIVATION))
            Model.add(Dense(self.WRKR_LYR_3,activation=self.WRKR_ACTIVATION))
            Model.compile(loss=self.WRKR_LOSS,optimizer=self.WRKR_OPT,metrics=self.WRKR_MET)
            Model.fit(Training_Inputs,Training_Outputs,epochs=self.WRKR_EPOCHS,batch_size=self.WRKR_BATCH_SIZE,verbose=DEBUG)
            Model_Accuracy[i] = Model.evaluate(x=Training_Inputs,y=Training_Outputs,verbose = DEBUG,batch_size = 128)[1]
            Worker_List[i] = Model
            #print("Worker " + str(i) + " Trained")
            
        #print('Model Accuracy',Model_Accuracy)
        return [Worker_List,Model_Accuracy]
#%%
    def Generate_Best_Worker_List(self,Worker_List,Accuracy):        
        Worker_Weights_List = np.empty(shape=self.NUM_WORKERS,dtype=object)
        for i in range(0,self.NUM_WORKERS):
            Worker_Weights_List[i] = Worker_List[i].get_weights()
            Worker_Weights_List[i] = Linearize(Worker_Weights_List[i])

        #run vbq on worker strings
        LVQ = LearningVectorQuantizer(self.LVQ_LEARNING_RATE)
        Classifications = np.empty(self.NUM_WORKERS)
        for i in range(0,self.NUM_WORKERS):
            Classifications[i] = LVQ.Execute(Worker_Weights_List[i],self.LVQ_PREDICTED_NUM_CATAGORIES)

        #find the best network for each lvq classification
        Keep_Index = np.full(self.LVQ_PREDICTED_NUM_CATAGORIES,-1,dtype=int)
        Max = 0
        Index = 0
        Flag = 0
        for i in range(0,self.LVQ_PREDICTED_NUM_CATAGORIES):
            for M in np.where(Classifications == i)[0]:
                #print('Catagory',i,'    Index',M,'    Accuracy',Accuracy[M],'    Max',Max)
                if(Accuracy[M] > Max):
                    #print('Max Update')
                    Max = Accuracy[M]
                    Keep_Index[Index] = M
                    Flag = 1
            if(Flag == 1):
                Index = Index+1
                Flag = 0
            Max = 0
        #copy the chosen networks to a new array, discard the others
        self.LVQ_ACTUAL_NUM_CATAGORIES = self.LVQ_PREDICTED_NUM_CATAGORIES - np.count_nonzero(Keep_Index == -1)
        #print('LVQRC ',self.LVQ_ACTUAL_NUM_CATAGORIES)
        self.Trained_Networks = np.empty(self.LVQ_ACTUAL_NUM_CATAGORIES,dtype=object)
        #print('Keep Index',Keep_Index)
        for i in range(0,Keep_Index.size):
            if(Keep_Index[i] != -1):
                self.Trained_Networks[i] = Worker_List[Keep_Index[i]]
        #print("Best Worker List Generated")
#%%
    def Train_Master(self,Training_Inputs,Training_Outputs):
        #create the training set for master
        Performance_Record = np.zeros([Training_Inputs.shape[0],self.LVQ_ACTUAL_NUM_CATAGORIES],dtype=int)
        for i in range(0,Training_Inputs.shape[0]): #for each training input
            Best_Index = -1
            Best_Accuracy = 1000
            for j in range(0,self.LVQ_ACTUAL_NUM_CATAGORIES): #for each expert
                Error = self.Trained_Networks[j].predict(x=np.array([Training_Inputs[i],]),verbose = DEBUG,batch_size = None)
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
        self.Master.add(Dense(self.LVQ_ACTUAL_NUM_CATAGORIES,activation=self.MSTR_ACTIVATION))
        self.Master.compile(loss=self.MSTR_LOSS,optimizer=self.MSTR_OPT,metrics=self.MSTR_MET)
        
        #train master
#%%
        #print("1")
        self.Master.fit(Training_Inputs,Performance_Record,epochs=self.MSTR_EPOCHS,batch_size=self.MSTR_BATCH_SIZE,verbose=DEBUG)
        #print("2")
        #print("Master Training Complete")
    