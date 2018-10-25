# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:02:51 2018

@author: John
"""

#imports
import numpy as np


#%% Import the settings file for the current test
exec(open("Iris_Settings.py").read())
#exec(open("MNIST_Settings.py").read())

#%% Import all the tool files
exec(open("LVQ.py").read())
exec(open("Tools.py").read())
exec(open("Cultivator.py").read())
exec(open("MLP.py").read())
exec(open("MixtureOfExperts.py").read())
#-------------------------------
#set system variables
DEBUG = 0

#%% Define Result storage variables
C_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
MLP_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
MoE_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
Composite_Targets = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
Worker_Results = np.empty([WRKR_NUM_WORKERS,VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)


#%% start trials loop
for T in range(0,NUMTRIALS):
    #%%Split the data into sections according to its use
    #randomize data order
    np.random.shuffle(dataset)
    Split_Data = np.split(dataset,[HOLDOUT,TRAINING+HOLDOUT])
    
    #%%
    #format data
    # split into input (X) and output (Y) variables
    
    Hold  = np.split(Split_Data[0],[NUM_INPUTS],1)
    Train = np.split(Split_Data[1],[NUM_INPUTS],1)
    Vali  = np.split(Split_Data[2],[NUM_INPUTS],1)
    
    
    Holdout_X = Hold[0]
    Holdout_Y = Hold[1]
    
    Training_X = Train[0]
    Training_Y = Train[1]
    
    Validation_X = Vali[0]
    Validation_Y = Vali[1]
    #%%
    Cult = Cultivator(NUM_INPUTS,NUM_OUTPUTS,WRKR_NUM_WORKERS,WRKR_LOSS,WRKR_MET,
                      WRKR_EPOCHS,WRKR_BATCH_SIZE,WRKR_ACTIVATION,WRKR_LEARNING_RATE,
                      WRKR_MOMENTUM,WRKR_LYR_1,WRKR_LYR_2,WRKR_LYR_3,WRKR_OPT,
                      LVQ_LEARNING_RATE,LVQ_PREDICTED_NUM_CATAGORIES,
                      MSTR_LOSS,MSTR_MET,MSTR_EPOCHS,
                      MSTR_BATCH_SIZE,MSTR_ACTIVATION,MSTR_LEARNING_RATE,
                      MSTR_MOMENTUM,MSTR_LYR_1,MSTR_LYR_2,MSTR_LYR_3,MSTR_OPT)
    
    Cult.Train(Training_X,Training_Y,Training_X,Training_Y)
    #Cult.Train(Training_X,Training_Y,Holdout_X,Holdout_Y)
    #print("Cultivator Training Complete")
    #%%
    Perceptron = MLP(NUM_INPUTS,NUM_OUTPUTS,NN_LOSS,NN_MET,NN_EPOCHS,
                     NN_BATCH_SIZE,NN_ACTIVATION,NN_LEARNING_RATE,NN_MOMENTUM,
                     NN_LYR_1,NN_LYR_2,NN_LYR_3,NN_OPT)
    Perceptron.Train(Training_X,Training_Y,Holdout_X,Holdout_Y)
     #print("Perceptron Training Complete")
     
    #%%
    MixtureOfExperts = MoE(NUM_INPUTS,NUM_OUTPUTS,WRKR_NUM_WORKERS,WRKR_LOSS,WRKR_MET,
                      WRKR_EPOCHS,WRKR_BATCH_SIZE,WRKR_ACTIVATION,WRKR_LEARNING_RATE,
                      WRKR_MOMENTUM,WRKR_LYR_1,WRKR_LYR_2,WRKR_LYR_3,WRKR_OPT,
                      LVQ_LEARNING_RATE,LVQ_PREDICTED_NUM_CATAGORIES,
                      MSTR_LOSS,MSTR_MET,MSTR_EPOCHS,
                      MSTR_BATCH_SIZE,MSTR_ACTIVATION,MSTR_LEARNING_RATE,
                      MSTR_MOMENTUM,MSTR_LYR_1,MSTR_LYR_2,MSTR_LYR_3,MSTR_OPT)
    MixtureOfExperts.Train(Training_X,Training_Y,Holdout_X,Holdout_Y)
    #%% NN Exeute Loop
    for i in range(0,VALIDATION):
        C_Results[(T*VALIDATION)+i] = Cult.Execute(np.array([Validation_X[i],]))
        MLP_Results[(T*VALIDATION)+i] = Perceptron.Execute(np.array([Validation_X[i],]))
        MoE_Results[(T*VALIDATION)+i] = MixtureOfExperts.Execute(np.array([Validation_X[i],]))
        Composite_Targets[(T*VALIDATION)+i] = Validation_Y[i]
        
        
        Worker_List = Cult.Get_Worker_List()
        Num_Workers = np.size(Worker_List)
        for j in range(0,Num_Workers):
            Worker_Results[j,(T*VALIDATION)+i] = Worker_List[j].predict(np.array([Validation_X[i],]),batch_size=None, verbose=0, steps=None)
    
    #print('Trial ',T,' Complete')    

#%% 
np.savetxt(ManagerOutputFile,C_Results,delimiter=",")
np.savetxt(NNOutputFile,MLP_Results,delimiter=",")
np.savetxt(MoEOutputFile,MoE_Results,delimiter=",")
np.savetxt(TrueOutputFile,Composite_Targets,delimiter=",")
for i in range(0,WRKR_NUM_WORKERS):
    np.savetxt((WorkerCompositeOutputFile + str(i) + ".txt"),Worker_Results[i],delimiter=",")
