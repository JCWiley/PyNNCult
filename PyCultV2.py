# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:02:51 2018

@author: John
"""

#imports
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

#%% Import the settings file for the current test
exec(open("Iris_Settings.py").read())
#exec(open("MilSong_Settings.py").read())
#exec(open("MNIST_Settings.py").read())

#%% Import all the tool files
exec(open("LVQ.py").read())
exec(open("Tools.py").read())
exec(open("Cultivator.py").read())
exec(open("MLP.py").read())
exec(open("SY_Keras_MoE.py").read())
#-------------------------------
#set system variables
DEBUG = 0

#%% Define Result storage variables
C_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
MLP_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
MoE_Results = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
Test_Targets = np.empty([VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
Test_Inputs = np.empty([VALIDATION*NUMTRIALS,NUM_INPUTS],dtype = object)
Worker_Results = np.empty([WRKR_NUM_WORKERS,VALIDATION*NUMTRIALS,NUM_OUTPUTS],dtype = object)
T = 0

#%% Build K-Fold
label_encoder = LabelEncoder()

Inputs = dataset[:,0:NUM_INPUTS]
Outputs = dataset[:,NUM_INPUTS:]
Output_Index = [np.where(index==1)[0][0] for index in Output]

kfold = StratifiedKFold(n_splits = NUMTRIALS,shuffle=True)

print("Data Management Complete")
#%% start trials loop
for Train_Indices,Vali_Indices in kfold.split(Inputs,Output_Index):
    np.random.shuffle(Train_Indices)
    np.random.shuffle(Vali_Indices)
    Vali_Length = len(Vali_Indices)
    
    #%% Cultivator Block
    Cult = Cultivator(NUM_INPUTS,NUM_OUTPUTS,WRKR_NUM_WORKERS,WRKR_LOSS,WRKR_MET,
                      WRKR_EPOCHS,WRKR_BATCH_SIZE,WRKR_ACTIVATION,WRKR_LEARNING_RATE,
                      WRKR_MOMENTUM,WRKR_LYR_1,WRKR_LYR_2,WRKR_LYR_3,WRKR_OPT,
                      LVQ_LEARNING_RATE,LVQ_PREDICTED_NUM_CATAGORIES,
                      MSTR_LOSS,MSTR_MET,MSTR_EPOCHS,
                      MSTR_BATCH_SIZE,MSTR_ACTIVATION,MSTR_LEARNING_RATE,
                      MSTR_MOMENTUM,MSTR_LYR_1,MSTR_LYR_2,MSTR_LYR_3,MSTR_OPT)
    
    Cult.Train(Inputs[Train_Indices],Outputs[Train_Indices])
    print("Cultivator Training Complete")
    #%% Perceptron Block
    Perceptron = MLP(NUM_INPUTS,NUM_OUTPUTS,NN_LOSS,NN_MET,NN_EPOCHS,
                     NN_BATCH_SIZE,NN_ACTIVATION,NN_LEARNING_RATE,NN_MOMENTUM,
                     NN_LYR_1,NN_LYR_2,NN_LYR_3,NN_OPT)
    Perceptron.Train(Inputs[Train_Indices],Outputs[Train_Indices])
    print("Perceptron Training Complete")
     
    #%% MoE Block
    MixtureOfExperts = MoE(NUM_INPUTS,NUM_OUTPUTS,MOE_NUM_EXPERTS,MOE_EXPRT_ACTIVATION,
                           MOE_GATE_ACTIVATION,MOE_BATCH_SIZE,MOE_LOSS,MOE_MET,MOE_OPT,
                           MOE_DEBUG,MOE_EPOCHS)
    MixtureOfExperts.Train(Inputs[Train_Indices],Outputs[Train_Indices])
#%%
    print("Training for trial {0} is complete".format(T))
    #%% NN Exeute Loop
    for i in range(0,Vali_Length):
        C_Results[(T*VALIDATION)+i] = Cult.Execute(np.array([Inputs[Vali_Indices[i]],]))
        MLP_Results[(T*Vali_Length)+i] = Perceptron.Execute(np.array([Inputs[Vali_Indices[i]],]))
        MoE_Results[(T*VALIDATION)+i] = MixtureOfExperts.Execute(np.array([Inputs[Vali_Indices[i]],]))
        Test_Targets[(T*VALIDATION)+i] = Outputs[Vali_Indices[i]]
        Test_Inputs[(T*VALIDATION)+i] = Inputs[Vali_Indices[i]]

    
    print("Execution for trial {0} is complete".format(T)) 
    T += 1

#%% 
np.savetxt(ManagerOutputFile,C_Results,delimiter=",")
np.savetxt(NNOutputFile,MLP_Results,delimiter=",")
np.savetxt(MoEOutputFile,MoE_Results,delimiter=",")
np.savetxt(TrueOutputFile,Test_Targets,delimiter=",")
np.savetxt(InputOutputFile,Test_Inputs,delimiter=",")