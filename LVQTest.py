# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:58:28 2018

@author: John
"""
import numpy as np

exec(open("LVQ.py").read())
exec(open("Tools.py").read())

DataFileName = r"C:\Users\John\Documents\Visual Studio 2015\Projects\AI\OIT-ML-2014-NNC\Neural Net Cultivator\Data and Results\iris.txt"
DataFileDelimiter = " "

LVQ_LEARNING_RATE = .8
LVQ_INPUT_COUNT = 4
LVQ_PREDICTED_NUM_CATAGORIES = 6

NUM_INPUTS = 4
NUM_OUTPUTS = 3


dataset = np.loadtxt(DataFileName, delimiter=DataFileDelimiter)
np.random.shuffle(dataset)
#X = dataset[:,0:NUM_INPUTS]
X = np.random.uniform(0,1.0,[150,4])
Y = dataset[:,NUM_INPUTS:NUM_INPUTS+NUM_OUTPUTS]

LVQ = LearningVectorQuantizer(LVQ_LEARNING_RATE,LVQ_INPUT_COUNT,LVQ_PREDICTED_NUM_CATAGORIES)

cat_count1 = np.zeros(shape=LVQ_PREDICTED_NUM_CATAGORIES,dtype = int)
#cat_count2 = np.zeros(shape=LVQ_PREDICTED_NUM_CATAGORIES,dtype = int)

for i in range(150):
    #print(X[i])
    #print(Normalize(X)[i])
    cat_count1[LVQ.Execute(X[i])] = cat_count1[LVQ.Execute(X[i])] + 1
    #cat_count2[LVQ.Execute(SCX[i])] = cat_count[LVQ.Execute(SCX[i])] + 1
    
    
print(cat_count1)
#print(cat_count2)