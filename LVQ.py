# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 16:11:55 2018

@author: John
"""
import numpy as np

    #an LVQ is identical to a MLP except that it lacks the hidden layer, it still has fully connected input and output layers
class LearningVectorQuantizer:    
    # Learning rate        rate at which weights are adjusted, mosly inconsequntial, must be between 0 and 1, preferably closer to 1
    # Input count          equal to the size of the vectors you are classifying, calculated from incoming vector size
    # Hidden Layer Nodes   inconsiquential, ignore
    # Output Count         best guess as to the number of classes the inputs will fall into, overestimating this number is fine, underestimating will provide incorrect results
    # Inc momentum         momentum value for the network, inconsequencial in this case
    def __init__(self,learningRate):
        self.m_learningRate = learningRate
    #returns a catagory for the input rather then an array of values, so int rather than double[]
    def Execute(self,inputArray,outputCount):
        #self.m_inputArray = Scale(inputArray)
        self.m_inputCount = len(inputArray)
        self.m_inputArray = inputArray
        self.m_outputCount = outputCount
        
        self.weights = np.random.uniform(low=-1.0,high=1.0,size=[self.m_inputCount,self.m_outputCount])
        
        self.ForwardPropagate()
        self.BackPropagate()
        
        temp = self.m_MaxValueLocation
        
        self.m_MaxValueLocation = 0
        
        return temp
    
    def BackPropagate(self):
        for i in range(0,self.m_inputCount):
            self.weights[i,self.m_MaxValueLocation] = self.weights[i,self.m_MaxValueLocation]+(self.m_learningRate * (self.m_inputArray[i] - self.weights[i,self.m_MaxValueLocation]))

    def ForwardPropagate(self):
        CurrentSum = 0
        CurrentSum = np.zeros(self.m_outputCount,dtype = float)
        for i in range(0,self.m_outputCount):
            for j in range(0,self.m_inputCount):
                CurrentSum[i] = CurrentSum[i] + self.m_inputArray[j] * self.weights[j][i]
        self.m_MaxValueLocation = np.argmax(CurrentSum)