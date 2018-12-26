# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:37:33 2018

@author: John
"""
#import tensorflow

#from tensorflow.examples.tutorials.mnist import input_data
#input_data.read_data_sets('C:/Users/John/Stuff/NN Cultivator/MNIST Data/Extracted Data')
import random

from mnist import MNIST
mndata = MNIST('C:\\Users\\John\\Stuff\\NN Cultivator\\MNIST Data')
Tr_Images,Tr_Labels = mndata.load_training()
Te_Images,Te_Lables = mndata.load_testing()

index = random.randrange(0,len(Tr_Images))
#print(mndata.display(Tr_Images[index]))
print(Tr_Images[index].length)