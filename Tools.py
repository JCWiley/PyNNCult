# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:20:19 2018

@author: John
"""

import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
import theano
import keras
import numpy as np
        
def Normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

#def Scale(Array,OutMax,OutMin):
 #   InMax = np.amax(Array)
  #  InMin = np.amin(Array)
    
def Scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def Linearize(string):
    flat = list()
    for item in string:
        if isinstance(item,np.ndarray):
            flat.extend(Linearize(item))
        else:
            flat.append(item)
    return flat

def MSE(Actual,Predicted):
    return np.mean((Actual - Predicted)**2)