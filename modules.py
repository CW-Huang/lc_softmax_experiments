# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:27:48 2017

@author: Chin-Wei, Shawn Tan

Locally Correlated Softmax

"""

from theano import tensor as T

softmax = T.nnet.softmax

def lc_softmax1(x):
    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    smoothed_e_x = 0.5 * e_x
    smoothed_e_x = T.inc_subtensor(
        smoothed_e_x[:,:-1],
        0.25 * e_x[:,1:]
    )
    smoothed_e_x = T.inc_subtensor(
        smoothed_e_x[:,1:],
        0.25 * e_x[:,:-1]
    )
    out = smoothed_e_x / smoothed_e_x.sum(axis=-1, keepdims=True)
    return out

def lc_softmax2(x):
    e_x = T.exp(x - x.max(axis=-1, keepdims=True))
    p = e_x.shape[-1]
    M = T.eye(p,p,0) * 0.5 + T.eye(p,p,-1) * 0.25 + T.eye(p,p,1) * 0.25 
    smoothed_e_x = T.dot(e_x,M)
    out = smoothed_e_x / smoothed_e_x.sum(axis=-1, keepdims=True)
    return out
    
    
    