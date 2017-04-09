# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 20:41:59 2017

@author: Chin-Wei

"""


import cPickle as pickle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import numpy as np
from lasagne.nonlinearities import linear, tanh, softmax
from lasagne.layers import get_output, get_all_params
from lasagne.objectives import categorical_crossentropy as cc
import lasagne
tnc = lasagne.updates.total_norm_constraint
nonl = tanh
import theano
import theano.tensor as T
floatX = theano.config.floatX

from modules import lc_softmax2


d0 = 50
d1 = 500
d2 = 784
lr = 0.001


def get_encoder():
    
    print '\tgetting encoder'
    enc = lasagne.layers.InputLayer(shape=(None,d2)) 
    
    enc = lasagne.layers.DenseLayer(enc,d1,nonlinearity=nonl)
    print enc.output_shape
    enc = lasagne.layers.DenseLayer(enc,d0,nonlinearity=linear)
    print enc.output_shape
    
    return enc

def get_decoder():
    
    print '\tgetting decoder'
    dec = lasagne.layers.InputLayer(shape=(None,d0)) 
    
    dec = lasagne.layers.DenseLayer(dec,d1,nonlinearity=nonl)
    print dec.output_shape
    dec = lasagne.layers.DenseLayer(dec,d2*256,nonlinearity=linear)
    print dec.output_shape

    
    return dec
    


def load_mnist(path):
    
    tr,va,te = pickle.load(open(path,'r'))
    tr_x,tr_y = tr
    va_x,va_y = va
    te_x,te_y = te
    
    enc = OneHotEncoder(10)
    
    tr_y = enc.fit_transform(tr_y).toarray().reshape(50000,10).astype(int)
    va_y = enc.fit_transform(va_y).toarray().reshape(10000,10).astype(int)    
    te_y = enc.fit_transform(te_y).toarray().reshape(10000,10).astype(int)
    
    train_x = tr_x
    train_y = tr_y 
    valid_x = va_x
    valid_y = va_y
    test_x = te_x
    test_y = te_y
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y
    
    

class VAE(object):
    
    def __init__(self,softmax=softmax):
        
        self.inpv = T.matrix('inpv')
        self.outv = T.imatrix('outv') # indices
        self.ep = T.matrix('ep')
        self.w = T.scalar('w')
        
        self.n = self.inpv.shape[0]
        
        self.enc_m = get_encoder() 
        self.enc_s = get_encoder() 
        self.dec = get_decoder()
        
        self.mu = get_output(self.enc_m,self.inpv)    
        self.log_s = get_output(self.enc_s,self.inpv)   
        self.log_v = 2*self.log_s
        self.sigma = T.exp(self.log_s)
        self.var = T.exp(self.log_s*2)
        self.z = self.mu + self.sigma * self.ep
        self.rec_linear = get_output(self.dec,self.z)
        self.rec_reshaped_ln = self.rec_linear.reshape((self.n*d2,256))
        self.rec_reshaped = softmax(self.rec_reshaped_ln)
        
        self.out_onehot = T.extra_ops.to_one_hot(
            self.outv.reshape((self.n*d2,)),256
        )
        
        # lazy modeling just using squared error ...
        self.rec_losses_reshaped = cc(self.rec_reshaped,self.out_onehot)
        self.rec_losses = self.rec_losses_reshaped.reshape((self.n,d2)).sum(1)
        self.klss = - 0.5 * (1+self.log_v) + \
                      0.5 * (self.mu**2 + self.var)
        self.kls = self.klss.sum(1)
        self.rec_loss = self.rec_losses.mean()
        self.kl = self.kls.mean()
        self.loss = self.rec_loss + self.kl*self.w

        self.params = get_all_params(self.enc_m) + \
                      get_all_params(self.enc_s) + \
                      get_all_params(self.dec)
        self.updates = lasagne.updates.adam(self.loss,self.params,lr)
        
        print '\tgetting train func'
        self.train_func = theano.function([self.inpv,self.outv,self.ep,self.w],
                                           [self.loss.mean(),
                                            self.rec_loss.mean(),
                                            self.kl.mean()],
                                           updates=self.updates)
        
        print '\tgetting other useful funcs'
        self.recon = theano.function(
            [self.inpv,self.ep],
             self.rec_reshaped.argmax(1).reshape((self.n,d2))
        )
        self.recon_ = theano.function(
            [self.inpv,self.ep],
             self.rec_reshaped.reshape((self.n,d2,256))
        )
        self.project = theano.function([self.inpv,self.ep],self.z)
        self.get_mu = theano.function([self.inpv],self.mu)
        self.get_var = theano.function([self.inpv],self.var)
        self.get_klss = theano.function([self.inpv],self.klss)

