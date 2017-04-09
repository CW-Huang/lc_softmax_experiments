# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:10:03 2017

@author: Chin-Wei
"""

from vanilla_vae import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10)
parser.add_argument('--bs', default=64)
parser.add_argument('--print_every', default=50)
parser.add_argument('--operator', default=0)
parser.add_argument('--path', default=r'/data/lisa/data/mnist/mnist.pkl.gz')
args = parser.parse_args()
print args

epochs = args.epochs
bs = args.bs
print_every = args.print_every
operator = args.operator
path = args.path

from modules import lc_softmax1, lc_softmax2
softmax_operator = {0:softmax,1:lc_softmax1}[operator]


# get data & build model
train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(path)
train_x_out = (train_x * 256).astype('int32')

model = VAE(lc_softmax2)


# training
t = 0
records = list()
for e in range(epochs):
    for i in range(50000/bs):
        
        w = 1.
        x = train_x[i*bs:(i+1)*bs]
        y = train_x_out[i*bs:(i+1)*bs]
        sample = np.random.normal(0,1,[x.shape[0],d0]).astype(floatX)
        loss, rec, kl = model.train_func(x,y,sample,w)
        records.append([loss,rec,kl])
        
        if t%20 == 0:
            print t,e,i, loss, rec, kl
        
        t+=1


