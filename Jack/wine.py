# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:01:44 2018

@author: jackc
"""

import numpy as np

red_filedata = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1)
white_filedata = np.genfromtxt('winequality-white.csv', delimiter=';', skip_header=1)

# Transpose so that features are along the rows and data points are along the columns
red_filedata = red_filedata.transpose()
white_filedata = white_filedata.transpose()

# Extract the targets and separate into vector
red_targets = red_filedata[11:,:]
red_data = red_filedata[:11,:]
white_targets = white_filedata[11:,:]
white_data = white_filedata[:11,:]

#column vector of features


def pca(data, labels, kFeat = 3):
    inds = np.random.choice(np.arange(len(data[0])), len(data[0]))
    data[:] = data[:,inds]
    labels[0,:] = labels[0,inds]
    
    m = np.mean(data,1)    
    m = m.reshape(len(m),1)
#    print(m)
    if kFeat >= len(m):
        kFeat = len(m)
    data = data-m
    covX = np.cov(data)
    vals, vects = np.linalg.eig(covX)
    inds = vals.argsort()[::-1]
    labels = labels.T
    U = vects[:,inds[0]].reshape(len(data),1)
    if kFeat > 1:
        for i in range(kFeat-1):
            temp = vects[:,inds[i]].reshape(len(m),1)
            U = np.concatenate((U, temp), axis = 1)
    
    W = np.matmul(data.T, U)    
    W = np.concatenate((W,np.ones((W.shape[0],1))),axis = 1)
    gamma = np.linalg.pinv(np.matmul(W.T,W))
    gamma = np.matmul(gamma, W.T)
    gamma = np.matmul(gamma, labels)
    
    return U.T, gamma
    



U, gamma = pca(white_data, white_targets, 5)
predicted = np.matmul(U,white_data[:,0])
for i in range(100):
    sample = np.matmul(U, white_data[:,i])
    sample = np.append(sample,[1])
    predicted = np.matmul(sample,gamma)
    print('Sample ',i+1,': predicted = ', predicted, '    actual = ', white_targets[0,i])



















