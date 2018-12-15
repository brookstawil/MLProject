# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:01:44 2018

@author: jackc
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
def calcAcc(trainingData, trainingLabels, testData, testLabels, kFeat = 3):
    U, gamma = pca(trainingData, trainingLabels, kFeat)
    predictedList = []
    err = 0
    for i in range(testData.shape[1]):
        sample = np.matmul(U, testData[:,i])
        sample = np.append(sample,[1])
        predicted = np.matmul(sample,gamma)
        predictedList.append(predicted)
        err += abs(predicted - testLabels[0,i])
    err /= testData.shape[1]    
    
    return err, predictedList
    

def kFold(data, labels, typeWine, kFeat = 5, kFolds = 5):
    #shuffle
    inds = np.random.choice(np.arange(data.shape[1]), data.shape[1])
    data[:] = data[:,inds]
    labels[:] = labels[:,inds]
    typeWine = typeWine[inds]

    startInd = 0
    stepSize = int(len(data)/kFolds)
    Errs = []
    predictions = []
    for i in range(kFolds):
        if i != kFolds-1:
            testData = data[:, startInd:startInd+stepSize]
            testLabels = labels[:, startInd:startInd+stepSize]
            trainingData = data[:, :startInd]
            trainingData = np.concatenate((trainingData, data[:, startInd+stepSize:]), axis = 1)
            trainingLabels = labels[:, :startInd]
            trainingLabels = np.concatenate((trainingLabels,labels[:, startInd+stepSize:]), axis = 1)
        else:
            testData = data[:, startInd:]
            testLabels = labels[:, startInd:]
            trainingData = data[:, :startInd]
            trainingLabels = labels[:, :startInd]

        startInd += stepSize
        
        temp, pList = calcAcc(trainingData, trainingLabels, testData, testLabels, kFeat = kFeat)
        predictions.extend(pList)
        Errs.append(temp)
    
    return Errs, predictions, labels, typeWine

U, gamma = pca(white_data, white_targets, 5)
predicted = np.matmul(U,white_data[:,0])


data = np.concatenate((red_data, white_data),axis = 1)
labels = np.concatenate((red_targets, white_targets),axis = 1)
typeWine = ['r' for i in range(red_data.shape[1])]
typeWine.extend(['w' for i in range(white_data.shape[1])])
typeWine = np.array(typeWine)

kFeats = 11
kF = 3
err, predictions, actual, typeWine = kFold(data, labels, typeWine, kFeat = kFeats, kFolds = kF)
avgErr = sum(err)/len(err)

fig, ax = plt.subplots(1,1,figsize = (10,10))
for i in range(len(predictions)):
    if typeWine[i] == 'r':
        ax.plot( actual[0,i],predictions[i], 'ro')
    elif typeWine[i] == 'w':
        ax.plot( actual[0,i],predictions[i], 'b^')
ax.set_title('Red Wine + White Wine Regression'+ ' with number of features = '+ str(kFeats) + ' and number of folds = ' + str(kF) + ' with Avg error = ' + str(avgErr))
ax.set_ylabel('Predictions')
ax.set_xlabel('Actual')
ax.set_xlim([0,11])
if typeWine[0] == 'r':
    ax.legend(['Red Wine', 'White Wine'])
else:
    ax.legend(['White Wine','Red Wine'])
    
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
lower = min([ymin,xmin])
upper = max([ymax, xmax])
print(lower, upper)
ax.plot( [lower, upper] , [lower, upper], 'k-')






data = white_data
labels = white_targets

typeWine = ['w' for i in range(white_data.shape[1])]
typeWine = np.array(typeWine)

kFeats = 11
kF = 3
err, predictions, actual, typeWine = kFold(data, labels, typeWine, kFeat = kFeats, kFolds = kF)
avgErr = sum(err)/len(err)

fig, ax = plt.subplots(1,1,figsize = (10,10))
for i in range(len(predictions)):
    if typeWine[i] == 'r':
        ax.plot( actual[0,i],predictions[i], 'ro')
    elif typeWine[i] == 'w':
        ax.plot( actual[0,i],predictions[i], 'b^')
ax.set_title('White Wine Regression'+ ' with number of features = '+ str(kFeats) + ' and number of folds = ' + str(kF) + ' with Avg error = ' + str(avgErr))
ax.set_ylabel('Predictions')
ax.set_xlabel('Actual')
ax.set_xlim([0,11])
ax.legend(['White Wine'])
    
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
lower = min([ymin,xmin])
upper = max([ymax, xmax])
print(lower, upper)
ax.plot( [lower, upper] , [lower, upper], 'k-')







data = red_data
labels = red_targets

typeWine = ['r' for i in range(red_data.shape[1])]
typeWine = np.array(typeWine)

kFeats = 11
kF = 3
err, predictions, actual, typeWine = kFold(data, labels, typeWine, kFeat = kFeats, kFolds = kF)
avgErr = sum(err)/len(err)

fig, ax = plt.subplots(1,1,figsize = (10,10))
for i in range(len(predictions)):
    if typeWine[i] == 'r':
        ax.plot( actual[0,i],predictions[i], 'ro')
    elif typeWine[i] == 'w':
        ax.plot( actual[0,i],predictions[i], 'b^')
ax.set_title('Red Wine Regression'+ ' with number of features = '+ str(kFeats) + ' and number of folds = ' + str(kF) + ' with Avg error = ' + str(avgErr))
ax.set_ylabel('Predictions')
ax.set_xlabel('Actual')
ax.set_xlim([0,11])
ax.legend(['Red Wine'])
    
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
lower = min([ymin,xmin])
upper = max([ymax, xmax])
print(lower, upper)
ax.plot( [lower, upper] , [lower, upper], 'k-')

