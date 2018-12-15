# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:02:47 2018

@author: jackc
"""

 
#from sklearn import svm 
#Classification

"""
Data: 
   (0:9) 10 Questions 
   (10) age [<20, <30, <40, <50, <60, <70]
   (11) gender => m = 1, f = 0
   (12) ethnicity
   (13) jaundice (bool)
   (14) autism type (bool)
   (15) country
   (16) used app (bool)
   (17) result type 0 - 10 range
   (18) '18 or more' thats it
   (19) whos completing = 'self' 'parent' 'health care professional' 'relative' 'others'
   (20) boolean = label
   
"""
import numpy as np
import arff as ARFF

def uNiQuE(vec):
    popCtr = 0
    for p in vec:
        if p in vec[:popCtr]:
            vec = np.delete(vec, popCtr, 0)
            popCtr -= 1
        popCtr += 1
        
    return vec

def preProcess(data):
    i = 0
    for q in range(len(data)):
        if None in data[i] or data[i,10] == 383:
            data = np.delete(data, i, 0)
            i -= 1        
        i += 1

        for j in range(len(data[i-1])):
            if type(data[i-1,j]) == str:
                data[i-1,j] = data[i-1,j].lower()
    
    labels = data[:,-1]
    data = np.delete(data, -1, 1)
    
    ethnicities = uNiQuE(data[:,12])
    countries = uNiQuE(data[:,15])
    completed = uNiQuE(data[:,-1])
    yesNo = np.array(['no', 'yes'])
    ageVec = np.arange(20,80, 10)
    for i in range(len(data)):
        data[i,:10] = data[i,:10].astype(int)
        
        tempInd = np.where(ageVec > int(data[i,10]))
        data[i,10] = int(tempInd[0][0])
    
        if data[i,11] == 'm':
            data[i,11] = 1
        else:
            data[i,11] = 0
        tempInd = np.where(ethnicities == data[i,12])
        data[i,12] = int(tempInd[0])
        
        tempInd = np.where(yesNo == data[i,13])
        data[i,13] = int(tempInd[0])
        tempInd = np.where(yesNo == data[i,14])
        data[i,14] = int(tempInd[0])
        
        tempInd = np.where(countries == data[i,15])
        data[i,15] = int(tempInd[0])
        
        tempInd = np.where(yesNo == data[i,16])
        data[i,16] = int(tempInd[0])
        
        data[i,17] = int(data[i,17])
        
        data[i,18] = 1
        
        tempInd =  np.where(completed==data[i,19])
        data[i,19] = int(tempInd[0])
        
        labels[i] = int(labels[i] == 'yes')
    
    return data, labels

def kNN(trainingData, trainingLabels, testData, k):
    distances = []
    for sample in trainingData:
        distances.append(np.linalg.norm(sample - testData))
    
    idx = np.argpartition(distances, k)
    labs = trainingLabels[idx]
    labs = labs[:k]
    unique, counts = np.unique(labs, return_counts=True)
    if len(counts) == 1:
        return unique[0]

    if counts[0] > counts[1]:
        return unique[0]
    elif counts[1] > counts[0]:
        return unique[1]
    return 1

def calcErr(trainingData, trainingLabels, testData, testLabels, k = 7):
    err = []
    for i in range(len(testData)):    
        err.append(int(kNN(trainingData, trainingLabels, testData[0],k) != testLabels[i]))
    
    return 1-sum(err)/len(err)

def kFold(data, labels, kFolds, typ = 'kNN', k = 7):
    #shuffle
    inds = np.random.choice(np.arange(len(data)), len(data))
    data[:] = data[inds]
    labels[:] = labels[inds]

    startInd = 0
    stepSize = int(len(data)/kFolds)
    max10 = max(data[:,10])
    max12 = max(data[:,12])
    max15 = max(data[:,15])
    max17 = max(data[:,17])
    max19 = max(data[:,19])
    Errs = []
    for i in range(kFolds):
        if i != kFolds-1:
            testData = data[startInd:startInd+stepSize]
            testLabels = labels[startInd:startInd+stepSize]
            trainingData = data[:startInd]
            trainingData = np.concatenate((trainingData, data[startInd+stepSize:]))
            trainingLabels = labels[:startInd]
            trainingLabels = np.concatenate((trainingLabels,labels[startInd+stepSize:]))
        else:
            testData = data[startInd:]
            testLabels = labels[startInd:]
            trainingData = data[:startInd]
            trainingLabels = labels[:startInd]
        startInd += stepSize      
        if typ == 'kNN':
            trainingData[:,10] = trainingData[:,10]/max10
            trainingData[:,12] = trainingData[:,12]/max12
            trainingData[:,15] = trainingData[:,15]/max15
            trainingData[:,17] = trainingData[:,17]/max17
            trainingData[:,19] = trainingData[:,19]/max19
            testData[:,10] = testData[:,10]/max10
            testData[:,12] = testData[:,12]/max12
            testData[:,15] = testData[:,15]/max15
            testData[:,17] = testData[:,17]/max17
            testData[:,19] = testData[:,19]/max19
            temp = calcErr(trainingData, trainingLabels, testData, testLabels, k)
        elif typ == 'naive':
            temp = calcErrNaive(trainingData, trainingLabels, testData, testLabels)
        
        Errs.append(temp)
    
    return Errs
        
def trainNaive(data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    prior = np.array([counts[0], counts[1]])
    prior = (prior+0.0)/len(data)
    conditional = np.zeros((2, len(data[0]), 60))
    for i in range(len(data)):
        for j in range(len(data[i])):
            conditional[labels[i], j, data[i,j]] += 1
            
    for i in range(len(conditional)):
        for j in range(len(conditional[0])):
            sumCondition = sum(conditional[0,j]) + sum(conditional[1,j])
            for k in range(len(conditional[0,j])):
                conditional[i,j,k] = conditional[i,j,k]/sumCondition
    
    return prior, conditional, unique

def testNaive(prior, conditional, unique, sample):
    prob = np.array([prior[0],prior[1]])
    for i in range(len(sample)):
        prob[0] = prob[0] * conditional[0,i,sample[i]]
        prob[1] = prob[1] * conditional[1,i,sample[i]]
    
    if prob[0] >= prob[1]:
        return unique[0]
        
    return unique[1]
        
def calcErrNaive(trainingData, trainingLabels, testData, testLabels):
    errs = 0
    prior, conditional, unique = trainNaive(trainingData, trainingLabels)
    for i in range(len(testData)):
        prediction = testNaive(prior, conditional, unique, testData[i])
        errs += int(prediction != testLabels[i])
    
    return np.round(1-errs/len(testLabels), 6)
    
dataset = ARFF.load(open('Autism-Adult-Data.arff'))
DATA = np.array(dataset['data'])    
DATA, LABELS = preProcess(DATA)
inds = np.random.choice(np.arange(len(DATA)), len(DATA))
DATA[:] = DATA[inds]
LABELS[:] = LABELS[inds]
for i in np.arange(3,13,2):
    kErrs = kFold(DATA, LABELS, i, typ = 'naive')
    print('k = ', i, ' with average accuracy = ' , np.average(kErrs).round(6))
    print('Accuracy for each fold: ', kErrs)










