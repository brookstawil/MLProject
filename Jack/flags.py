# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 13:46:13 2018

@author: jackc
"""

import numpy as np

filedata = np.genfromtxt('flag.data', dtype=None, delimiter=',', encoding='utf-8')

data = [[None for _ in range(len(filedata[0]))] for _ in range(len(filedata))]

# Data is stored as mostly integers, but these correspond to a string in the data description, stored in these lists 
landmass = [None, 'N.America', 'S.America', 'Europe', 'Africa', 'Asia', 'Oceania']
quadrant = [None, 'NE', 'SE', 'SW', 'NW']
languages = [None, 'English', 'Spanish', 'French', 'German', 'Slavic', 'Other Indo-European', 'Chinese', 'Arabic', 'Japanese/Turkish/Finnish/Magyar', 'Others']
religions = ['Catholic', 'Other Christian', 'Muslim', 'Buddhist', 'Hindu', 'Ethnic', 'Marxist', 'Others']
for i in range(len(data)):
    for j in range(len(data[i])):
        # Country Name
        if (j == 0):
            data[i][j] = str(filedata[i][j])
        # Landmass
        elif (j == 1):
            data[i][j] = landmass[filedata[i][j]]
        elif (j  == 2):
            data[i][j] = quadrant[filedata[i][j]]
        elif (j  == 5):
            data[i][j] = languages[filedata[i][j]]
        elif (j == 6):
            data[i][j] = religions[filedata[i][j]]
        else:
            data[i][j] = filedata[i][j]
        # Make the row into a numpy array
        data[i] = np.array(data[i])

# Transpose so that features are along the rows and data points are along the columns
data = np.array(data).transpose()

# Extract the religions as the labels, row 6
names = data[0]
labels = data[6] 
data = np.delete(data, 6, axis=0)

# Extract the non flag related data
for i in range(6):
    data = np.delete(data, 0, axis=0)
    
    
    
#### Post code

def uNiQuE(vec):
    popCtr = 0
    for p in vec:
        if p in vec[:popCtr]:
            vec = np.delete(vec, popCtr, 0)
            popCtr -= 1
        popCtr += 1
        
    return vec   

def kFold(data, labels, kFolds):
    #shuffle
    inds = np.random.choice(np.arange(len(data)), len(data))
    data[:] = data[inds]
    labels[:] = labels[inds]

    startInd = 0
    stepSize = int(len(data)/kFolds)
    Acc = []
    predictions = []
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
        temp, pList = calcErrNaive(trainingData, trainingLabels, testData, testLabels)
        Acc.append(temp)
        predictions.extend(pList)
    
    return Acc, labels, predictions
        

def trainNaive(data, labels):
    unique, counts = np.unique(labels, return_counts=True)
    prior = counts
    prior = (prior+0.0)/len(data)
    
    conditional = np.zeros((8, len(data[0]), 50))
    #conditional = (labels, feature, values in feature)
    for i in range(len(data)):
        for j in range(len(data[i])):
            conditional[labels[i], j, data[i,j]] += 1
    for i in range(len(conditional)):
        for j in range(len(conditional[0])):
            sumCondition = 0
            for k in range(0,conditional.shape[0]):
                sumCondition += sum(conditional[k,j,:])
            for k in range(0,len(conditional[0,j])):
                conditional[i,j,k] = conditional[i,j,k]/sumCondition
    

    return prior, conditional, unique


def testNaive(prior, conditional, unique, sample):
    prob = prior
    for i in range(len(sample)):
        for j in range(len(prob)):
            prob[j] = prob[j] * conditional[j,i,sample[i]]
    maxVal = np.argmax(prob)
        
    return unique[maxVal]
        
def calcErrNaive(trainingData, trainingLabels, testData, testLabels):
    errs = 0
    prior, conditional, unique = trainNaive(trainingData, trainingLabels)
    pList = []
    for i in range(len(testData)):
        prediction = testNaive(prior, conditional, unique, testData[i])
        errs += int(prediction == testLabels[i])
        pList.append(prediction)
        
    return np.round(errs/len(testLabels), 6), pList


def preProcess(data):
    i = 0
    for q in range(len(data)):
        if None in data[i]:
            data = np.delete(data, i, 0)
            i -= 1        
        i += 1
        for j in range(len(data[i-1])):
            if type(data[i-1,j]) == str:
                data[i-1,j] = data[i-1,j].lower()

    domColor = uNiQuE(data[:,10])
    topLeftColor = uNiQuE(data[:,-2])
    botRightColor = uNiQuE(data[:,-1])
    numStars = np.array([6, 5, 4, 3, 2, 1, 0])
    for i in range(len(data)):
        tempInd = np.where(domColor == data[i,10])
        data[i,10] = int(tempInd[0])
        tempInd = np.where(topLeftColor == data[i,-2])
        data[i,-2] = int(tempInd[0])
        tempInd = np.where(botRightColor == data[i,-1])
        data[i,-1] = int(tempInd[0])
        
        #make stars be in range [0 to >5]
        tempInd = np.where(numStars <= data[i,15])
        data[i,15] = int(tempInd[0][0])
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i,j] = int(data[i,j])
    return data

tempData = preProcess(data.T)
religions = uNiQuE(labels)
tempLabels = labels[:]
for i in range(len(labels)):
    tempInd = np.where(religions == labels[i])
    tempLabels[i] = int(tempInd[0])
    
for i in np.arange(3,13,2):
    acc, actual, predictions = kFold(tempData, tempLabels, 5)
    print('k = ', i, ' with average accuracy = ' , np.average(acc).round(6))
    print('Accuracy for each fold: ', acc)




acc, actual, predictions = kFold(tempData, tempLabels, 11)
print('Actual Religions and Predicted Religion for first 20 samples: ')
for i in range(20):
    if actual[i] == 2:
        print('Actual: ', religions[actual[i]], '\t Predicted:', religions[predictions[i]])
    elif actual[i] == 6:
        print('Actual: ', religions[actual[i]], '\t \t \t Predicted:', religions[predictions[i]])
    else:
        print('Actual: ', religions[actual[i]], '\t \t Predicted:', religions[predictions[i]])

















