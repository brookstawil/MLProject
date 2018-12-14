# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:02:47 2018

@author: jackc
"""

import numpy as np
import arff as ARFF
import matplotlib.pyplot as plt
from scipy.io import arff 

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



#data, meta = ARFF.loadarff('Autism-Adult-Data.arff')
dataset = ARFF.load(open('Autism-Adult-Data.arff'))
data = np.array(dataset['data'])    


data,labels = preProcess(data)


inds = np.random.choice(np.arange(len(data)), len(data))
data[:] = data[inds]
labels[:] = labels[inds]

splitInd = int(len(data)*.9)
trainingData = data[:splitInd].T
trainingLabels = labels[:splitInd]
testData = data[splitInd:].T
testLabels = labels[splitInd:]























