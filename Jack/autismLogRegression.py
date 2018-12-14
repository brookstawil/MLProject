# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:02:47 2018

@author: jackc
"""

import numpy as np
import arff

#Classification

"""
Data: 
   (0:9) 10 Questions 
   (10) age
   (11) gender
   (12) ethnicity
   (13:14) boolean 1 and 2
   (15) country
   (16) boolean
   (17) 0 - 10 range
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
    
    
    return data


dataset = arff.load(open('Autism-Adult-Data.arff'))
data = np.array(dataset['data'])

data = preProcess(data)
labels = data[:,-1]
data = np.delete(data, -1, 1)

#print(data[:,12])
print(uNiQuE(data[:,19]))

print('ages', min(data[:,10]), max(data[:,10]))
ethnicities = uNiQuE(data[:,12])
countries = uNiQuE(data[:,15])

print(len(data[0]))
#print(ethnicities)
print(len(data))

inds = np.random.choice(np.arange(len(data)), len(data))
data[:] = data[inds]
labels[:] = labels[inds]

splitInd = int(len(data)*.9)
trainingData = data[:splitInd].T
trainingLabels = labels[:splitInd]
testData = data[splitInd:].T
testLabels = labels[splitInd:]



























