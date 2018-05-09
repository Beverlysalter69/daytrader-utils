from functools import reduce
import numpy as np
import math
import matplotlib.pyplot as plt
from dateutil import parser
from sklearn import preprocessing
import glob
import os
import csv
import json
import math


def loadData(path, subset = -1):        
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    if(subset > 0):
        allFiles = allFiles[0:subset]
    data = []
    for file in allFiles:
        print(file)
        with open(file, 'r') as f:
            data.append( [float(x[1]) for x in list(csv.reader(f))] )   
    return np.array(data)

def centerAroundEntry(data):
    # extract the price at 20 min after entry
    labels = data[:,-1]
    # remove the last 20 min of history from our data..
    data = data[:,0:-20]
    # normalise data to the ENTRY point
    for i in range(data.shape[0]):
        labels[i] = (labels[i]/data[i,-1]) - 1.0
        data[i,] = (data[i,]/data[i,-1]) - 1.0
    return (data, labels)

def scale(data):
    return preprocessing.scale(data)


def toClasses(labels, num_classes):
    sorted = np.sort(np.array(labels, copy=True))
    bsize = math.floor( len(sorted) / num_classes )
    buckets = []
    for i in range(num_classes):        
        buckets.append(sorted[i*bsize])
    targets = np.digitize(labels, buckets) - 1
    one_hot_targets = np.eye(num_classes)[targets]
    print(one_hot_targets)
    return one_hot_targets
    
def printLabelDistribution(x):
    unq_rows, count = np.unique(x, axis=0, return_counts=1)
    out = {tuple(i):j for i,j in zip(unq_rows,count)}
    print(out)
    return out


def findStrangeRecords(path):        
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    num_strange = 0
    for file in allFiles:
        with open(file, 'r') as f:
            data = np.array( [float(x[1]) for x in list(csv.reader(f))] )   
            label =  labels = data[-1]
            data = data[0:-20]
            check = (label/data[-1]) - 1.0
            if( math.fabs(check) >= 0.05):
                print("strange values in file: " + file)
                num_strange += 1
    print("Found " + num_strange + " files with strange values.")
