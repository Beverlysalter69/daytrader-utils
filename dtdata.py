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

# TODO: make sure the above have equal size classes    
def printLabelDistribution(x):
    unq_rows, count = np.unique(x, axis=0, return_counts=1)
    out = {tuple(i):j for i,j in zip(unq_rows,count)}
    print(out)
