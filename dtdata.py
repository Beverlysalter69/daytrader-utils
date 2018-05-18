from functools import reduce
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from dateutil import parser
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import glob
import os
import csv
import json
import math
import sys


def loadData(path, subset = -1):        
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    if(subset > 0):
        allFiles = allFiles[0:subset]
    data = []
    # NOTE: set np.random.seed for reproducability
    shuffleFiles = shuffle(sorted(allFiles))
    for file in shuffleFiles:
        print(file)
        with open(file, 'r') as f:
            data.append( [float(x[1]) for x in list(csv.reader(f))] )   
    return np.array(data)

def centerAroundEntry(data, crop_future):
    # extract the price at 20 min after entry
    labels = np.copy(data[:,-1])
    # remove the last 20 min of history from our data..
    if crop_future < 0:
        data = data[:,0:crop_future]
    # normalise data to the ENTRY point
    for i in range(data.shape[0]):
        labels[i] = (labels[i]/data[i,-1]) - 1.0
        data[i,] = (data[i,]/data[i,-1]) - 1.0
    return (data, labels)


def filterOutliers(data, labels, pos, neg):
    filteredData = []
    filteredLabels = []
    for i in range(data.shape[0]):
        if(labels[i] > neg and labels[i] < pos):
            filteredData.append(data[i,])
            filteredLabels.append(labels[i])        
    return (np.array(filteredData), np.array(filteredLabels) )


def toClasses(labels, num_classes):
    sorted = np.sort(np.array(labels, copy=True))
    bsize = math.floor( len(sorted) / num_classes )
    buckets = []
    for i in range(num_classes):        
        buckets.append(sorted[i*bsize])
    print("buckets: " + str(buckets))
    targets = np.digitize(labels, buckets) - 1
    one_hot_targets = np.eye(num_classes)[targets]
    print(one_hot_targets)
    return one_hot_targets
    
def printLabelDistribution(x):
    unq_rows, count = np.unique(x, axis=0, return_counts=1)
    out = {tuple(i):j for i,j in zip(unq_rows,count)}
    print(out)
    return out


def plotTrainingExample(te):
    plt.plot(range(len(te)),te)
    plt.show()

def cacheLoadData(path, crop_future, num_classes, input_size, scaler = StandardScaler() ):
    cache = "/tmp/daytrader_"+str(input_size)+"-"+str(crop_future)+".npy"
    labelsCache = "/tmp/daytrader_labels_"+str(input_size)+".npy"
    if( not os.path.isfile(cache) ):
        data = loadData(path)

        (data, labels) = centerAroundEntry(data, crop_future)
        print(data.shape)
        data_scaled = scaler.fit_transform(data)        

        labels_classed = toClasses(labels, num_classes)

        printLabelDistribution(labels_classed)

        if(input_size > 0):
            pca = PCA(n_components=input_size, svd_solver='full')
            data_reduced = pca.fit_transform(data_scaled) 
            np.save(cache, data_reduced)
            np.save(labelsCache, labels_classed)
        else:
            np.save(cache, data_scaled)
            np.save(labelsCache, labels_classed)
    data = np.load(cache)
    labels_classed = np.load(labelsCache)
    return (data, labels_classed, scaler)


def plotHistory(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()