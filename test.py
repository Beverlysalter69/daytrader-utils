
import numpy as np
import os
import dtdata as dt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(90210)

subset = -1

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)

(data, labels) = dt.centerAroundEntry(data)
print(data.shape)

a = dt.toClasses(labels, 5)

dt.printLabelDistribution(a)

#data = data[:,-90:-1]

#plt.hist(data, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()

pca = PCA(n_components=20, svd_solver='full')
pca.fit(data)        
print(pca.explained_variance_ratio_)  

'''
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(data)
distances, indices = nbrs.kneighbors(data)

print(data[:,-1])
print(labels)

right = 0
wrong = 0
for inds in indices:
    print(labels[inds])
'''
