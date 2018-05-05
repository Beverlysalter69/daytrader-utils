
import numpy as np
import os
import dtdata as dt
from sklearn.neighbors import NearestNeighbors

# fix random seed for reproducibility
np.random.seed(90210)

subset = -1

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)

(data, labels) = dt.normaliseAroundEntry(data)
print(data.shape)

data = data[:,-90:-1]
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(data)
distances, indices = nbrs.kneighbors(data)

print(data[:,-1])
print(labels)

right = 0
wrong = 0
for inds in indices:
    print(labels[inds])
