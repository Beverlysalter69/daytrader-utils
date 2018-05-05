
import numpy as np
import os
import dtdata as dt

# fix random seed for reproducibility
np.random.seed(90210)

subset = 2

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)
(data, labels) = dt.normaliseAroundEntry(data)
print(data.shape)

print(data)
print(labels)