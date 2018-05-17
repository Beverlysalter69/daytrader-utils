
import numpy as np
import os
import dtdata as dt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

plt.rcParams['interactive'] == True

# fix random seed for reproducibility
np.random.seed(90210)

subset = -1

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)

(data, labels) = dt.centerAroundEntry(data, -20)
print(data.shape)

print(np.sort(labels))
print("min: " + str(labels.min()) )
print("max: " + str(labels.max()) )

sns.distplot(labels)  
plt.show()

(data2, labels2) = dt.filterOutliers(data, labels, 0.018, -0.016)

sns.distplot(labels2)  
plt.show()

a = dt.toClasses(labels, 5)

dt.printLabelDistribution(a)



#data = data[:,-90:-1]

#plt.hist(data, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()

#pca = PCA(n_components=20, svd_solver='full')
#pca.fit(data)        
#print(pca.explained_variance_ratio_)  

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
