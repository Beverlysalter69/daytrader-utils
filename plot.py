
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

data_unsorted = dt.loadData(path, subset)

data = np.array(sorted(data_unsorted, key=lambda x: (x[-20]/x[-1]) - 1.0 ) )

print(data.shape)

labels = data[:,-1]
entries = data[:,-20]



print( (labels[0]/entries[0]) - 1.0 )
print( (labels[len(labels)-1]/entries[len(entries)-1]) - 1.0 )

#dt.plotTrainingExample(data[0,:])
dt.plotTrainingExample(data[-1,:])
dt.plotTrainingExample(data[-2,:])
dt.plotTrainingExample(data[-3,:])
dt.plotTrainingExample(data[-4,:])
dt.plotTrainingExample(data[-5,:])
dt.plotTrainingExample(data[-6,:])

'''
(data, labels) = dt.centerAroundEntry(data, -20)
print(data.shape)

print(np.sort(labels))
print("min: " + str(labels.min()) )
print("max: " + str(labels.max()) )

sns.distplot(labels)  
plt.show()

a = dt.toClasses(labels, 5)

dt.printLabelDistribution(a)
'''


#data = data[:,-90:-1]

#plt.hist(data, bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()

#pca = PCA(n_components=20, svd_solver='full')
#pca.fit(data)        
#print(pca.explained_variance_ratio_)  


