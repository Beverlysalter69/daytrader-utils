import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)


# fix random seed for reproducibility
np.random.seed(90210)
crop_future = -20
class_to_view = 3   # class

num_classes = 5

input_size = 128

#savePath = r'/home/suroot/Documents/train/daytrader/'
#path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
savePath = r'/home/suroot/Documents/train/raw/'
path =r'/home/suroot/Documents/train/raw/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
(data, labels_classed, _) = dt.cacheLoadData(path, crop_future, num_classes, input_size, symbols=dt.CA_EXTRA)
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels_classed, test_size=0.1)

print("TEST SIZE: " + str(x_test.shape))

#/home/suroot/Documents/train/raw/mlp[3]-28-0.48.hdf5
from keras.models import load_model
model = load_model(savePath+'mlp[5]-154-0.32.hdf5')

model.summary()

class5Data = []

for i in range(len(x_test)):
    if( y_test[i, (class_to_view-1) ] == 1 ):
        class5Data.append(x_test[i,:])


class5 = np.array(class5Data)  
print("class: " + str(class_to_view) + "   " +  str(class5.shape) )
prediction = model.predict(class5)
print(prediction)
onehot_ind = np.argmax(prediction, axis=1)+1
print(onehot_ind)
plt.hist(onehot_ind, bins='auto')  # arguments are passed to np.histogram
plt.title("Class "+str(class_to_view)+" Prediction Distribution")
plt.show()
