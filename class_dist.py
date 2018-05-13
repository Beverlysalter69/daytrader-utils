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

class_to_view = 5   # class

num_classes = 5
batch_size = 256
epochs = 500

input_size = 512

savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
(data, labels_classed) = dt.cacheLoadData(path, num_classes, input_size)
x_train, x_test, y_train, y_test = train_test_split(data, labels_classed, test_size=0.1)


from keras.models import load_model
model = load_model(savePath+'models/mlp-19-0.29.hdf5')

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

class5Data = []

for i in range(len(x_test)):
    if( y_test[i, (class_to_view-1) ] == 1 ):
        class5Data.append(x_test[i,:])


class5 = np.array(class5Data)  
print(class5.shape)
prediction = model.predict(class5)
onehot_ind = np.argmax(prediction, axis=1)+1
print(onehot_ind)
plt.hist(onehot_ind, bins='auto')  # arguments are passed to np.histogram
plt.title("Class "+str(class_to_view)+" Prediction Distribution")
plt.show()
