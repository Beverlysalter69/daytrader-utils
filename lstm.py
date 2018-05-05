import numpy as np
import os
import dtdata as dt

import matplotlib.pyplot as plt
import math
import random
import pprint as pp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten
from functools import reduce

# fix random seed for reproducibility
np.random.seed(90210)

subset = 1005

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)

data = data[:,-99:-1]

(data, labels) = dt.normaliseAroundEntry(data)
data = np.reshape(data, [data.shape[0], data.shape[1],1] )
print(data.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(X_train.shape))

batch_size = 32
data_dim = 1
timesteps = data.shape[1]


# modified from here: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
model = Sequential()
#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(timesteps,return_sequences=True, stateful=True), batch_input_shape=(batch_size, timesteps, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))

# try using different optimizers and different optimizer configs
model.compile("adam", "mean_squared_error", metrics=["mean_absolute_percentage_error","mean_squared_error"])

print('Train...')
model.fit(X_train, y_train,
	    verbose=1,
        batch_size=batch_size,
        epochs=250,
        validation_data=[X_test, y_test])