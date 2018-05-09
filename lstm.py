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
from keras.optimizers import RMSprop, Adam
from functools import reduce

# fix random seed for reproducibility
np.random.seed(90210)

subset = 1005
num_classes = 5
input_size = 128
epochs = 250

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
(data, labels) = dt.cacheLoadData(path, num_classes, input_size)

data = np.reshape(data, [data.shape[0], data.shape[1],1] )
print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(x_train.shape))

batch_size = 32
timesteps = data.shape[1]


# modified from here: https://github.com/keras-team/keras/blob/master/examples/imdb_bidirectional_lstm.py
model = Sequential()
#model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(timesteps,return_sequences=True, stateful=True), batch_input_shape=(batch_size, timesteps, 1)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])    