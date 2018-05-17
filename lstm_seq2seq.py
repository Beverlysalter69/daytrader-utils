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
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Flatten, TimeDistributed
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from functools import reduce


# fix random seed for reproducibility
np.random.seed(90210)

batch_size = 64
epochs = 100
latent_dim = 256  # Latent dimensionality of the encoding space.

num_classes = 5
input_size = 256


path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
savePath =r'/home/suroot/Documents/train/daytrader/'
(data, labels) = dt.cacheLoadData(path, num_classes, input_size)

data = np.reshape(data, [data.shape[0], data.shape[1], 1] )
labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1] )
print(data.shape)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(x_train.shape))


timesteps = data.shape[1]

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, 1))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]



# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, 1))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

data2 = data[:,-1,:]
data2 = np.reshape(data2, [data2.shape[0], data2.shape[1], 1] )

# Run training
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy') #loss='categorical_crossentropy')
model.fit([data, data2], labels,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


# checkpoint
modelPath= savePath+"lstm-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

print('Train...')
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])   

dt.plotHistory(history) 