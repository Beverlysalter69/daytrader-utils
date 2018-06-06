import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.regularizers import l1
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import Model
import matplotlib.pyplot as plt

# fix random seed for reproducibility
random_seed = 90210
np.random.seed(random_seed)

batch_size = 256
epochs = 5000
hold_out = 1250
# this is the size of our encoded representations
encoding_dim = 480 

#savePath = r'/home/suroot/Documents/train/daytrader/'
#path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

savePath = r'/home/suroot/Documents/train/raw/'
path =r'/home/suroot/Documents/train/raw/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data


scaler = StandardScaler() 

data = dt.loadData(path, symbols=dt.CA_EXTRA)
for i in range(data.shape[0]):
    data[i,] = (data[i,]/data[i,-20]) - 1.0
data = scaler.fit_transform(data) 
print(data.shape)

#################################################################################################
## TRAIN PAST ENCODER
#################################################################################################
x_train_past = data[hold_out:,0:2400]
print("training past on: " + str(x_train_past.shape))

input_past = Input(shape=(x_train_past.shape[1],))
encoded_past = Dense(encoding_dim, activation='relu')(input_past)
decoded_past = Dense(x_train_past.shape[1], activation='linear')(encoded_past)
autoencoder_past = Model(input_past, decoded_past)
autoencoder_past.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

modelPath_past= savePath+"/models/autoencoder-past-"+str(encoding_dim)+".hdf5"
checkpoint_past = ModelCheckpoint(modelPath_past, monitor='acc', verbose=2, save_best_only=True, mode='max')

history_past = autoencoder_past.fit(x_train_past, x_train_past,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint_past],
                    )

#################################################################################################
## TRAIN FUTURE ENCODER
#################################################################################################
x_train_future = data[hold_out:,20:2420]
print("training on: " + str(x_train_future.shape))

input_future = Input(shape=(x_train_future.shape[1],))
encoded_future = Dense(encoding_dim, activation='relu')(input_future)
decoded_future = Dense(x_train_future.shape[1], activation='linear')(encoded_future)
autoencoder_future = Model(input_future, decoded_future)
autoencoder_future.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

modelPath_future= savePath+"/models/autoencoder-future-"+str(encoding_dim)+".hdf5"
checkpoint_future = ModelCheckpoint(modelPath_future, monitor='acc', verbose=2, save_best_only=True, mode='max')

history_future = autoencoder_future.fit(x_train_future, x_train_future,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint_future],
                    )
                    