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

batch_size = 64
epochs = 250
hold_out = 350

savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

scaler = StandardScaler() 

data = dt.loadData(path)
for i in range(data.shape[0]):
    data[i,] = (data[i,]/data[i,-20]) - 1.0
data = scaler.fit_transform(data) 
print(data.shape)

# TRAINING PROCESS (Hold out training on TEST set)
#x_train, x_test, _, _ = train_test_split(data, np.zeros( (data.shape[0], 1) ), test_size=0.1, random_state=random_seed)
x_train = data[hold_out:, :]
print("training on: " + str(x_train.shape))

# this is the size of our encoded representations
encoding_dim = 121 

# this is our input placeholder
input = Input(shape=(x_train.shape[1],))
encoded = Dense(encoding_dim, activation='relu')(input)
decoded = Dense(x_train.shape[1], activation='linear')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

encoder = Model(input, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# checkpoint
modelPath= savePath+"autoencoder-"+str(encoding_dim)+".hdf5"
checkpoint = ModelCheckpoint(modelPath, monitor='acc', verbose=2, save_best_only=True, mode='max')

history = autoencoder.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint],
                    )

x_test = data[0:hold_out, :]
encoded_ts = encoder.predict(x_test)
decoded_ts = decoder.predict(encoded_ts)

for i in range(x_test.shape[0]):
    l1, = plt.plot(range(len(x_test[i,:])), x_test[i,:], 'g', label = 'truth')
    l2, = plt.plot(range(len(decoded_ts[i,:])), decoded_ts[i,:], 'r', label = 'encoded')
    plt.legend(handles = [l1, l2], loc = 'lower left')
    plt.show()
