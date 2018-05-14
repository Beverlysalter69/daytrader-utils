import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras.models import Model

# fix random seed for reproducibility
np.random.seed(90210)

num_classes = 5
batch_size = 256
epochs = 2500

input_size = -1

subset = -1 # -1 to use the entire data set

savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
(data, labels_classed) = dt.cacheLoadData(path, num_classes, input_size)

x_train = data
y_train = labels_classed

# visualize some data from training
#dt.plotTrainingExample(data[50,:])
#dt.plotTrainingExample(data[150,:])
#dt.plotTrainingExample(data[4500,:])


# this is the size of our encoded representations
encoding_dim = 128 

# this is our input placeholder
input = Input(shape=(data.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(2048, activation='relu')(input)
encoded = Dense(1024, activation='relu')(input)
encoded = Dense(512, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(1024, activation='relu')(decoded)
decoded = Dense(data.shape[1], activation='linear')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

encoder = Model(input, encoded)
# create a placeholder for an encoded (256-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# checkpoint
modelPath= savePath+"autoencoder.hdf5"
checkpoint = ModelCheckpoint(modelPath, monitor='acc', verbose=2, save_best_only=True, mode='max')

history = autoencoder.fit(x_train, x_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[checkpoint],
                    #validation_data=(x_test, y_test)
                    )


# encode and decode some digits
# note that we take them from the *test* set
encoded_ts = encoder.predict(x_train)
decoded_ts = decoder.predict(encoded_ts)

np.save(savePath + "encoder-"+str(encoding_dim)+".npy", encoded_ts)

for i in range(data.shape[0]):
    dt.plotTrainingExample(x_train[i,:])
    dt.plotTrainingExample(decoded_ts[i,:])
