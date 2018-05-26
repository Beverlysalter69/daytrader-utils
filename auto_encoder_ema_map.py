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
from keras.models import load_model
from keras import regularizers
from keras.models import Model
import matplotlib.pyplot as plt

# fix random seed for reproducibility
random_seed = 90210
np.random.seed(random_seed)

original_seq_len = 2400
batch_size = 256
epochs = 3000
hold_out = 350
# this is the size of our encoded representations
encoding_dim = 240 

savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

###############################################################################################
# PAST - load auto encoder.. for PAST data.
###############################################################################################
autoencoder_past_path = "/home/suroot/Documents/train/daytrader/models/autoencoder-past-"+str(encoding_dim)+".hdf5"
autoencoder_past = load_model(autoencoder_past_path)
input_past = Input(shape=(original_seq_len,))
encoder_past_layer = autoencoder_past.layers[-2]
encoder_past = Model(input_past, encoder_past_layer(input_past))
encoded_past_input = Input(shape=(encoding_dim,))
decoder_past_layer = autoencoder_past.layers[-1]
decoder_past = Model(encoded_past_input, decoder_past_layer(encoded_past_input))

###############################################################################################
# FUTURE load auto encoder.. for FUTURE data.
###############################################################################################
autoencoder_future_path = "/home/suroot/Documents/train/daytrader/models/autoencoder-future-"+str(encoding_dim)+".hdf5"
autoencoder_future = load_model(autoencoder_future_path)
input_future = Input(shape=(original_seq_len,))
encoder_future_layer = autoencoder_future.layers[-2]
encoder_future = Model(input_future, encoder_future_layer(input_future))
encoded_future_input = Input(shape=(encoding_dim,))
decoder_future_layer = autoencoder_future.layers[-1]
decoder_future = Model(encoded_future_input, decoder_future_layer(encoded_future_input))

scaler = StandardScaler() 

data = dt.loadData(path)
for i in range(data.shape[0]):
    data[i,] = (data[i,]/data[i,-20]) - 1.0
data = scaler.fit_transform(data) 
print(data.shape)

# get and encode PAST data..
x_train_past = data[hold_out:,0:2400]
print("past: " + str(x_train_past.shape))
x_train_past_encoded = encoder_past.predict(x_train_past)
print("past encoded: " + str(x_train_past_encoded.shape))

# get and encode FUTURE data..
x_train_future = data[hold_out:,20:2420]
print("future: " + str(x_train_future.shape))
x_train_future_encoded = encoder_future.predict(x_train_future)
print("future encoded: " + str(x_train_future_encoded.shape))


#################################################################################################
## MAP ENCODER
#################################################################################################
modelPath_mapper= savePath+"/models/autoencoder-mapper-"+str(encoding_dim)+".hdf5"
if( not os.path.isfile( modelPath_mapper ) ):
    input_mapper = Input(shape=(encoding_dim,))
    encoded_mapper = Dense(2048, activation='relu')(input_mapper)
    decoded_mapper = Dense(encoding_dim, activation='linear')(encoded_mapper)
    autoencoder_mapper = Model(input_mapper, decoded_mapper)
    autoencoder_mapper.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

    checkpoint_mapper = ModelCheckpoint(modelPath_mapper, monitor='val_acc', verbose=2, save_best_only=True, mode='max')

    history_future = autoencoder_mapper.fit(x_train_past_encoded, x_train_future_encoded,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=[checkpoint_mapper],
                        validation_split=0.1
                        )                                        
else:
    print("loading model from cache.")   
    autoencoder_mapper = load_model(modelPath_mapper)             


# TEST holdout PAST
x_test_past = data[0:hold_out:,0:2400]
x_test_past_encoded = encoder_past.predict(x_test_past)

# TEST holdout FUTURE
x_test_future = data[0:hold_out:,20:2420]
x_test_future_encoded = encoder_future.predict(x_test_future)


# we predict all past - which should map to our future encodings..
x_test_future_predicted = autoencoder_mapper.predict(x_test_past_encoded)

# now we can map this back to future vals with the future decoder
y_test = decoder_future.predict(x_test_future_predicted)

# lets view some results...

for i in range(len(y_test)):
    y = y_test[i]
    print(y.shape)
    print("----------------------------------")
    l1, = plt.plot(range(2420), data[i,:], label = 'Truth')
    l2, = plt.plot(range(20, 2420), y, 'r', label = 'Pred')
    plt.legend(handles = [l1, l2], loc = 'lower left')
    plt.show()