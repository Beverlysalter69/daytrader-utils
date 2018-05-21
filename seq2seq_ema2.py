import numpy as np
import os
import math
import dtdata as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
from build_model_basic import * 

## Parameters
random_seed = 90210
num_classes = 5
np.random.seed(random_seed)
latent_dim = 256  # Latent dimensionality of the encoding space.

original_seq_len = 2400
encoding_dim = 120
hold_out = 350
batch_size = 128
epochs = 50

savePath = r'/home/suroot/Documents/train/daytrader/'


# load auto encoder.. for PAST data.
autoencoder_past_path = "/home/suroot/Documents/train/daytrader/models/autoencoder-past-"+str(encoding_dim)+".hdf5"
autoencoder_past = load_model(autoencoder_past_path)
input_past = Input(shape=(original_seq_len,))
encoder_past_layer = autoencoder_past.layers[-2]
encoder_past = Model(input_past, encoder_past_layer(input_past))
encoded_past_input = Input(shape=(encoding_dim,))
decoder_past_layer = autoencoder_past.layers[-1]
decoder_past = Model(encoded_past_input, decoder_past_layer(encoded_past_input))

# load auto encoder.. for PAST data.
autoencoder_future_path = "/home/suroot/Documents/train/daytrader/models/autoencoder-future-"+str(encoding_dim)+".hdf5"
autoencoder_future = load_model(autoencoder_future_path)
input_future = Input(shape=(original_seq_len,))
encoder_future_layer = autoencoder_future.layers[-2]
encoder_future = Model(input_future, encoder_future_layer(input_future))
encoded_future_input = Input(shape=(encoding_dim,))
decoder_future_layer = autoencoder_future.layers[-1]
decoder_future = Model(encoded_future_input, decoder_future_layer(encoded_future_input))


# Load our data...
scaler = StandardScaler() 
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
data = dt.loadData(path)
for i in range(data.shape[0]):
    data[i,] = (data[i,]/data[i,-20]) - 1.0
data = scaler.fit_transform(data) 
print(data.shape)

# get and encode PAST data..
x_train_past = data[0:-hold_out,0:2400]
print("past: " + str(x_train_past.shape))
x_train_past_encoded = encoder_past.predict(x_train_past)
print("past encoded: " + str(x_train_past_encoded.shape))

# get and encode FUTURE data..
x_train_future = data[0:-hold_out,20:2420]
print("future: " + str(x_train_future.shape))
x_train_future_encoded = encoder_future.predict(x_train_future)
print("future encoded: " + str(x_train_future_encoded.shape))


## Network Parameters
# length of input signals
input_seq_len = 120 
# length of output signals
output_seq_len = 120
# size of LSTM Cell
hidden_dim = 256 
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers 
num_stacked_layers = 2 
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5
model_name = "seq2seq-ema"



def generate_train_samples(x, y, batch, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):        
    input_seq =  x[batch*batch_size:(batch*batch_size)+batch_size, :]
    output_seq = y[batch*batch_size:(batch*batch_size)+batch_size, :]
    return np.array(input_seq), np.array(output_seq)

epochs = 100
batch_size = 128
total_iteractions = int(math.floor(x_train_past_encoded.shape[0] / batch_size))
KEEP_RATE = 0.5
train_losses = []
val_losses = []


if( not os.path.isfile( os.path.join(savePath, model_name+'.meta') ) ):
    print("building model..")
    rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=False)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):        
            print("EPOCH: " + str(epoch))
            for i in range(total_iteractions):        
                batch_input, batch_output = generate_train_samples(x = x_train_past_encoded, y=x_train_future_encoded, batch=i, batch_size=batch_size)
                feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
                feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
                _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
                print(loss_t)
                
            temp_saver = rnn_model['saver']()
            save_path = temp_saver.save(sess, os.path.join(savePath, model_name))
            
    print("Checkpoint saved at: ", save_path)
else:
    print("using cached model...")

x_test_past = data[-hold_out:, 0:2400]
x_test_past_encoded = encoder_past.predict(x_test_past)

x_test_future = data[-hold_out:, 20:2420]
x_test_future_encoded = encoder_future.predict(x_test_future)

l1, = plt.plot(range(120), x_test_past_encoded[22], label = 'past')
l2, = plt.plot(range(1,121), x_test_future_encoded[22], "r", label = 'future')
plt.legend(handles = [l1,l2], loc = 'lower left')
plt.show()


rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=False)


predictions = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = rnn_model['saver']().restore(sess, os.path.join(savePath, model_name))
    
    for i in range(len(x_test_past)):
        test_seq_input = x_test_past[i]
        feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
        final_preds = np.concatenate(final_preds, axis = 1)
        print(str(final_preds))

        predicted_ts = final_preds.reshape(-1)
        predictions.append( predicted_ts )

        
        #print(str(final_preds) + " <=> " + str(x_test[i,input_seq_len:]) )
        #print("***")        

        #predicted_ts = np.copy( x_test[i,0:input_seq_len] )
        #predicted_ts = np.append( predicted_ts, final_preds.reshape(-1))
        #predicted_ts = np.reshape(predicted_ts, (1, encoding_dim))

        #print(x_test[i,(input_seq_len-2):])
        #print(predicted_ts[0,(input_seq_len-2):])
        #predictions.append(predicted_ts)


for final_preds in predictions:
    print(final_preds.shape)
    predicted = decoder_future.predict( np.reshape(final_preds, (1, 120) ) )
    print(predicted.shape)
    l1, = plt.plot(range(2420), predicted, label = 'Predicted')
    plt.legend(handles = [l1], loc = 'lower left')
    plt.show()

'''

use_cache = False

centered_data = []

if( not use_cache or not os.path.isfile(cache) ):
    data = dt.loadData(path)
    for i in range(data.shape[0]):
        data[i,] = (data[i,]/data[i,-20]) - 1.0
    centered_data = np.copy(data)    
    # scale data .. don't forget to stor the scaler weights as we will need them after.
    data = scaler.fit_transform(data) 
    print(data.shape)        
    # encode all of the data .. should now be of length 480    
    encoded_ts = encoder.predict(data)
    print(encoded_ts.shape)    
    # cache this data and the scaler weights.
    np.save(cache, encoded_ts)
    # TODO: cache the scaler weights

print("loading cached data")

data = np.load(cache)
print(data.shape)

def generate_train_samples(x, batch, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):        
    input_seq = x[batch*batch_size:(batch*batch_size)+batch_size, 0:input_seq_len]
    output_seq = x[batch*batch_size:(batch*batch_size)+batch_size, input_seq_len:input_seq_len+output_seq_len]
    return np.array(input_seq), np.array(output_seq)

# TRAINING PROCESS
x_train = data[0:-hold_out, :]
x_test = data[-hold_out:, :]
x_train_center = centered_data[0:-hold_out, :]
x_test_center = centered_data[-hold_out:, :]

print(x_test_center.shape)

print("x_test: " + str(x_test.shape))
decoded_ts = decoder.predict(np.copy(x_test) )
print("decoded_ts: "+ str(decoded_ts.shape))

epochs = 50
batch_size = 128
total_iteractions = int(math.floor(x_train.shape[0] / batch_size))
KEEP_RATE = 0.5
train_losses = []
val_losses = []


if( not os.path.isfile( os.path.join(savePath, model_name+'.meta') ) ):
    print("building model..")
    rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=False)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):        
            print("EPOCH: " + str(epoch))
            for i in range(total_iteractions):        
                batch_input, batch_output = generate_train_samples(x = x_train, batch=i, batch_size=batch_size)
                feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
                feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
                _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
                print(loss_t)
                
            temp_saver = rnn_model['saver']()
            save_path = temp_saver.save(sess, os.path.join(savePath, model_name))
            
    print("Checkpoint saved at: ", save_path)
else:
    print("using cached model...")


rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=True)

predictions = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = rnn_model['saver']().restore(sess, os.path.join(savePath, model_name))
    
    for i in range(len(x_test)):
        test_seq_input = x_test[i,0:input_seq_len]
        feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
        final_preds = np.concatenate(final_preds, axis = 1)
        print(str(final_preds) + " <=> " + str(x_test[i,input_seq_len:]) )
        print("***")        

        predicted_ts = np.copy( x_test[i,0:input_seq_len] )
        predicted_ts = np.append( predicted_ts, final_preds.reshape(-1))
        predicted_ts = np.reshape(predicted_ts, (1, encoding_dim))

        print(x_test[i,(input_seq_len-2):])
        print(predicted_ts[0,(input_seq_len-2):])
        predictions.append(predicted_ts)


decoded_ts = scaler.inverse_transform(decoded_ts)


buckets = dt.labelBuckets(x_test_center[:,-1], num_classes)
print("BUCKETS: " + str(buckets))
targets = np.digitize(x_test_center[:,-1], buckets) - 1
one_hot_targets = np.eye(num_classes)[targets]
print(one_hot_targets)

decoded_predictions = decoder.predict(  np.reshape(np.array(predictions), (hold_out, encoding_dim) ) )
decoded_predictions = scaler.inverse_transform(decoded_predictions)
predicted_targets = np.digitize(decoded_predictions[:,-1], buckets) - 1
predicted_one_hot_targets =  np.eye(num_classes)[predicted_targets]
print(predicted_one_hot_targets)

right = 0
for i in range(predicted_one_hot_targets.shape[0]):
    a1 = one_hot_targets[i]
    a2 = predicted_one_hot_targets[i]
    if np.array_equal(a1, a2):
        right += 1

print("ACC: " + str(right/hold_out) )


for i in range(len(x_test_center)):
    predicted_ts = predictions[i]
    print(predicted_ts.shape)
    predicted_decoded_ts = decoder.predict( predicted_ts )
    print(predicted_decoded_ts.shape)
    predicted_decoded_ts = scaler.inverse_transform(predicted_decoded_ts)
    print("----------------------------------")
    #print(x_test_center[i,2398:])
    #print(predicted_decoded_ts[0,2398:])
    #print( x_test_center[i,:].shape)
    print("entry: " + str(x_test_center[i,2400]) )
    l1, = plt.plot(range(2420), x_test_center[i,:], label = 'Truth')
    l3, = plt.plot(range(2420), decoded_ts[i,:], 'y', label = 'Decoded')
    l2, = plt.plot(range(2400,2420), predicted_decoded_ts[0,2400:], 'r', label = 'Pred')
    #l2, = plt.plot(range(2418,2420), decoded_ts[i,2418:], 'yo', label = 'Test truth')
    #l3, = plt.plot(range(2418,2420), predicted_decoded_ts[0,2418:], 'ro', label = 'Test predictions')
    #plt.legend(handles = [l1, l3, l2], loc = 'lower left')
    plt.legend(handles = [l1, l2], loc = 'lower left')
    plt.show()

'''