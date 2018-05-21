import numpy as np
import os
import math
import dtdata as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import load_model
import matplotlib.pyplot as plt
from build_model_basic import * 

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003  

encoding_dim = 2420

original_seq_len = 2420

## Network Parameters
# length of input signals
input_seq_len = 2400 
# length of output signals
output_seq_len = 20
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

scaler = StandardScaler() 
autoencoder_path = "/home/suroot/Documents/train/daytrader/models/autoencoder-"+str(encoding_dim)+".hdf5"
cache = "/home/suroot/Documents/train/daytrader/autoencoded-"+str(encoding_dim)+".npy"
savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
model_name = "raw_ts_model0"

use_cache = True

if( not use_cache or not os.path.isfile(cache) ):
    data = dt.loadData(path)
    (data, labels) = dt.centerAroundEntry(data, 0)
    # scale data .. don't forget to stor the scaler weights as we will need them after.
    data = scaler.fit_transform(data) 
    print(data.shape) 
    # cache this data and the scaler weights.
    np.save(cache, data)
    # TODO: cache the scaler weights

print("loading cached data")
data = np.load(cache)
print(data.shape)

def generate_train_samples(x, batch, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):        
    input_seq = x[batch*batch_size:(batch*batch_size)+batch_size, 0:input_seq_len]
    output_seq = x[batch*batch_size:(batch*batch_size)+batch_size, input_seq_len:input_seq_len+output_seq_len]
    return np.array(input_seq), np.array(output_seq)

# TRAINING PROCESS
x_train, x_test, _, _ = train_test_split(data, data[:,-1], test_size=0.1)

epochs = 50
batch_size = 32
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
        print(final_preds)

        predicted_ts = np.append(x_test[i,0:input_seq_len], final_preds.reshape(-1))
        predicted_ts = np.reshape(predicted_ts, (1, encoding_dim))
        predictions.append(predicted_ts)

for i in range(len(x_test)):
    predicted_ts = predictions[i]       
    print(predicted_ts.shape)
    #predicted_decoded_ts = scaler.inverse_transform(decoded_ts)
    #decoded_ts = scaler.inverse_transform(decoded_ts)
    l1, = plt.plot(range(2400), x_test[i,0:2400], label = 'Training truth')
    l2, = plt.plot(range(2400, 2420), x_test[i,2400:], 'y', label = 'Test truth')
    l3, = plt.plot(range(2400, 2420), predicted_ts[0,2400:], 'r', label = 'Test predictions')
    plt.legend(handles = [l1, l2, l3], loc = 'lower left')
    plt.show()

