import numpy as np
import os
import math
import dtdata as dt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import load_model
import matplotlib.pyplot as plt
from build_model_basic import * 

random_seed = 90210
np.random.seed(random_seed)

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003  


holdout_size = 350

## Network Parameters
# length of input signals
input_seq_len = 120 
# length of output signals
output_seq_len = 1
# size of LSTM Cell
hidden_dim = 256 
# num of input signals
input_dim = 1
crop_future = 0
encoding_dim = input_seq_len + output_seq_len
# num of output signals
output_dim = 1
# num of stacked lstm layers 
num_stacked_layers = 2 
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

scaler = StandardScaler() 
cache = "/home/suroot/Documents/train/daytrader/seq2seq_raw-"+str(encoding_dim)+".npy"
#savePath = r'/home/suroot/Documents/train/daytrader/'
#path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
savePath = r'/home/suroot/Documents/train/raw/'
path =r'/home/suroot/Documents/train/raw/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
model_name = "pca2_ts_model"

use_cache = False

if( not use_cache or not os.path.isfile(cache) ):
    data = dt.loadData(path, symbols=dt.CA_EXTRA)
    #for i in range(data.shape[0]):
    #    data[i,] = (data[i,]/data[i,-20]) - 1.0
    # scale data .. don't forget to stor the scaler weights as we will need them after.
    data_scaled = scaler.fit_transform(data) 
    pca = PCA(n_components=encoding_dim, svd_solver='full')
    data_reduced = pca.fit_transform(data_scaled) 
    print(data.shape) 
    # cache this data and the scaler weights.
    np.save(cache, data_reduced)
    # TODO: cache the scaler weights

print("loading cached data")
full_data = np.load(cache)
holdout = full_data[0:holdout_size,:]
data = full_data[holdout_size:,:]
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
    l1, = plt.plot(range(2), x_test[i,119:121], label = 'Training truth')
    l2, = plt.plot(range(2), predicted_ts[0,119:121], 'r', label = 'Test predictions')
    plt.legend(handles = [l1, l2], loc = 'lower left')
    plt.show()

