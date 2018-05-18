import numpy as np
import os
import math
import dtdata as dt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.models import load_model
import matplotlib.pyplot as plt
from build_model_basic import * 


# in this model we are trying to take the input and just map it to the output label.
# bacause we are not going back to a seq we don't need to use our autoencoder.  We can just work
# with the PCA output and the class label

## Parameters
learning_rate = 0.01
lambda_l2_reg = 0.003  

input_size = 128
num_classes = 5
crop_future = -20
## Network Parameters
# length of input signals
input_seq_len = 128 
# length of output signals
output_seq_len = 1
# size of LSTM Cell
hidden_dim = 128 
# num of input signals
input_dim = 1
# num of output signals
output_dim = 1
# num of stacked lstm layers 
num_stacked_layers = 2 
# gradient clipping - to avoid gradient exploding
GRADIENT_CLIPPING = 2.5

scaler = StandardScaler() 
savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data


savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
(data, labels_classed, _) = dt.cacheLoadData(path, crop_future, num_classes, input_size)
print("data: "+str(data.shape))
ss = StratifiedShuffleSplit(test_size=0.1)
for train_index, test_index in ss.split(data, labels_classed):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels_classed[train_index], labels_classed[test_index]

print(x_train.shape)

def generate_train_samples(x, y, batch, batch_size = 10, input_seq_len = input_seq_len, output_seq_len = output_seq_len):        
    input_seq = x[batch*batch_size:(batch*batch_size)+batch_size, 0:input_seq_len]
    output_seq = y[batch*batch_size:(batch*batch_size)+batch_size, :]
    output_classes = [ [np.where(r==1)[0][0]] for r in output_seq ]
    return np.array(input_seq), np.array(output_classes)


epochs = 50
batch_size = 64
total_iteractions = int(math.floor(x_train.shape[0] / batch_size))
KEEP_RATE = 0.5
train_losses = []
val_losses = []


if( not os.path.isfile( os.path.join(savePath, 'univariate_ts_model1.meta') ) ):
    print("building model..")
    rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=False)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):        
            print("EPOCH: " + str(epoch))
            for i in range(total_iteractions):        
                batch_input, batch_output = generate_train_samples(x = x_train, y=y_train, batch=i, batch_size=batch_size)
                feed_dict = {rnn_model['enc_inp'][t]: batch_input[:,t].reshape(-1,input_dim) for t in range(input_seq_len)}
                feed_dict.update({rnn_model['target_seq'][t]: batch_output[:,t].reshape(-1,output_dim) for t in range(output_seq_len)})
                _, loss_t = sess.run([rnn_model['train_op'], rnn_model['loss']], feed_dict)
                print(loss_t)
                
            temp_saver = rnn_model['saver']()
            save_path = temp_saver.save(sess, os.path.join(savePath, 'univariate_ts_model1'))
            
    print("Checkpoint saved at: ", save_path)
else:
    print("using cached model...")


rnn_model = build_graph(input_seq_len = input_seq_len, output_seq_len = output_seq_len, hidden_dim=hidden_dim, feed_previous=True)

predictions = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    saver = rnn_model['saver']().restore(sess, os.path.join(savePath, 'univariate_ts_model0'))
    
    for i in range(len(x_test)):
        test_seq_input = x_test[i,0:input_seq_len]
        feed_dict = {rnn_model['enc_inp'][t]: test_seq_input[t].reshape(1,1) for t in range(input_seq_len)}
        feed_dict.update({rnn_model['target_seq'][t]: np.zeros([1, output_dim]) for t in range(output_seq_len)})
        final_preds = sess.run(rnn_model['reshaped_outputs'], feed_dict)
        
        final_preds = np.concatenate(final_preds, axis = 1)
        print(final_preds)

        predictions.append(final_preds.reshape(-1))

print(predictions)