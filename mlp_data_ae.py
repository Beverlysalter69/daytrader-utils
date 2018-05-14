import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint
from keras import regularizers

# fix random seed for reproducibility
np.random.seed(90210)

num_classes = 5
batch_size = 256
epochs = 7500

input_size = 256

subset = -1 # -1 to use the entire data set

savePath = r'/home/suroot/Documents/train/daytrader/'
path =r'/home/suroot/Documents/train/daytrader/encoder-'+str(input_size)+'.npy' # path to data


(data, labels_classed) = dt.cacheLoadData(path, num_classes, -1) # just to load labels
data = np.load(path)

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
for train_index, test_index in ss.split(data, labels_classed):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = data[train_index], data[test_index]
    y_train, y_test = labels_classed[train_index], labels_classed[test_index]


dt.plotTrainingExample(x_train[15,:])

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=data.shape[1], kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

# checkpoint
modelPath= savePath+"mlp-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(modelPath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

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