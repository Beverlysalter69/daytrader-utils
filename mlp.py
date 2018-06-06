import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import train_test_split
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
crop_future = -20

input_size = 128

#savePath = r'/home/suroot/Documents/train/daytrader/'
#path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data
savePath = r'/home/suroot/Documents/train/raw/'
path =r'/home/suroot/Documents/train/raw/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
(data, labels_classed, _) = dt.cacheLoadData(path, crop_future, num_classes, input_size, symbols=dt.CA_EXTRA)
print(data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels_classed, test_size=0.1)

model = Sequential()
model.add(Dense(128, activation='relu', input_dim=data.shape[1], kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=['accuracy'])

# checkpoint
modelPath= savePath+"mlp["+str(num_classes)+"]-{epoch:02d}-{val_acc:.2f}.hdf5"
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