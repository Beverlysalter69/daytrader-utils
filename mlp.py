import numpy as np
import os
import dtdata as dt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop


# fix random seed for reproducibility
np.random.seed(90210)

num_classes = 5
batch_size = 128
epochs = 200

subset = -1 # -1 to use the entire data set

path =r'/home/suroot/Documents/train/daytrader/ema-crossover' # path to data

data = dt.loadData(path, subset)

(data, labels) = dt.centerAroundEntry(data)
print(data.shape)
data_scaled = dt.scale(data)
labels_classed = dt.toClasses(labels, num_classes)

x_train, x_test, y_train, y_test = train_test_split(data_scaled, labels_classed, test_size=0.2)

model = Sequential()
model.add(Dense(1024, activation='relu', input_dim=2400))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])              