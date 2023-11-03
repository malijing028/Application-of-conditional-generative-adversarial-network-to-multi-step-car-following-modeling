# *-* coding:utf-8 *-*

import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import os
from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.regularizers import l2


# Load data
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)

# seperated for training and validating
n_train = round(len(X_train)*0.7)
trainX , trainy = X_train[:n_train,:], y_train[:n_train,:]
valX, valy = X_train[n_train:,:], y_train[n_train:,:]
# reshape data for seq2seq model
trainy = trainy.reshape(trainy.shape[0],trainy.shape[1],1)
valy = valy.reshape(valy.shape[0],valy.shape[1],1)


# # the dataset knows the number of steps and features
n_steps_in = trainX.shape[1]
n_steps_out = trainy.shape[1]
lookback, delay = n_steps_in, n_steps_out
n_features_in = trainX.shape[2]
n_features_out = trainy.shape[2]
print(n_steps_in, n_steps_out, n_features_in, n_features_out)


# define model
model = Sequential()
model.add(LSTM(32, activation='tanh', input_shape=(n_steps_in, n_features_in)))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(32, activation='tanh', return_sequences=True))
model.add(TimeDistributed(Dense(n_features_out)))

model.compile(loss='mean_squared_error', optimizer='adam')

# patient early stopping
callbacks_list = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5),
                  ModelCheckpoint(filepath='seq2seq_model.h5', monitor='val_loss', save_best_only=True)]

# fit model
history = model.fit(trainX, trainy, batch_size=128, validation_data=(valX, valy), epochs=200,
                    verbose=1, callbacks=callbacks_list)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("seq2seq_model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved model to disk")
