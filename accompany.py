import os

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop


def prepare_data():
    '''Load data, compute features, and construct training set.'''

    # Load the data. 
    # Concatenate all the vectorized 
    base_dir = '/Users/snikolov/Downloads/groove-monkee-midi-gm'
    arrays = []
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.split('.')[-1] == 'npy':
                array = np.load(os.path.join(root, filename))
                arrays.append(array)
    seq = np.concatenate(arrays, axis=0)

    # Select pitches and construct windows of 32 32nd notes.
    pitches = np.array([36,38,42,51])
    sel_seq = seq[:, pitches]
    windows = []
    breaks = range(len(sel_seq))[::32]
    for i in xrange(len(breaks)-1):
        windows.append(sel_seq[breaks[i]:breaks[i+1]][np.newaxis, :])
    windows = np.concatenate(windows[:-1])

    # Predict closed hihat (2) from kick (0)
    X = np.zeros((windows.shape[0], windows.shape[1] * 2))
    Y = np.zeros((windows.shape[0], windows.shape[1] * 2))
    for i,w in enumerate(windows):
        # Construct amplitude and phase spectrum features.
        f0 = np.fft.fft(1*(w[:,0]>64))
        af0 = np.abs(f0)
        pf0 = np.angle(f0)
        X[i,:] = np.hstack([af0, pf0])

        f2 = np.fft.fft(1*(w[:,2]>64))
        af2 = np.abs(f2)
        pf2 = np.angle(f2)
        Y[i,:] = np.hstack([af2, pf2])

    return windows, X, Y


def train_model(X, Y):
    '''Train a model mapping a kick pattern to an accompanying hihat
    pattern.'''

    model = Sequential()
    model.add(Dense(32, input_shape=(64,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(64))
    model.add(Activation('tanh'))

    rms = RMSprop()
    model.compile(loss='mse', optimizer=rms)

    batch_size = 16
    nb_epoch = 1000
    model.fit(X, Y,
              batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=2, validation_split=0.1)

    return model


windows, X, y = prepare_data()
model = train_model(X, y)
