import os

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop

import midi_util


def features(pattern):
    '''Construct amplitude and phase spectrum features for single track
    drum pattern.'''

    f = np.fft.fft(1*(pattern>64))
    af = np.abs(f)
    pf = np.angle(f)
    return np.hstack([af, pf])


def pattern(features):
    '''Construct pattern from phase and amplitude features.'''


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
        X[i,:] = features(w[:,0])
        Y[i,:] = features(w[:,2])

    return windows, X, Y


def init_model():
    '''Define, initialize, and return model.'''

    model = Sequential()
    model.add(Dense(32, input_shape=(64,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.05))
    model.add(Dense(64))
    model.add(Activation('tanh'))

    rms = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer=rms)

    return model


def train_model(X, Y, path=None, load=True, save=True):
    '''Train a model mapping a kick pattern to an accompanying hihat
    pattern.'''

    model = init_model()
    if path is not None and load:
        model.load_weights(path)

    batch_size = 16
    nb_epoch = 200
    model.fit(X, Y,
              batch_size=batch_size, nb_epoch=nb_epoch,
              show_accuracy=True, verbose=2, validation_split=0.1)

    # Save model.
    if path is not None and save:
        out_dir, out_name = os.path.split(path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model.save_weights(path)

    return model


def load_model(path):
    '''Load a previous model and return it.'''

    model = init_model()
    model.load_weights(path)
    return model


windows, X, y = prepare_data()
model = train_model(X, y, path='models/accompaniment/model')
#model = load_model('models/accompaniment/model')

in_pattern = np.zeros(32)
in_pattern[::8] = 1
in_pattern[2::8] = 1
in_pattern[::5] = 1
x = features(in_pattern)[np.newaxis, :]
y = model.predict(x)
print y
y_amplitude = y[:,:32]
y_phase = y[:,32:]
y_complex = y_amplitude + 1j * y_phase
out_pattern = np.abs(np.fft.ifft(y_complex))
print out_pattern
out_pattern[out_pattern < 0.05] = 0
print out_pattern
out_pattern = (128 * out_pattern).astype(int)
print out_pattern

combined_pattern = np.vstack([in_pattern, out_pattern])
print combined_pattern
mid = midi_util.array_to_midi(combined_pattern.transpose(),
                              'test', quantization=5)
mid.save('beatzzz.mid')
