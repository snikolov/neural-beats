'''Model a sequence of MIDI data using an LSTM. Represent each point
in the sequence as a binary vector representing active pitches.'''

import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np

from midi_util import print_array

# TODO: Specify pitch range during midi to array conversion, not here.
PITCH_RANGE = (30,60)
NUM_PITCHES = PITCH_RANGE[1] - PITCH_RANGE[0]

HIDDEN_UNITS_PER_PITCH = 16
NUM_HIDDEN_UNITS = NUM_PITCHES * HIDDEN_UNITS_PER_PITCH

PHRASE_LEN = 32

NUM_ITERATIONS = 60
BATCH_SIZE = 64

# Load the data.
# Concatenate all the vectorized midi files.
num_steps = 0
#base_dir = '/Users/snikolov/Downloads/groove-monkee-midi-gm/array'
base_dir = '/home/ubuntu/neural-beats/midi_arrays/'
arrays = []
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.split('.')[-1] == 'npy':
            array = np.load(os.path.join(root, filename))
            arrays.append(array[:, PITCH_RANGE[0] : PITCH_RANGE[1]])
seq = np.concatenate(arrays, axis=0)

# Construct labeled examples.
num_examples = seq.shape[0] - PHRASE_LEN
X = np.zeros((num_examples, PHRASE_LEN, NUM_PITCHES), dtype=np.bool)
y = np.zeros((num_examples, NUM_PITCHES), dtype=np.bool) 
for i in xrange(num_examples):
    X[i, :, :] = seq[i:i+PHRASE_LEN] > 0
    y[i, :] = seq[i+PHRASE_LEN] > 0
X = 1 * X
y = 1 * y


# Build the model.
model = Sequential()
model.add(LSTM(NUM_HIDDEN_UNITS, return_sequences=True, input_shape=(PHRASE_LEN, NUM_PITCHES)))
model.add(Dropout(0.2))
model.add(LSTM(NUM_HIDDEN_UNITS, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(NUM_HIDDEN_UNITS))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(NUM_PITCHES))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


np.random.seed(10)

def sample(preds, temperature=1.0):
    '''Given an array of values in [0,1] representing biased coin
    probabilities, generate a binary array of the same size.

    Flip each coin and generate a 0 or 1 according to the probability
    and the temperature. A higher temperature means it is more likely
    that a bit gets flipped.'''
    res = np.zeros(preds.shape)
    eps = 0.001
    for i, pred in enumerate(preds):
        # Adjust the binary probability distribution according to the
        # temperature.
        binary_dist = np.array([
            max(pred, eps), max(1-pred, eps)
        ])
        adjusted_dist = np.log(binary_dist) / temperature
        adjusted_dist = np.exp(adjusted_dist) / np.sum(np.exp(adjusted_dist))
        # Sample the adjusted distribution.
        if np.random.rand() < adjusted_dist[0]:
            res[i] = 1
    return res


# Train the model
print('Training the model...')
for i in range(NUM_ITERATIONS):
    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1)
    start_index = np.random.randint(0, len(seq) - PHRASE_LEN - 1)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- temperature:', temperature)

        generated = None
        phrase = seq[start_index: start_index + PHRASE_LEN]
        if generated is None:
            generated = phrase
        else:
            generated = np.concatenate([generated, phrase], axis=0)
        print('----- Generating with seed:')
        print_array(phrase)
        print('-----')
        print_array(generated)

        for i in range(400):
            preds = model.predict(phrase[np.newaxis,:], verbose=0)[0]
            next_slice = sample(preds, temperature)[np.newaxis,:]

            generated = np.concatenate([generated, next_slice], axis=0)
            phrase = np.concatenate([phrase[1:], next_slice], axis=0)

            print_array(next_slice)
        print()

# At the end, have a layer that blockwise predicts a number for each
# block of HIDDEN_UNITS_PER_PITCH, then passes that number through a
# sigmoid.

# Before the final layer, have multiple fully-connected layers with
# nonlinearities in between.

# Make the loss a sum of the binary cross entropies of all pitches.


#model.add(Dense(NUM_PITCHES))
#model.add(Activation('softmax'))
