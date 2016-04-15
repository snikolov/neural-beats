'''Model a sequence of MIDI data. Each point in the sequence is a
number from 0 to 2**p-1 that represents a configuration of p pitches
that may be on or off.'''

import itertools
import os

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras import backend as K
import numpy as np

from midi_util import array_to_midi, print_array


# The pitches we are paying attention to in the data
# kick, snare, closed hihat, open hihat
PITCHES = [36,38,42,51]
# The pitches we want to generate (potentially different drum kit)
OUT_PITCHES = [54,56,58,60]
NUM_HIDDEN_UNITS = 128
PHRASE_LEN = 32
NUM_ITERATIONS = 60
BATCH_SIZE = 64
MIN_HITS = 8

# Encode each configuration of p pitches, each on or off, as a
# number between 0 and 2**p-1.
assert len(PITCHES) <= 8, 'Too many configurations for this many pitches!'
encodings = {
    config : i
    for i, config in enumerate(itertools.product([0,1], repeat=len(PITCHES)))
}

decodings = {
    i : config
    for i, config in enumerate(itertools.product([0,1], repeat=len(PITCHES)))
}


def encode(midi_array):
    '''Encode a folded MIDI array into a sequence of integers.'''
    return [
        encodings[tuple((time_slice>0).astype(int))]
        for time_slice in midi_array
    ]


def decode(config_ids):
    '''Decode a sequence of integers into a folded MIDI array.'''
    velocity = 120
    return velocity * np.vstack(
        [list(decodings[id]) for id in config_ids])


def unfold(midi_array, pitches):
    '''Unfold a folded MIDI array with the given pitches.'''
    # Create an array of all the 128 pitches and fill in the
    # corresponding pitches.
    res = np.zeros((midi_array.shape[0], 128))
    assert midi_array.shape[1] == len(pitches), 'Mapping between unequal number of pitches!'
    for i in xrange(len(pitches)):
        res[:,pitches[i]] = midi_array[:,i]
    return res


# Load the data.
# Concatenate all the vectorized midi files.
num_steps = 0
#base_dir = '/Users/snikolov/Downloads/groove-monkee-midi-gm/array'
base_dir = '/home/ubuntu/neural-beats/midi_arrays/'
# Sequence of configuration numbers representing combinations of
# active pitches.
config_seq = []
for root, dirs, files in os.walk(base_dir):
    for filename in files:
        if filename.split('.')[-1] != 'npy':
            continue
        array = np.load(os.path.join(root, filename))
        print np.sum(np.sum(array[:, PITCHES]>0))
        if np.sum(np.sum(array[:, PITCHES]>0)) < MIN_HITS:
            continue
        config_seq.extend(encode(array[:,PITCHES]))
config_seq = np.array(config_seq)

# Construct labeled examples.
num_examples = len(config_seq) - PHRASE_LEN
X = np.zeros((num_examples, PHRASE_LEN, 2**len(PITCHES)), dtype=np.bool)
y = np.zeros((num_examples, 2**len(PITCHES)), dtype=np.bool) 
for i in xrange(num_examples):
    for j, cid in enumerate(config_seq[i:i+PHRASE_LEN]):
        X[i, j, cid] = 1
    y[i, config_seq[i+PHRASE_LEN]] = 1
X = 1 * X
y = 1 * y


# Build the model.
model = Sequential()
model.add(LSTM(NUM_HIDDEN_UNITS, return_sequences=True, input_shape=(PHRASE_LEN, 2**len(PITCHES))))
model.add(Dropout(0.2))
model.add(LSTM(NUM_HIDDEN_UNITS, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(2**len(PITCHES)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


np.random.seed(10)

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


# Train the model

midi_out_dir = '/home/ubuntu/neural-beats/midi-out'
model_out_dir = '/home/ubuntu/neural-beats/models'
model_name = 'model'
if not os.path.exists(midi_out_dir):
    os.makedirs(midi_out_dir)
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)
print('Training the model...')

LOAD_WEIGHTS = True
if LOAD_WEIGHTS:
    print('Loading previous weights...')
    model.load_weights(os.path.join(model_out_dir, model_name))

for i in range(NUM_ITERATIONS):
    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1)
    start_index = np.random.randint(0, len(config_seq) - PHRASE_LEN - 1)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        #print()
        #print('----- temperature:', temperature)

        generated = []
        phrase = list(config_seq[start_index: start_index + PHRASE_LEN])

        #print('----- Generating with seed:')
        phrase_array = decode(phrase)
        print_array(phrase_array)
        #print('-----')

        for j in range(512):
            x = np.zeros((1, PHRASE_LEN, 2**len(PITCHES)))
            for t, config_id in enumerate(phrase):
                x[0, t, config_id] = 1
            preds = model.predict(x, verbose=0)[0]
            next_id = sample(preds, temperature)

            generated += [next_id]
            phrase = phrase[1:] + [next_id]
            #print_array(decode([next_id]))
        #print()
        mid_name = 'out_{}_{}.mid'.format(i,temperature)
        mid = array_to_midi(unfold(decode(generated), OUT_PITCHES), mid_name)
        mid.save(os.path.join(midi_out_dir, mid_name))
    model.save_weights(os.path.join(model_out_dir, model_name), overwrite=True)
