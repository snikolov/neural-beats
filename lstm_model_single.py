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


np.random.seed(10)

# All the pitches represented in the MIDI data arrays.
# TODO: Read pitches from pitches.txt file in corresponding midi array
# directory.
PITCHES = [36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 49, 50, 58, 59, 60, 61, 62, 63, 64, 66]
# The subset of pitches we'll actually use.
IN_PITCHES = [36, 38, 41, 42, 47, 58, 59, 61]
# The pitches we want to generate (potentially for different drum kit)
OUT_PITCHES = [54, 56, 58, 60, 61, 62, 63, 64]
# The minimum number of hits to keep a drum loop after the types of
# hits have been filtered by IN_PITCHES.
MIN_HITS = 8

########################################################################
# Network architecture parameters.
########################################################################
NUM_HIDDEN_UNITS = 128
# The length of the phrase from which the predict the next symbol.
PHRASE_LEN = 512
# Dimensionality of the symbol space.
SYMBOL_DIM = 2**len(IN_PITCHES)
NUM_ITERATIONS = 60
BATCH_SIZE = 64

MIDI_IN_DIR = '/home/ubuntu/neural-beats/midi_arrays/mega'
#MIDI_IN_DIR = '/Users/snikolov/Downloads/groove-monkee-midi-gm/array'
MIDI_OUT_DIR = '/home/ubuntu/neural-beats/midi-out'
MODEL_OUT_DIR = '/home/ubuntu/neural-beats/models'
MODEL_NAME = 'model_20160517'
LOAD_WEIGHTS = False


# Encode each configuration of p pitches, each on or off, as a
# number between 0 and 2**p-1.
assert len(IN_PITCHES) <= 8, 'Too many configurations for this many pitches!'
encodings = {
    config : i
    for i, config in enumerate(itertools.product([0,1], repeat=len(IN_PITCHES)))
}

decodings = {
    i : config
    for i, config in enumerate(itertools.product([0,1], repeat=len(IN_PITCHES)))
}


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))


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


def prepare_data():
    # Load the data.
    # Concatenate all the vectorized midi files.
    num_steps = 0

    # Sequence of configuration numbers representing combinations of
    # active pitches.
    config_seq = []
    num_dirs = len([x for x in os.walk(MIDI_IN_DIR)])

    in_pitch_indices = [ PITCHES.index(p) for p in IN_PITCHES ]
    for dir_idx, (root, dirs, files) in enumerate(os.walk(MIDI_IN_DIR)):
        for filename in files:
            if filename.split('.')[-1] != 'npy':
                continue
            array = np.load(os.path.join(root, filename))
            if np.sum(np.sum(array[:, in_pitch_indices]>0)) < MIN_HITS:
                continue
            config_seq.extend(encode(array[:, in_pitch_indices]))
        print 'Loaded {}/{} directories'.format(dir_idx, num_dirs)
    config_seq = np.array(config_seq)

    # Construct labeled examples.
    num_examples = len(config_seq) - PHRASE_LEN
    X = np.zeros((num_examples, PHRASE_LEN, SYMBOL_DIM), dtype=np.bool)
    y = np.zeros((num_examples, SYMBOL_DIM), dtype=np.bool) 
    for i in xrange(num_examples):
        for j, cid in enumerate(config_seq[i:i+PHRASE_LEN]):
            X[i, j, cid] = 1
        y[i, config_seq[i+PHRASE_LEN]] = 1
    X = 1 * X
    y = 1 * y

    return config_seq, X, y


def generate(model, seed, mid_name, temperature=1.0, length=512):
    '''Generate sequence using model, seed, and temperature.'''

    generated = []
    phrase = seed

    #print('----- Generating with seed:')
    phrase_array = decode(phrase)
    print_array(phrase_array)
    #print('-----')

    for j in range(length):
        x = np.zeros((1, PHRASE_LEN, SYMBOL_DIM))
        for t, config_id in enumerate(phrase):
            x[0, t, config_id] = 1
        preds = model.predict(x, verbose=0)[0]
        next_id = sample(preds, temperature)

        generated += [next_id]
        phrase = phrase[1:] + [next_id]
        #print_array(decode([next_id]))
    #print()
    mid = array_to_midi(unfold(decode(generated), OUT_PITCHES), mid_name)
    mid.save(os.path.join(MIDI_OUT_DIR, mid_name))
    return mid


def init_model():
    # Build the model.
    model = Sequential()
    model.add(LSTM(
        NUM_HIDDEN_UNITS,
        return_sequences=True,
        input_shape=(PHRASE_LEN, SYMBOL_DIM)))
    model.add(Dropout(0.2))
    model.add(LSTM(NUM_HIDDEN_UNITS, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(SYMBOL_DIM))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def train(config_seq, X, y):
    '''Train model and save weights.'''

    model = init_model()
    # Train the model
    if not os.path.exists(MIDI_OUT_DIR):
        os.makedirs(MIDI_OUT_DIR)
    if not os.path.exists(MODEL_OUT_DIR):
        os.makedirs(MODEL_OUT_DIR)
    print('Training the model...')

    if LOAD_WEIGHTS:
        print('Loading previous weights...')
        model.load_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME))

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
                x = np.zeros((1, PHRASE_LEN, SYMBOL_DIM))
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
            mid.save(os.path.join(MIDI_OUT_DIR, mid_name))
        model.save_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME), overwrite=True)
    return model


def run():
    config_seq, X, y = prepare_data()
    train(config_seq, X, y)

    '''
    model = init_model()
    model.load_weights(os.path.join(MODEL_OUT_DIR, MODEL_NAME))
    seed = np.zeros((32, 4))

    """
    # Normal techno pattern
    seed[0,0] = 1 # Kick
    seed[4,2] = 1 # hat
    seed[8,0] = 1 # Kick
    seed[12,2] = 1 # hat
    seed[16,0] = 1 # Kick
    seed[20,2] = 1 # hat
    seed[24,0] = 1 # Kick
    seed[28,2] = 1 # hat
    """

    # Broken beat / electro pattern
    seed[0,0] = 1 # Kick
    seed[8,1] = 1 # Snare
    seed[12,0] = 1 # Kick
    seed[24,1] = 1 # Snare
    seed[30,1] = 1 # Snare

    length = 4096
    for temperature in [0.7,0.9,1.1]:
        for i in xrange(5):
            generate(model,
                     encode(seed),
                     'out_electro_{}_{}_{}.mid'.format(length, temperature, i),
                     temperature=1.1,
                     length=length)
    '''
run()
