from collections import defaultdict
import copy
from math import log, floor, ceil
import pprint

import mido
from mido import MidiFile, MidiTrack
import numpy as np


DEBUG = False

def quantize_tick(tick, ticks_per_quarter, quantization):
    '''Quantize the timestamp or tick.

    Arguments:
    tick -- An integer timestamp
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as 1/2**quantization
    '''
    assert (ticks_per_quarter * 4) % 2 ** quantization == 0, \
        'Quantization too fine. Ticks per quantum must be an integer.'
    ticks_per_quantum = (ticks_per_quarter * 4) / float(2 ** quantization)
    quantized_ticks = int(
        round(tick / float(ticks_per_quantum)) * ticks_per_quantum)
    return quantized_ticks


def quantize_track(track, ticks_per_quarter, quantization):
    '''Return the differential time stamps of the note_on, note_off, and
    end_of_track events, in order of appearance, with the note_on events
    quantized to the grid given by the quantization.

    Arguments:
    track -- MIDI track containing note event and other messages
    ticks_per_quarter -- The number of ticks per quarter note
    quantization -- The note duration, represented as
      1/2**quantization.'''

    pp = pprint.PrettyPrinter()
    # Message timestamps are represented as differences between
    # consecutive events. Annotate messages with cumulative timestamps.
    cum_msgs = zip(
        np.cumsum([msg.time for msg in track[1:-1]]),
        [msg for msg in track[1:-1]])

    quantized_track = MidiTrack()
    quantized_track.append(track[0])
    quantized_track.append(track[-1])
    # Keep track of note_on events that have not had an off event yet.
    # note number -> message
    open_msgs = defaultdict(list)
    quantized_msgs = []
    for cum_time, msg in cum_msgs:
        if DEBUG:
            print msg
            pp.pprint(open_msgs)
        if msg.type == 'note_on' and msg.velocity > 0:
            # Store until note off event. Note that there can be
            # several note events for the same note. Subsequent
            # note_off events will be associated with these note_on
            # events in FIFO fashion.
            open_msgs[msg.note].append((cum_time, msg))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            assert msg.note in open_msgs, \
                'Bad MIDI. Cannot have note off event before note on event'
            note_on_open_msgs = open_msgs[msg.note]
            note_on_cum_time, note_on_msg = note_on_open_msgs[0]
            open_msgs[msg.note] = note_on_open_msgs[1:]
            # Quantized note_on time
            quantized_note_on_cum_time = quantize_tick(
                note_on_cum_time, ticks_per_quarter, quantization)
            # The cumulative time of note_off is the quantized
            # cumulative time of note_on plus the orginal difference
            # of the unquantized cumulative times.
            quantized_note_off_cum_time = quantized_note_on_cum_time + (cum_time - note_on_cum_time)
            quantized_msgs.append((quantized_note_on_cum_time, note_on_msg))
            quantized_msgs.append((quantized_note_off_cum_time, msg))

    # Now, sort the quantized messages by (cumulative time,
    # note_type), making sure that note_on events come before note_off
    # events when two event have the same cumulative time. Compute
    # differential times and construct the quantized track messages.
    quantized_msgs.sort(
        key=lambda (cum_time, msg): cum_time if msg.type=='note_on' else cum_time + 0.5)
    diff_times = [0] + list(
        np.diff([ msg[0] for msg in quantized_msgs ]))
    for diff_time, (cum_time, msg) in zip(diff_times, quantized_msgs):
        quantized_track.insert(-2, msg.copy(time=diff_time))
    if DEBUG:
        pp.pprint(quantized_msgs)
        pp.pprint(diff_times)
    return quantized_track


def quantize(mid, quantization=5):
    '''Return a midi object whose notes are quantized to
    1/2**quantization notes.

    Arguments:
    mid -- MIDI object
    quantization -- The note duration, represented as
      1/2**quantization.'''

    quantized_mid = copy.deepcopy(mid)
    # By convention, Track 0 contains metadata and Track 1 contains
    # the note on and note off events.
    quantized_mid.tracks[1] = quantize_track(
        mid.tracks[1], mid.ticks_per_beat, quantization)
    return quantized_mid


def midi_to_array(mid, quantization):
    '''Return array representation of a 4/4 time signature, MIDI object.

    Normalize the number of time steps in track to a power of 2. Then
    construct a T x N array A (T = number of time steps, N = number of
    MIDI note numbers) where A(t,n) is the velocity of the note number
    n at time step t if the note is active, and 0 if it is not.

    Arguments:
    mid -- MIDI object with a 4/4 time signature
    quantization -- The note duration, represented as
      1/2**quantization.'''

    time_sig_msgs = [ msg for msg in mid.tracks[0] if msg.type == 'time_signature' ]
    assert len(time_sig_msgs) == 1, 'No time signature found'
    time_sig = time_sig_msgs[0]
    assert time_sig.numerator == 4 and time_sig.denominator == 4, 'Not 4/4 time.'

    # Quantize the notes to a grid of time steps.
    mid = quantize(mid, quantization=quantization)

    # Convert the note timing and velocity to an array.
    track = mid.tracks[1]
    ticks_per_quarter = mid.ticks_per_beat
    
    time_msgs = [msg for msg in track if hasattr(msg, 'time')]
    cum_times = np.cumsum([msg.time for msg in time_msgs])
    track_len_ticks = cum_times[-1]
    if DEBUG:
        print 'Track len in ticks:', track_len_ticks
    notes = [
        (time * (2**quantization/4) / (ticks_per_quarter), msg.note, msg.velocity)
        for (time, msg) in zip(cum_times, time_msgs)
        if msg.type == 'note_on' ]
    num_steps = int(round(track_len_ticks / float(ticks_per_quarter)*2**quantization/4))
    normalized_num_steps = nearest_pow2(num_steps)

    if DEBUG:
        print num_steps
        print normalized_num_steps

    num_note_numbers = 128
    midi_array = np.zeros((normalized_num_steps, num_note_numbers))
    for (position, note_num, velocity) in notes:
        if position == normalized_num_steps:
            print 'Warning: truncating from position {} to {}'.format(position, normalized_num_steps - 1)
            position = normalized_num_steps - 1
        if position > normalized_num_steps:
            print 'Warning: skipping note at position {} (greater than {})'.format(position, normalized_num_steps)
            continue
        midi_array[position, note_num] = velocity
    
    return midi_array
    

def nearest_pow2(x):
    '''Normalize input to nearest power of 2. Round down when halfway
    between two powers of two.'''

    low = 2**int(floor(log(x, 2)))
    high = 2**int(ceil(log(x, 2)))
    if high - x < x - low:
        nearest = high
    else:
        nearest = low
    return nearest
