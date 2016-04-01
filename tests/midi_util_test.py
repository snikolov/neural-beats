from mido import MidiFile
from midi_util import get_quantized_note_times

def test_quantize():
    mid = MidiFile('/Users/snikolov/Downloads/groove-monkee-midi-gm/Twisted/Rock/080 Rock Toms 1 F2.mid')
    get_quantized_note_times(mid.tracks[1], mid.ticks_per_beat, 5)
