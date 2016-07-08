from nose.tools import *

from midi_util import *
import tests

def test_quantize_track_simple():
    qtrack0 = quantize_track(tests.track0, 240, 5)
    print qtrack0[0].time
    print qtrack0[1].time
    print qtrack0[2].time
    assert qtrack0[0].time == 0
    assert qtrack0[1].time == 0
    assert qtrack0[2].time == 50

    qtrack1 = quantize_track(tests.track1, 240, 5)
    print qtrack1[0].time
    print qtrack1[1].time
    print qtrack1[2].time
    assert qtrack1[0].time == 0
    assert qtrack1[1].time == 0
    assert qtrack1[2].time == 50


def test_quantize():
    mid = MidiFile('tests/midi/groove-monkee/Big Easy/Song 01 Funk/090 S01 Chorus Fill 2.mid')
    qmid = quantize(mid)

    # Check that the note onset and offset times are correct. Ignore
    # the meta messages at the beginning and the end.
    note_events = [
        Message(type='note_on', channel=9, note=42, velocity=71, time=0),
        Message(type='note_on', channel=9, note=36, velocity=100, time=0),
        Message(type='note_off', channel=9, note=42, velocity=64, time=36),
        Message(type='note_off', channel=9, note=36, velocity=64, time=1),
        Message(type='note_on', channel=9, note=42, velocity=90, time=83),
        Message(type='note_off', channel=9, note=42, velocity=64, time=36),
        Message(type='note_on', channel=9, note=36, velocity=88, time=24),
        Message(type='note_off', channel=9, note=36, velocity=64, time=36),
        Message(type='note_on', channel=9, note=38, velocity=98, time=24),
        Message(type='note_on', channel=9, note=42, velocity=78, time=0),
        Message(type='note_off', channel=9, note=38, velocity=64, time=36),
        Message(type='note_off', channel=9, note=42, velocity=64, time=0),
        Message(type='note_on', channel=9, note=38, velocity=90, time=24),
        Message(type='note_off', channel=9, note=38, velocity=64, time=36),
        Message(type='note_on', channel=9, note=42, velocity=97, time=24),
        Message(type='note_off', channel=9, note=42, velocity=64, time=36),
        Message(type='note_on', channel=9, note=36, velocity=100, time=24),
        Message(type='note_off', channel=9, note=36, velocity=64, time=36),
        Message(type='note_on', channel=9, note=46, velocity=111, time=24),
        Message(type='note_off', channel=9, note=46, velocity=64, time=36),
        Message(type='note_on', channel=9, note=38, velocity=95, time=24),
        Message(type='note_off', channel=9, note=38, velocity=64, time=36),
        Message(type='note_on', channel=9, note=38, velocity=106, time=24),
        Message(type='note_off', channel=9, note=38, velocity=64, time=36),
        Message(type='note_on', channel=9, note=36, velocity=100, time=84),
        Message(type='note_off', channel=9, note=36, velocity=64, time=37),
        Message(type='note_on', channel=9, note=38, velocity=95, time=23),
        Message(type='note_off', channel=9, note=38, velocity=64, time=36),
        Message(type='note_on', channel=9, note=38, velocity=106, time=24),
        Message(type='note_off', channel=9, note=38, velocity=64, time=31)]

    for i, msg in enumerate(qmid.tracks[1][2:-1]):
        assert msg == note_events[i]


def test_get_note_track():
    i, track = get_note_track(tests.midi_notes_in_track0)
    assert i == 0
    i, track = get_note_track(tests.midi_notes_in_track1)
    assert i == 1


@raises(AssertionError)
def test_quantize_tick_error():
    # Trigger an error when quantization is too high for the midi
    # resolution.
    quantize_tick(0, 240, 7)


def test_quantize_tick_normal():
    assert 0 == quantize_tick(14, 240, 5)
    assert 30 == quantize_tick(15, 240, 5)


