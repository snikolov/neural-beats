from midi_util import quantize_track
from tests import *

def test_quantize():
    qtrack0 = quantize_track(track0, 240, 5)
    print qtrack0[0].time
    print qtrack0[1].time
    assert qtrack0[0].time == 0
    assert qtrack0[1].time == 50
    
    qtrack1 = quantize_track(track1, 240, 5)
    print qtrack1[0].time
    print qtrack1[1].time
    assert qtrack1[0].time == 0
    assert qtrack1[1].time == 50
