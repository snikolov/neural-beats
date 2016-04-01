from mido import Message, MetaMessage, MidiFile, MidiTrack

track0 = MidiTrack()
track0.append(MetaMessage('track_name', name='test0', time=0))
track0.append(Message('note_on', note=60, velocity=64, time=0))
track0.append(Message('note_off', note=60, velocity=64, time=50))
track0.append(MetaMessage('end_of_track', time=0))

track1 = MidiTrack()
track1.append(MetaMessage('track_name', name='test1', time=0))
track1.append(Message('note_on', note=60, velocity=64, time=2))
track1.append(Message('note_off', note=60, velocity=64, time=50))
track1.append(MetaMessage('end_of_track', time=0))
