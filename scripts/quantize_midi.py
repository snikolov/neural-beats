import argparse
import os

import mido
from mido import MidiFile

from midi_util import quantize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Quantize a directory of MIDI files.')
    parser.add_argument('path', help='Input path')
    parser.add_argument(
        '--quantization',
        default=4,
        help='Defines a 1/2**quantization note quantization grid')
    args = parser.parse_args()

    path_prefix, path_suffix = os.path.split(args.path)
    # Handle case where a trailing / requires two splits.
    if len(path_suffix) == 0:
        path_prefix, path_suffix = os.path.split(path_prefix)
    base_path_out = os.path.join(
        path_prefix, 'quantized')
    for root, dirs, files in os.walk(args.path):
        for file in files:
            if file.split('.')[-1] == 'mid':
                mid = quantize(MidiFile(os.path.join(root,file)))
                suffix = root.split(args.path)[-1]
                out_dir = os.path.join(base_path_out, suffix)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(out_dir, file)
                print 'Saving', out_file
                mid.save(out_file)
