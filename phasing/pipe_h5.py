#!/usr/bin/env python3

import h5py
import pickle
import sys
import argparse
import tqdm

if __name__ == '__main__':
    description = "pip h5 datasets to stdout"
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fnam', type=str, \
                        help="filename with optional dataset selection, e.g. file.h5/dataset")
    parser.add_argument('-s', '--stream', action="store_true", \
                        help="stream data rather than send the entire dataset.")
    args = parser.parse_args()
    fnam = args.fnam 

# if fnam is of the type "/loc/filename.h5/dataset"
if len(fnam.split('.h5/')) > 1 :
    dataset = fnam.split('.h5/')[1]
    fnam = fnam.split('.h5/')[0] + '.h5'
elif len(fnam.split('.cxi/')) > 1 :
    dataset = fnam.split('.cxi/')[1]
    fnam = fnam.split('.cxi/')[0] + '.cxi'
else :
    dataset = '/'

def pipe_to_stdout(name, object):
    if isinstance(object, h5py.Dataset):
        print('sending data:', name, file=sys.stderr)

        if args.stream :
            for i in tqdm.tqdm(range(object.shape[0]), desc='streaming data'):
                pickle.dump({name: object[i]}, sys.stdout.buffer)
        else :
            pickle.dump({name: object[()]}, sys.stdout.buffer)

with h5py.File(fnam, 'r') as f:
    if isinstance(f[dataset], h5py.Dataset):
        pipe_to_stdout(f[dataset].name, f[dataset])
    else :
        f[dataset].visititems(pipe_to_stdout) 

