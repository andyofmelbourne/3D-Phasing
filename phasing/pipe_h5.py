#!/usr/bin/env python3

import h5py
import pickle
import sys

fnam = sys.argv[1] 

# if fnam is of the type "/loc/filename.h5/dataset"
if len(fnam.split('.h5/')) > 1 :
    dataset = fnam.split('.h5/')[1]
    fnam = fnam.split('.h5/')[0] + '.h5'
else :
    dataset = '/'

def pipe_to_stdout(name, object):
    if isinstance(object, h5py.Dataset):
        print('sending data:', name, file=sys.stderr)
        pickle.dump({name: object[()]}, sys.stdout.buffer)

with h5py.File(fnam, 'r') as f:
    if isinstance(f[dataset], h5py.Dataset):
        pipe_to_stdout(f[dataset].name, f[dataset])
    else :
        f[dataset].visititems(pipe_to_stdout) 

