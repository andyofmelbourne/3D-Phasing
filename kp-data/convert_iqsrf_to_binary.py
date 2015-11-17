import numpy as np

import os, sys, getopt, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils import zero_pad
from utils import io_utils

fnam = 'iqsrf'

f = open(fnam + '.dat', 'r')

d = [l.split() for l in f]

data = np.array(d, dtype=np.float64)

i = int(np.rint(len(d)**(1/3.)))

data = data[:, 3].reshape( (i, i, i) )

data = zero_pad.zero_pad_to_nearest_pow2(data)

data.tofile(fnam + '_' + str(data.shape[0]) + 'x'+str(data.shape[1]) + 'x' + str(data.shape[2]) + '_float64.bin')
f.close()

