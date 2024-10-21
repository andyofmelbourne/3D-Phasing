import numpy as np
import pickle
from scipy.ndimage import gaussian_filter
import os

import argparse
import sys

if __name__ == '__main__':
    description = "Produce file for 3D-phasing input from EMC files."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                        help="Python pickle file containing a dictionary with keys 'intensity' and 'support'")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), \
                        help="Python pickle output file. The result is written as a dictionary with the key 'object'")
    parser.add_argument('--beamstop', type=int, default = 16, \
                        help="must be >= beamstop radius in pixel units. Set to -1 to disable.")
    parser.add_argument('--cut_edges', action = 'store_true',
                        help="cut the (probably noisy) diffraction in outer most shell")
    parser.add_argument('--soft_edges', action = 'store_true',
                        help="taper the edges of the diffraction volume to avoid sharp change in intensity")
    parser.add_argument('--size', type=float, default = 0,
                        help="sample size in nm for initial support, if 0 then size is set to = shape / 2")
    args = parser.parse_args()

# merge with emc intensities
d = pickle.load(args.input)
I       = np.fft.ifftshift(d['I'])
overlap = np.fft.ifftshift(d['overlap'])
dq      = d['dq']

# assume cube
assert(np.allclose(I.shape, I.shape[0]))

i = np.fft.fftfreq(I.shape[0], 1/I.shape[0])
r = (i[:,None,None]**2 + i[None,:,None]**2 + i[None,None,:]**2)**0.5
mask = np.ones(I.shape, dtype = bool)
rmax = i.max()

# mask inner pixels (allow to float) if overlap is zero and within beamstop region
if args.beamstop > 0 :
    mask[ (r<args.beamstop) * (overlap == 0) ] = False

if args.cut_edges :
    rmax = 0.95 * i.max()
    I[r > rmax] = 0.

if args.soft_edges :
    mask_filter = r < rmax
    mask_filter = np.fft.ifftshift(gaussian_filter(np.fft.fftshift(mask_filter.astype(float)), 2, mode='constant', truncate=8.)) > 0.8
    mask_filter = np.fft.ifftshift(gaussian_filter(np.fft.fftshift(mask_filter.astype(float)), 2, mode='constant', truncate=8.))
    I *= mask_filter

if args.size == 0 :
    args.size = I.shape[0]/2 - 2

# dx = 1 / (N dq) 
dx = 1 / (I.shape[0] * dq) * 1e9
S = (dx * r) <= args.size

pickle.dump({'intensity': I, 'mask': mask, 'support': S}, args.output)

