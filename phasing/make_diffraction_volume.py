#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Create a noisy diffraction volume
# 1. read in electron density from file
#    zero padd the array to a length of 2*L in each dimension
#    where L is the largest side length of the array
# 2. calculate mod square of Fourier transform (I)
# 3. calculate support region for phase retrieval
# 4. write the Fourier intensities back into original file
#----------------------------------------------------------------------
#----------------------------------------------------------------------
import numpy as np
import argparse
import pickle
import sys

description = "Calculate the diffraction volume of a molecule from its electron density. The density "
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--support_threshold', type=float, default=1e-7, \
                    help="Values above this threshold are included in the support volume (1 for inside molecule and 0 otherwise)")
parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                    help="Python pickle file containing a dictionary with keys 'electron_density' and 'voxel_size'")
parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout.buffer, \
                    help="Python pickle output file. The result is written as a dictionary with the keys 'intensity', 'support' and 'voxel_size'")
args = parser.parse_args()

# 1. read in electron density from stdin
pipe = pickle.load(args.input)
den = pipe['electron_density']
vox = pipe['voxel_size']

# 2. calculate mod square of Fourier transform (I)
N = 2 * max(den.shape)

den2 = np.zeros((N,N,N), den.dtype)
den2[:den.shape[0], :den.shape[1], :den.shape[2]] = den

S = den2 > args.support_threshold

# normalised to approximate the continuous fourier trasform I = | int rho(x,y,z) e^{-2pi i r . q} dx dy dz |^2
I = np.abs(vox**3 * np.fft.fftn(den2))**2

pickle.dump({'intensity': I, 'support': S, 'voxel_size': vox}, args.output)


