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
args = parser.parse_args()

# 1. read in electron density from stdin
pipe = pickle.load(sys.stdin.buffer)
den = pipe['electron_density']
vox = pipe['voxel_size']

# 2. calculate mod square of Fourier transform (I)
N = 2 * max(den.shape)

den2 = np.zeros((N,N,N), den.dtype)
den2[:den.shape[0], :den.shape[1], :den.shape[2]] = den

S = den2 > args.support_threshold

# normalised to approximate the continuous fourier trasform I = | int rho(x,y,z) e^{-2pi i r . q} dx dy dz |^2
I = np.abs(vox**3 * np.fft.fftn(den2))**2

print('number of electrons: {:.2e}'.format(vox**3*np.sum(den)), file=sys.stderr)
print('voxel_size:', vox, file=sys.stderr)
print('I.dtype  : {}'.format(I.dtype), file=sys.stderr)
print('mean(rho)  : {:.2e}'.format(np.mean(den)), file=sys.stderr)
print('I[0]  : {:.2e}'.format(I[0,0,0]), file=sys.stderr)
print('sum I : {:.2e}'.format(np.sum(I)), file=sys.stderr)

pickle.dump({'intensity': I, 'support': S, 'voxel_size': vox}, sys.stdout.buffer)


