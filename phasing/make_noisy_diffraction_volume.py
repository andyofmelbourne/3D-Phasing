#----------------------------------------------------------------------
#----------------------------------------------------------------------
# Create a noisy diffraction volume
# 1. read in Fourier intensities of molecule
# 2. convert to expected number of photons per voxel
#    n(q) = k I0 / (2 pi L q) x (wav r0 / (2L))^2 I
#    n = expected number of photons (q) 
#    k = quantum efficiency of detector pixels 
#    I0 = number of incident photons per square metre for experiment
#    L = linear size of molecule
#    wav = wavelength of incident light
#    r0 = Thomson scattering length or classical electron radius
#    I = mod square of Fourier transform of electron density
#    q = magnitude of reciprocal space coordinate
# 4. generate noisy diffraction using poisson statistics (n_noise)
# 5. rescale to recover noisy I (I_noise)
#    I_noise(q) = n_noise(q) / [k I0 / (2 pi L q) x (wav r0 / (2L))^2]
#----------------------------------------------------------------------
#----------------------------------------------------------------------

import numpy as np
import scipy.constants as sc
import argparse
import pickle
import sys


description = "Calculate the noisy diffraction volume of a molecule from its diffraction volume."
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--number_of_frames', type=float, default=1e6, \
                    help="Calculate the integrated pulse fluence on the sample from the number of frames at the SPB/SFX hutch at European-XFEL.")
args = parser.parse_args()


# 1. read in diffraction volume from stdin
pipe = pickle.load(sys.stdin.buffer)
I          = pipe['intensity']
voxel_size = pipe['voxel_size'] * 1e-10
S          = pipe['support']


number_of_frames = args.number_of_frames

photon_energy = 8e3 * sc.e
k = 1
wav = sc.h * sc.c / photon_energy
beam_width = 200e-9
pulse_energy = 2e-3 
pulse_photons = pulse_energy / photon_energy
pulse_fluence = pulse_photons / beam_width**2
r0 = sc.physical_constants['classical electron radius'][0]
# total incident photons per unit area over entire experiment
I0 = number_of_frames * 0.63 * pulse_fluence  

N = I.shape[0]
L = voxel_size * N/2

print('pulse photons: {:.2e}'.format(pulse_photons), file=sys.stderr)
print('total integrated incident photons / nm^2: {:.2e}'.format(I0 * 1e-18), file=sys.stderr)
print('total integrated incident photons over sample area: {:.2e}'.format(I0 * L**2), file=sys.stderr)



# 3. convert to expected number of photons per voxel
#    n(q) = k I0 / (2 pi L q) x (wav r0 / (2L))^2 I
qx = np.fft.fftfreq(N, d=voxel_size)
q  = np.sqrt(qx[:, np.newaxis, np.newaxis]**2 + qx[np.newaxis, :, np.newaxis]**2 + qx[np.newaxis, np.newaxis, :]**2)

# intersection probability, carefull with the origin
q[0, 0, 0] = 1 / (2 * np.pi * L)
q = 1 / (2 * np.pi * L * q)

# scale factor
q = k * I0 * q * (wav * r0 / (2 * L))**2

print(k * I0 * (wav * r0 / (2 * L))**2, file=sys.stderr)

# make n
I *= q 

print('number of recorded photons: {:.2e}'.format(np.sum(I)), file=sys.stderr)

# 4. generate noisy diffraction using poisson statistics (n_noise)
I = np.random.poisson(I)

# 5. rescale to recover noisy I (I_noise)
#    I_noise(q) = n_noise(q) / [k I0 / (2 pi L q) x (wav r0 / (2L))^2]
I = I / q

print('piping noisy diffraction data...', file=sys.stderr)
pickle.dump({'intensity': I, 'support': S, 'voxel_size': voxel_size}, sys.stdout.buffer)
