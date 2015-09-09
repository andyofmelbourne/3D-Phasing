#!/usr/bin/env python

import numpy as np
import time
import sys
import ConfigParser
from utils import io_utils

"""
#
# GPU stuff 
try :
    import pyfft
    import pyopencl
    import pyopencl.array
    from pyfft.cl import Plan
    import pyopencl.clmath
    GPU_calc = True
    from src.projection_maps_gpu import *
except :
    GPU_calc = False
    from src.projection_maps import *


# get the CUDA platform
platforms = pyopencl.get_platforms()
for p in platforms:
    if p.name == 'NVIDIA CUDA':
        platform = p

# get one of the gpu's device id
device = platform.get_devices()[0]

# create a context for the device
context = pyopencl.Context([device])

# create a command queue for the device
queue = pyopencl.CommandQueue(context)

# make a plan for the ffts
plan = Plan(shape, dtype=np.complex128, queue=queue)
"""

shape = (128, 128, 128)

psi     = np.random.random(shape) + 1.0J * np.random.random(shape)
support = np.zeros(shape, dtype=np.int8)
support[: 10, : 12, : 5] = 1
amp     = np.abs(np.fft.fftn(psi * support))**2
amp     = np.sqrt(amp)
phase   = np.zeros_like(amp)

psi     = np.random.random(shape) + 1.0J * np.random.random(shape)
psi    *= support
psi     = np.fft.ifftshift(psi)
support = np.fft.ifftshift(support)

good_pix = (np.random.random(shape) > 0.5).astype(np.int8)

"""
# send it to the gpu
psi_gpu     = pyopencl.array.to_device(queue, np.ascontiguousarray(psi))
support_gpu = pyopencl.array.to_device(queue, np.ascontiguousarray(support))
amp_gpu     = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
phase_gpu   = pyopencl.array.to_device(queue, np.ascontiguousarray(phase))
good_pix_gpu  = pyopencl.array.to_device(queue, np.ascontiguousarray(good_pix))
"""

"""
plan.execute(psi_gpu.data)
#psi  = np.fft.fftn(psi)
#print '\n fft psi:', np.sum(np.abs( psi_gpu.get() - psi)**2) / np.sum(np.abs(psi)**2)

phase_gpu = pyopencl.clmath.atan2(psi_gpu.imag, psi_gpu.real, queue=queue)
#phase     = np.angle(psi)
#print '\n phase :', np.sum(np.abs( phase_gpu.get() - phase)**2) / np.sum(np.abs(phase)**2)

psi_gpu   = psi_gpu * (1. - good_pix_gpu)
psi_gpu   = psi_gpu + amp_gpu * pyopencl.clmath.exp(1J * phase_gpu, queue=queue) * good_pix_gpu

#psi           = psi * (1 - good_pix)
#psi           = psi + amp * np.exp(1J * phase) * good_pix 
#print '\n exp psi:', np.sum(np.abs( psi_gpu.get() - psi)**2) / np.sum(np.abs(psi)**2)

plan.execute(psi_gpu.data, inverse=True)
#psi  = np.fft.ifftn(psi)
#print '\n ifft psi:', np.sum(np.abs( psi_gpu.get() - psi)**2) / np.sum(np.abs(psi)**2)
"""

import src.projection_maps_gpu as pm_gpu
proj = pm_gpu.Projections(psi.shape, psi.dtype)
psi_gpu, amp_gpu, support_gpu, good_pix_gpu = proj.send_to_gpu(psi, amp, support, good_pix)

#psi_gpu = proj.Pmod(amp_gpu, psi_gpu, good_pix_gpu)
print 'performing gpu ERA...'
for i in range(10):
    psi_gpu, mod_err_gpu = proj.DM(psi_gpu, support_gpu, good_pix_gpu, amp_gpu)
    print 'gpu', i, mod_err_gpu

import src.projection_maps as pm
#psi = pm.Pmod(amp, psi, good_pix)
print 'performing ERA...'
for i in range(10):
    psi, mod_err, sup_err = pm.DM(psi, support, good_pix, amp)
    print 'cpu', i, mod_err

#print '\n comparison with Pmod:', np.sum(np.abs(psi - psi_gpu.get())**2)
print '\n comparison with ERA', np.sum(np.abs(psi - psi_gpu.get())**2)
print '\n comparison with mod error', mod_err, mod_err_gpu
