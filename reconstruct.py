#!/usr/bin/env python

import numpy as np
import time
import sys
import ConfigParser
from scipy import ndimage
from utils import io_utils

#
# GPU stuff 
try :
    import pyfft
    import pyopencl
    import pyopencl.array
    from pyfft.cl import Plan
    GPU_calc = True
except :
    GPU_calc = False


def read_data(params):
    diff = np.fromfile(params['input']['filename'], dtype = np.dtype(params['input']['dtype']))
    diff = diff.reshape((params['input']['i'], params['input']['j'], params['input']['k']))
    return diff


def fit_sphere_to_autoc(diff):
    autoc = np.fft.ifftn(diff)
    autoc = np.fft.fftshift(autoc)
    autoc = np.abs(autoc)

    norm  = np.sum(autoc)
    norm2 = np.sum(autoc**2)
    
    i, j, k = np.indices(diff.shape)
    r = (i-diff.shape[0]/2)**2 + (j-diff.shape[1]/2)**2 + (k-diff.shape[2]/2)**2 

    print '\n finding the least squares fit for the autocorrelation function'
    print   ' of a sphere with the autocorrelation from the diffraction data'
    print '\npixel radius     error'

    errs = []
    rads = []
    for rad in range(2, diff.shape[0]/2):
        sphere       = (r < rad**2).astype(np.float64)
        sphere_autoc = np.abs(np.fft.fftn(sphere))**2
        sphere_autoc = np.fft.ifftn(sphere_autoc)
        sphere_autoc = np.abs(np.fft.fftshift(sphere_autoc))
        sphere_autoc *= norm / np.sum(sphere_autoc)
        
        error = np.sum( (autoc - sphere_autoc)**2 ) / norm2
        
        print rad, error
        errs.append(error)
        rads.append(rad)
    
    rad = rads[np.argmin(errs)]
    sphere       = (r < rad**2).astype(np.float64)
    
    print '\n\n best match with r =', rad, 'pixels'
    return sphere


def Pmod(amp, psi, good_pix):
    psi           = np.fft.fftn(psi)
    phase         = np.angle(psi)
    phase         = ndimage.gaussian_filter(phase, 0.5)
    psi[good_pix] = amp[good_pix] * np.exp(1J * phase[good_pix])
    psi           = np.fft.ifftn(psi)
    return psi


def shrink(arrayin, thresh, blur):
    mask = arrayin > np.median(arrayin)
    mask = threshExpand(mask, 0.5, blur)
    return mask


def threshExpand(arrayin, thresh=0.1e0, blur=8):
    """Threshold the array then gaussian blur then rethreshold to expand the region.
    
    Output a True/False mask."""
    arrayout = ndimage.gaussian_filter(np.abs(arrayin).astype(np.float64),blur)
    thresh2  = np.max(np.abs(arrayout)) * thresh
    arrayout = 1.0 * (np.abs(arrayout) > thresh2)
    
    arrayout = ndimage.gaussian_filter(arrayout,2*blur)
    thresh2  = np.max(np.abs(arrayout))*thresh
    arrayout = np.array(1.0 * (np.abs(arrayout) > thresh2), dtype=np.bool)  
    return arrayout


def _ERA(psi, support, good_pix, amp):
    psi_sup = psi * support
    psi     = Pmod(amp, psi_sup.copy(), good_pix) 
    
    delta = psi - psi_sup
    support_err = np.sum( (delta * np.conj(delta)).real )
    
    delta = psi - psi_sup
    mod_err = np.sum( (delta * np.conj(delta)).real )

    return psi, mod_err, support_err


def _DM(psi, support, good_pix, amp):
    psi_sup = psi * support

    psi += Pmod(amp, 2*psi_sup - psi, good_pix) - psi_sup
    
    delta       = psi * ~support
    support_err = np.sum( (delta * np.conj(delta)).real ) 
    
    delta       = psi - Pmod(amp, psi, good_pix)
    mod_err     = np.sum( (delta * np.conj(delta)).real ) 
    return psi, mod_err, support_err


def iterate(diff, support, mask, params):
    # shift quadrants for faster iters
    good_pix = np.where(np.fft.ifftshift(mask))
    amp      = np.sqrt(np.fft.ifftshift(diff))
    support  = np.fft.ifftshift(support)
    
    # initial guess
    print '\n inital estimate: random numbers b/w 0 and 1 (just real)'
    support_error, mod_error = [], []
    psi  = np.random.random(amp.shape) + 0J 
    psi *= support
    
    alg = params['recon']['alg'].split()
    iters = np.array(alg[::2], dtype=np.int)
    algs  = alg[1::2]
    
    support_error, mod_error = [], []
    i = 0
    for it, alg in zip(iters, algs):
        for j in range(it):
            if alg == 'DM' :
                psi, mod_err, support_err = _DM(psi, support, good_pix, amp)
                
            elif alg == 'ERA':
                psi, mod_err, support_err = _ERA(psi, support, good_pix, amp)

            elif alg == 'shrink':
                print '\n performing shrink wrap:'
                shrink_mask = shrink(psi * support, thresh=params['shrink']['thresh'], blur=params['shrink']['blur'])
                print ' cut', np.sum(support) - np.sum(shrink_mask), 'pixels'
                support = shrink_mask.copy()
                
            support_error.append(support_err)
            mod_error.append(mod_err)
            print alg, i, support_error[-1], mod_error[-1]
            i += 1
    
    return psi, support, support_error, mod_error
        

def truncate(diff, n):
    c = np.array(diff.shape)/2
    diff_out = diff[c[0]-n:c[0]+n, c[1]-n:c[1]+n, c[2]-n:c[2]+n].copy()
    return diff_out
        

def test_gpu_ffts():
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
    plan = Plan((128, 128, 128), dtype=np.complex128, queue=queue)

    # make some data to fft on the cpu
    data = (np.random.random((128,128,128,)) + 0J).astype(np.complex128)

    # send it to the gpu
    gpu_data = pyopencl.array.to_device(queue, data)

    # do 1000 forward and backward ffts
    for i in range(1000):
        print i
        plan.execute(gpu_data.data)
        plan.execute(gpu_data.data, inverse=True)
    
    # send the data back to the cpu
    result = gpu_data.get()
    error = np.abs(np.sum(np.abs(data) - np.abs(result)) / data.size)
    print error

def test_cpu_ffts():
    # make some data to fft on the cpu
    data = (np.random.random((128,128,128,)) + 0J).astype(np.complex128)

    result = data.copy()
    # do 1000 forward and backward ffts
    for i in range(1000):
        print i
        result = np.fft.fftn(result)
        result = np.fft.ifftn(result)

    # send the data back to the cpu
    error = np.abs(np.sum(np.abs(data) - np.abs(result)) / data.size)
    print error

if __name__ == "__main__":

    #test_gpu_ffts()
    #test_cpu_ffts()

    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    params = io_utils.parse_parameters(config)
    
    diff = read_data(params)
    
    if params['mask']['support_rad'] != 0 :
        i, j, k = np.indices(diff.shape)
        r       = (i-diff.shape[0]/2)**2 + (j-diff.shape[1]/2)**2 + (k-diff.shape[2]/2)**2 
        support = r < params['mask']['support_rad']**2
    else :
        support = fit_sphere_to_autoc(diff)

    # mask is "good", so good_diff_pixels = diff * mask
    i, j, k = np.indices(diff.shape)
    r       = (i-diff.shape[0]/2)**2 + (j-diff.shape[1]/2)**2 + (k-diff.shape[2]/2)**2 
    mask    = r > params['mask']['centre_rad']**2

    psi, support, support_error, mod_error = iterate(diff, support, mask, params)
