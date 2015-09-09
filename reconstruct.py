#!/usr/bin/env python

import numpy as np
import time
import sys
import ConfigParser
from utils import io_utils


def read_data(params):
    diff = np.fromfile(params['input']['filename'], dtype = np.dtype(params['input']['dtype']))
    diff = diff.reshape((params['input']['i'], params['input']['j'], params['input']['k']))
    
    support = np.fromfile(params['mask']['support'], dtype = np.dtype(params['mask']['support_dtype']))
    support = support.reshape((params['input']['i'], params['input']['j'], params['input']['k']))
    
    if params['mask']['good_pix'] == True :
        good_pix = np.ones_like(support, dtype=np.bool)
    else :
        good_pix = np.fromfile(params['mask']['good_pix'], dtype = np.dtype(params['mask']['good_pix_dtype']))
        good_pix = good_pix.reshape((params['input']['i'], params['input']['j'], params['input']['k']))
    
    return diff, support, good_pix


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


def iterate(diff, support, mask, params):
    # shift quadrants for faster iters
    good_pix = np.fft.ifftshift(mask)
    amp      = np.sqrt(np.fft.ifftshift(diff))
    support  = np.fft.ifftshift(support)
    
    # initial guess
    print '\n inital estimate: random numbers b/w 0 and 1 (just real)'
    psi  = np.random.random(amp.shape) + 0J 
    psi *= support
    
    alg   = params['recon']['alg'].split()
    iters = np.array(alg[::2], dtype=np.int)
    algs  = alg[1::2]

    if params['recon']['gpu'] :
        import src.projection_maps_gpu as pm_gpu
        import pyopencl.array
        proj = pm_gpu.Projections(psi.shape, psi.dtype)
        psi, amp, support, good_pix = proj.send_to_gpu(psi, amp, support, good_pix)
        print pyopencl.array.sum( psi.__abs__() )
        
        ERA = proj.ERA
        DM  = proj.DM
    else :
        import src.projection_maps as pm
        ERA = pm.ERA
        DM  = pm.DM
    
    mod_error = []
    i = 0
    for it, alg in zip(iters, algs):
        for j in range(it):
            if alg == 'DM' :
                psi, mod_err = DM(psi, support, good_pix, amp)
                
            elif alg == 'ERA':
                psi, mod_err = ERA(psi, support, good_pix, amp)

            elif alg == 'shrink':
                print '\n performing shrink wrap:'
                if params['recon']['gpu'] :
                    shrink_mask = shrink(psi.get() * support.get(), thresh=params['shrink']['thresh'], blur=params['shrink']['blur'])
                    print ' cut', np.sum(support.get()) - np.sum(shrink_mask), 'pixels'
                    support = shrink_mask.copy()
                    support = pyopencl.array.to_device(proj.queue, np.ascontiguousarray(support.astype(np.int8)))
                else :
                    shrink_mask = shrink(psi * support, thresh=params['shrink']['thresh'], blur=params['shrink']['blur'])
                    print ' cut', np.sum(support) - np.sum(shrink_mask), 'pixels'
                    support = shrink_mask.copy()
                
            mod_error.append(mod_err)
            print alg, i, mod_error[-1]
            i += 1

            # output files for viewing if selected
            if params['output']['every'] is not False :
                if i % params['output']['every'] == 0 :
                    if params['recon']['gpu'] :
                        tpsi = psi.get()
                        tsupport = support.get()
                    
                    # un-shift quadrants
                    tsupport  = np.fft.fftshift(tsupport)
                    tpsi      = np.fft.fftshift(tpsi)
                    
                    # output
                    dir = params['output']['dir']
                    io_utils.binary_out(tpsi, dir + 'psi_'+str(i))
                    io_utils.binary_out(tsupport, dir + 'support_'+str(i))
                    io_utils.binary_out(mod_error, dir + 'mod_err_'+str(i))

    if params['recon']['gpu'] :
        psi = psi.get()
        support = support.get()
    
    # un-shift quadrants
    support  = np.fft.fftshift(support)
    psi      = np.fft.fftshift(psi)
     
    return psi, support, mod_error
        

def truncate(diff, n):
    c = np.array(diff.shape)/2
    diff_out = diff[c[0]-n:c[0]+n, c[1]-n:c[1]+n, c[2]-n:c[2]+n].copy()
    return diff_out
        

if __name__ == "__main__":
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    params = io_utils.parse_parameters(config)
    
    diff, support, good_pix = read_data(params)
    
    psi, support, mod_error = iterate(diff, support, good_pix, params)
    
    """
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
    """
