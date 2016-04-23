import numpy as np
import sys
from itertools import product
import era
import era_gpu
import dm

import pyfft
import pyopencl
import pyopencl.array
import pyopencl.clmath 
from   pyfft.cl import Plan

def DM_gpu(I, iters, support, mask = 1, O = None, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', queue = None, plan = None, full_output = True):
    """
    GPU variant of dm.DM
    """
    
    method = 1
    
    if dtype is None :
        dtype   = I.dtype
        c_dtype = (I[0,0,0] + 1J * I[0, 0, 0]).dtype
    
    elif dtype == 'single':
        dtype   = np.float32
        c_dtype = np.complex64

    elif dtype == 'double':
        dtype   = np.float64
        c_dtype = np.complex128


    if O is None :
        O = np.random.random((I.shape)).astype(c_dtype)

    I_norm  = np.sum(mask * I)
    amp     = np.sqrt(I).astype(dtype)
    O       = O.astype(c_dtype)
    
    eMods     = []
    eCons     = []

    # set up the gpu
    #---------------
    #---------------
    if queue is None and plan is None :
        # get the CUDA platform
        #print 'opencl platforms found:'
        platforms = pyopencl.get_platforms()
        for p in platforms:
            #print '\t', p.name
            if p.name == 'NVIDIA CUDA':
                platform = p
                #print '\tChoosing', p.name

        # get one of the gpu's device id
        #print '\nopencl devices found:'
        devices = platform.get_devices()
        #for d in devices:
        #    print '\t', d.name

        #print '\tChoosing', devices[0].name
        device = devices[0]
        
        # create a context for the device
        context = pyopencl.Context([device])
        
        # create a command queue for the device
        queue = pyopencl.CommandQueue(context)
        
        # make a plan for the ffts
        plan = Plan(I.shape, dtype=c_dtype, queue=queue)

    O_g       = pyopencl.array.to_device(queue, np.ascontiguousarray(O))
    O0        = O_g.copy()
    amp_g     = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
    support_g = pyopencl.array.to_device(queue, np.ascontiguousarray(np.ones_like(O).astype(np.int8)))
    if type(support) is not int :
        support_g.set(support)
    
    if mask is not 1 :
        mask_g     = pyopencl.array.to_device(queue, np.ascontiguousarray(mask.astype(np.int8)))
        mask2_g    = mask.astype(np.int8)*2 - 1
        mask2_g    = pyopencl.array.to_device(queue, np.ascontiguousarray(mask2_g))
    else :
        mask_g  = 1
        mask2_g = 1
    
    # method 1
    #---------
    if method == 1 :
        if iters > 0 :
            print '\n\nalgrithm progress iteration convergence modulus error'
        for i in range(iters) :
            
            # reference
            O_bak = O_g.copy()
            
            # update 
            #-------
            # support projection 
            if type(support) is int and (i % 1) == 0 :
                S = era.choose_N_highest_pixels( (O_g * O_g.conj()).real.get(), support)
                support_g.set(S.astype(np.int8))
            
            O0 = O_g * support_g
            
            O_g  -= O0
            O0   -= O_g
            O0    = era_gpu.pmod_gpu(amp_g, O0, plan, mask2_g, alpha = alpha)
            O_g  += O0
            
            # metrics
            #--------
            O_bak = O_bak - O_g

            tot    = pyopencl.array.sum((O_g * O_g.conj()).real).get()
            eCon   = pyopencl.array.sum((O_bak * O_bak.conj()).real).get()
            eCon   = np.sqrt(eCon / tot)
            
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            O0    = O_g * support_g
            eMod  = model_error_gpu(amp_g, O0, plan, mask_g, background = 0)
            eMod  = np.sqrt( eMod / I_norm )
            
            era.update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        O0   = O_g * support_g
        O    = O0.get()
        queue.flush()    
        queue.finish()    
        if full_output : 
            info = {}
            info['I']     = np.abs(np.fft.fftn(O))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O, info
        else :
            return O


def model_error_gpu(amp, O, plan, mask, background = 0):
    plan.execute(O.data)
    M   = pyopencl.clmath.sqrt((O.conj() * O).real + background**2)
    M   = (M - amp)**2
    M   = M * mask
    #pyopencl.array.if_positive(mask, M, 0, out = M)
    err = pyopencl.array.sum( M ).get()
    return err
