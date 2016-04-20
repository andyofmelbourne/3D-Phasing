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

def DM_gpu(I, iters, support, mask = 1, O = None, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', full_output = True):
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
    
    O     = O.astype(c_dtype)
    O     = O * support
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    eMods     = []
    eCons     = []

    # set up the gpu
    #---------------
    #---------------
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

    O_g   = pyopencl.array.to_device(queue, np.ascontiguousarray(O))
    amp_g = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
    if type(support) is not int :
        support_g = pyopencl.array.to_device(queue, np.ascontiguousarray(support.astype(np.int8)))
    if mask is not 1 :
        mask_g    = np.empty(I.shape, dtype=np.int8)
        mask_g[:] = mask.astype(np.int8)*2 - 1
        mask_g    = pyopencl.array.to_device(queue, np.ascontiguousarray(mask_g))
    else :
        mask_g  = 1
    O_sol = O.copy()
    
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
            if type(support) is int :
                S = era.choose_N_highest_pixels( (O_g * O_g.conj()).real, support_g)
            else :
                S = support_g
            O0 = O_g * S
            
            O_g  -= O0
            O0   -= O_g
            O0    = era_gpu.pmod_gpu(amp_g, O0, plan, mask_g, alpha = alpha)
            O_g  += O0
            
            # metrics
            #--------
            O_bak -= O_g.copy()

            tot    = pyopencl.array.sum((O_g * O_g.conj()).real).get()
            eCon   = pyopencl.array.sum((O_bak * O_bak.conj()).real).get()
            eCon   = np.sqrt(eCon / tot)
            
            # f* = Ps f_i = PM (2 Ps f_i - f_i)
            O_sol = O_g * S
            eMod  = model_error_gpu(amp_g, O_sol, plan, mask_g, background = 0)
            eMod  = np.sqrt( eMod / I_norm )
            
            era.update_progress(i / max(1.0, float(iters-1)), 'DM', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            O = O_g.get()
            info = {}
            info['I']     = np.abs(np.fft.fftn(O))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O_sol.get(), info
        else :
            return O_sol.get()



def model_error_gpu(amp, O, plan, mask, background = 0):
    plan.execute(O.data)
    M   = pyopencl.clmath.sqrt((O.conj() * O).real + background**2)
    err = pyopencl.array.sum( mask * (M - amp)**2 ).get()
    return err
