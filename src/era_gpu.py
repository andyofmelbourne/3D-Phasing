import numpy as np
import sys
from itertools import product
import era

import pyfft
import pyopencl
import pyopencl.array
from   pyfft.cl import Plan
import pyopencl.clmath 


def ERA_gpu(I, iters, support, mask = 1, O = None, background = None, \
        method = None, hardware = 'cpu', alpha = 1.0e-10, \
        dtype = 'single', queue = None, plan = None, full_output = True):
    """
    GPU variant of era.ERA
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
    
    O    = O.astype(c_dtype)
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    eMods     = []
    eCons     = []

    # set up the gpu
    #---------------
    #---------------
    if queue is None and plan is None :
        # get the CUDA platform
        print 'opencl platforms found:'
        platforms = pyopencl.get_platforms()
        for p in platforms:
            print '\t', p.name
            if p.name == 'NVIDIA CUDA':
                platform = p
                print '\tChoosing', p.name

        # get one of the gpu's device id
        print '\nopencl devices found:'
        devices = platform.get_devices()
        for d in devices:
            print '\t', d.name

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
    support_g = pyopencl.array.to_device(queue, np.ascontiguousarray(np.ones_like(O).astype(np.int8)))
    if type(support) is not int :
        support_g.set(support)
    if mask is not 1 :
        mask_g    = np.empty(I.shape, dtype=np.int8)
        mask_g[:] = mask.astype(np.int8)*2 - 1
        mask_g    = pyopencl.array.to_device(queue, np.ascontiguousarray(mask_g))
    else :
        mask_g  = 1

    O_sol = None
    # method 1
    #---------
    if method == 1 :
        if iters > 0 :
            print '\n\nalgrithm progress iteration convergence modulus error'
        for i in range(iters) :
            O0 = O_g.copy()
            
            # modulus projection 
            O = pmod_gpu(amp_g, O_g, plan, mask_g, alpha = alpha)
            
            O1 = O_g.copy()
            
            # support projection 
            if type(support) is int and (i % 1) == 0 :
                S = era.choose_N_highest_pixels( (O_g * O_g.conj()).real.get(), support)
                support_g.set(S.astype(np.int8))
            O_g = O_g * support_g
            
            # metrics
            O2 = O_g.copy()
            
            O2    -= O0
            tot    = pyopencl.array.sum((O0 * O0.conj()).real).get()
            eCon   = pyopencl.array.sum((O2 * O2.conj()).real).get()
            eCon   = np.sqrt(eCon / tot)
            
            O1    -= O0
            eMod   = pyopencl.array.sum((O1 * O1.conj()).real).get() / I_norm
            eMod   = np.sqrt(eMod)
            
            era.update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if O_sol is None :
            O_sol = O_g
        
        O = O_g.get()
        queue.flush()    
        queue.finish()    
        if full_output : 
            info = {}
            info['I']     = np.abs(np.fft.fftn(O))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['queue'] = queue
            info['plan']  = plan
            return O, info
        else :
            return O

def pmod_gpu(amp, O, plan, mask = 1, alpha = 1.0e-10):
    plan.execute(O.data)
    O = Pmod_gpu(amp, O, mask = mask, alpha = alpha)
    plan.execute(O.data, inverse = True)
    return O
    
def Pmod_gpu(amp, O, mask = 1, alpha = 1.0e-10):
    import pyopencl.array
    if mask is 1 :
        O  = O * amp / (abs(O) + alpha)
    else :
        #exits  = mask * exits * amp / (abs(exits) + alpha)
        O2 = O * amp / (abs(O) + alpha)
        pyopencl.array.if_positive(mask, O2, O, out = O)
        #exits.mul_add(mask * amp / (abs(exits) + alpha), (1 - mask), exits)
    return O

def choose_N_highest_pixels(array, N, sorter):
    # sort the array
    a  = np.random.random((10,))
    sgs, evt = sorter(a)
    print sgs.get()
    import sys
    sys.exit()
