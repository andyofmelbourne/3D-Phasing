import numpy as np
from scipy import ndimage
import pyopencl.clmath

import pyfft
import pyopencl
import pyopencl.array
from   pyfft.cl import Plan
import pyopencl.clmath

class Projections():
    def __init__(self, psi, amp, support, good_pix):
        """ 
        send input numpy arrays to the gpu.
        store needed dummy arrays on the gpu.
        """
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
        self.queue = pyopencl.CommandQueue(context)
        
        # make a plan for the ffts
        self.plan = Plan(psi.shape, dtype=psi.dtype, queue=queue)
        
        # send it to the gpu
        psi_gpu      = pyopencl.array.to_device(queue, np.ascontiguousarray(psi))
        support_gpu  = pyopencl.array.to_device(queue, np.ascontiguousarray(support))
        amp_gpu      = pyopencl.array.to_device(queue, np.ascontiguousarray(amp))
        good_pix_gpu = pyopencl.array.to_device(queue, np.ascontiguousarray(good_pix))

        # send dummy arrays to the gpu
        self.dummy_real = pyopencl.array.to_device(queue, np.ascontiguousarray(np.zeros_like(amp)))
        self.dummy_comp = pyopencl.array.to_device(queue, np.ascontiguousarray(np.zeros_like(psi)))
        return psi_gpu, amp_gpu, support_gpu, good_pix_gpu

    def Pmod(self, amp, psi, good_pix):
        self.plan.execute(psi.data)
        
        self.dummy_real = pyopencl.clmath.atan2(psi.imag, psi.real, queue=self.queue)
        
        psi   = psi * (1. - good_pix)
        psi   = psi + amp * pyopencl.clmath.exp(1J * phase, queue=self.queue) * good_pix
        
        self.plan.execute(psi.data, inverse=True)
        return psi


    def ERA(psi, support, good_pix, amp):
        psi = psi * support
        self.dummy_comp = psi.copy(queue = self.queue)
        psi = Pmod(amp, psi, good_pix) 
        
        self.dummy_comp = psi - self.dummy_comp
        support_err = np.sum( (delta * np.conj(delta)).real )
        
        delta = psi - psi_sup
        mod_err = np.sum( (delta * np.conj(delta)).real )

        return psi, mod_err, support_err


def DM(psi, support, good_pix, amp):
    psi_sup = psi * support

    psi += Pmod(amp, 2*psi_sup - psi, good_pix) - psi_sup
    
    delta       = psi * ~support
    support_err = np.sum( (delta * np.conj(delta)).real ) 
    
    delta       = psi - Pmod(amp, psi, good_pix)
    mod_err     = np.sum( (delta * np.conj(delta)).real ) 
    return psi, mod_err, support_err
