import numpy as np
import pyopencl.clmath

import pyfft
import pyopencl
import pyopencl.array
from   pyfft.cl import Plan
import pyopencl.clmath

def l2norm_gpu(array1, array2):
    """Calculate sqrt ( sum |array1 - array2|^2 / sum|array1|^2 )."""
    tot = pyopencl.array.sum(abs(array1)**2).get()
    return np.sqrt(pyopencl.array.sum(abs(array1-array2)**2).get()/tot)

class Proj():
    def __init__(self, shape, dtype):
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
        self.plan = Plan(shape, dtype=dtype, queue=self.queue)

    def send_to_gpu(self, psi, amp, support, good_pix):
        # send it to the gpu
        psi_gpu      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(psi))
        support_gpu  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(support.astype(np.int8)))
        amp_gpu      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(amp))
        good_pix_gpu = pyopencl.array.to_device(self.queue, np.ascontiguousarray(good_pix.astype(np.int8)))
        return psi_gpu, amp_gpu, support_gpu, good_pix_gpu

    def _ERA(self, psi, Pmod, Psup):
        psi = Pmod(Psup(psi))
        return psi

    def _HIO(self, psi, Pmod, Psup, beta):
        out = Pmod(psi)
        out = psi + beta * Psup( (1.+1./beta)*out - 1./beta * psi ) - beta * out  
        return out

    def _HIO_beta1(self, psi, Pmod, Psup):
        out = Pmod(psi)
        out = psi + Psup( 2.* out - psi ) - out  
        return out

    def _Pmod(self, psi, amp, good_pix, alpha = 1.0e-10):
        out  = good_pix * psi * amp / (abs(psi) + alpha)
        out += (1 - good_pix) * psi
        return out

    def Pmod(self, x, amp, good_pix):
        y = x.copy(queue = self.queue)
        self.plan.execute(y.data)
	
        y = self._Pmod(y, amp, good_pix)
	
        self.plan.execute(y.data, inverse=True)
        return y

    def Psup(self, x, support, real = True, pos = True):
        y = x.copy(queue = self.queue)
	
        # apply support
        y = x * support
        
        # apply reality
        if real :
            y = y.real.astype(x.dtype)
        
        #if real and pos :
        #    # apply positivity
        #    y[np.where(y<0)] = 0.0
        return y


