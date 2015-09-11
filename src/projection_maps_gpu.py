import numpy as np
import pyopencl.clmath

import pyfft
import pyopencl
import pyopencl.array
from   pyfft.cl import Plan
import pyopencl.clmath

class Projections():
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
        print psi.dtype, amp.dtype, np.zeros_like(psi).dtype
        print float(3*psi.nbytes + 2*support.astype(np.int8).nbytes + 2*amp.nbytes) / 1024**3
        # send it to the gpu
        psi_gpu      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(psi))
        support_gpu  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(support.astype(np.int8)))
        amp_gpu      = pyopencl.array.to_device(self.queue, np.ascontiguousarray(amp))
        good_pix_gpu = pyopencl.array.to_device(self.queue, np.ascontiguousarray(good_pix.astype(np.int8)))

        # send dummy arrays to the gpu
        self.dummy_real  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(amp)))
        self.dummy_comp  = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(psi)))
        self.dummy_comp2 = pyopencl.array.to_device(self.queue, np.ascontiguousarray(np.zeros_like(psi)))
        return psi_gpu, amp_gpu, support_gpu, good_pix_gpu

    def Pmod(self, amp, psi, good_pix):
        self.plan.execute(psi.data)

        self.dummy_real = pyopencl.clmath.atan2(psi.imag, psi.real, queue=self.queue)
        
        psi   *= (1 - good_pix)

        psi   = psi + amp * pyopencl.clmath.exp(1J * self.dummy_real, queue=self.queue) * good_pix
        
        self.plan.execute(psi.data, inverse=True)
        return psi
    
    def ERA(self, psi, support, good_pix, amp):
        psi = psi * support
        psi = self.Pmod(amp, psi, good_pix) 
        
        mod_err = self.calc_modulus_err(psi, support, good_pix, amp)
        return psi, mod_err
    
    def DM(self, psi, support, good_pix, amp, beta):
        if beta != 1 :
            print '\n warning! this routine is only designed for beta=1'
            print ' use DM_beta instead'

        #print pyopencl.array.sum( psi.__abs__() )
        
        self.dummy_comp = psi * (2 * support - 1)
        
        psi += self.Pmod(amp, self.dummy_comp, good_pix) - psi * support
        
        mod_err = self.calc_modulus_err(psi, support, good_pix, amp)
        return psi, mod_err

    def DM_beta(self, psi, support, good_pix, amp, beta):
        """
        psi_j+1 = psi_j - Ps psi_j - Pm psi_j
                + b(1+1/b) Ps Pm psi_j
                - b(1-1/b) Pm Ps psi_j
        """
        self.dummy_comp  = psi.copy(queue = self.queue)
        self.dummy_comp  = self.Pmod(amp, self.dummy_comp, good_pix)
        self.dummy_comp2 = support * psi 
        psi  -= self.dummy_comp + self.dummy_comp2
        psi  += np.float32(beta * (1. + 1. / beta)) * support * self.dummy_comp
        self.dummy_comp2 = self.Pmod(amp, self.dummy_comp2, good_pix)
        psi  -= np.float32(beta * (1. - 1. / beta)) * self.dummy_comp2

        self.dummy_comp = self.DM_to_sol(psi, support, good_pix, amp, beta)
        mod_err = self.calc_modulus_err(self.dummy_comp, support, good_pix, amp)
        return psi, mod_err

    def DM_to_sol(self, psi, support, good_pix, amp, beta):
        self.dummy_comp = psi.copy(queue = self.queue)
        self.dummy_comp = self.Pmod(amp, self.dummy_comp, good_pix)
        self.dummy_comp = (1. + 1./beta) * self.dummy_comp - 1./beta * psi
        self.dummy_comp = self.dummy_comp * support
        return self.dummy_comp
    
    def calc_modulus_err(self, psi, support, good_pix, amp):
        self.dummy_comp  = psi.copy(queue = self.queue) 
        self.dummy_comp *= support
        self.plan.execute(self.dummy_comp.data) 
        self.dummy_real  = self.dummy_comp.__abs__() - amp 
        self.dummy_real *= good_pix
        mod_err = pyopencl.array.sum( self.dummy_real * self.dummy_real, queue = self.queue ) / float(psi.size)
        return np.sqrt(mod_err.get())
