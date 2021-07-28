import numpy as np
import reikna.cluda as cluda 
import reikna.fft
import pyopencl as cl

from .mappers import *

## Step #1. Obtain an OpenCL platform.
# with a cpu device
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.CPU)
    if len(devices) > 0:
        platform = p
        device   = devices[0]
        break

## Step #3. Create a context for the selected device.
context = cl.Context([device])
queue   = cl.CommandQueue(context)

## Step #4. Initialise a reikna thread with the opencl queue
api = cluda.ocl_api()
thr = api.Thread(queue)

class Mapper():
    
    def __init__(self, I, **args):
        modes = Modes()
        
        # check if there is a background
        if isValid('background', args):
            if args['background'] is True :
                B = np.random.random((I.shape)).astype(args['dtype'])
            else :
                B = np.sqrt(args['background']).astype(args['dtype'])
            modes['B'] = cl.array.to_device(queue, B)
        
        if isValid('O', args):
            modes['O'] = cl.array.to_device(queue, args['O'])
        else :
            modes['O'] = cl.array.to_device(queue, np.random.random(I.shape).astype(args['c_dtype']))
        
        # this is the radial value for every pixel 
        # in the volume
        self.rs    = None 
        
        if isValid('mask', args):
            self.mask = cl.array.to_device(queue, args['mask'])
            self.I_norm = (self.mask.get() * I).sum()
        else :
            self.mask = 1
            self.I_norm = I.sum()
        
        self.alpha = 1.0e-10
        if isValid('alpha', args):
            self.alpha = args['alpha']

        self.amp = self.mask * cl.array.to_device(queue, np.sqrt(I.astype(args['dtype'])))

        # define the data projection
        # --------------------------
        if 'B' in modes.keys() :
            self.Pmod = self.Pmod_back
        else :
            self.Pmod = self.Pmod_single
    
        # define the support projection
        # -----------------------------
        if isValid('voxel_number', args) :
            self.voxel_number = args['voxel_number']
        else :
            self.voxel_number = False
            self.S    = cl.array.to_device(queue, args['support'].astype(np.int8))
        
        support = np.ones(I.shape, dtype=np.int8)
        if isValid('support', args):
            support = args['support']
        self.support = cl.array.to_device(queue, support.astype(np.int8))
        
        self.modes = modes

        # compile fft routine for the object
        # ----------------------------------
        fft = reikna.fft.FFT(modes['O'])
        self.cfft = fft.compile(thr)
    
    def object(self, modes):
        return modes['O'].get()

    def Psup(self, modes):
        modes = modes.copy()
        
        if self.voxel_number :
            O = modes['O']
            self.S = choose_N_highest_pixels( self.support * (O * O.conj()).real, self.voxel_number)
        
        modes['O'] *= self.S
        
        if 'B' in modes.keys() :
            B, self.rs, self.r_av = radial_symetry(modes['B'].get(), rs = self.rs)
            modes['B'] = cl.array.to_device(queue, B)
         
        return modes

    def Pmod_single(self, modes):
        #out = modes.copy()
        self.cfft(modes['O'], modes['O'])
        modes['O'] *= self.amp / (abs(modes['O']) + self.alpha)
        modes['O'] += (1 - self.mask) * modes['O']
        self.cfft(modes['O'], modes['O'], 1)
        return modes
    
    def Pmod_back(self, modes):
        out = modes.copy()
        self.cfft(out['O'], out['O'])
        M = self.amp / ((out['O'].conj() * out['O']).real + out['B']**2 + self.alpha)**0.5
        out['O'] *= M
        out['B'] *= M
        out['O'] += (1 - self.mask) * out['O']
        self.cfft(out['O'], out['O'], 1)
        return out

    def Imap(self, modes):
        O = cl.array.empty_like(modes['O'])
        self.cfft(O, modes['O'])
        if 'B' in modes.keys() :
            I = (O.conj() * O).real + modes['B']**2
        else :
            I = (O.conj() * O).real 
        return I
    
    def Emod(self, modes):
        M         = self.Imap(modes)
        eMod      = cl.array.sum( self.mask * ( M**0.5 - self.amp )**2 ).get()[()]
        eMod      = np.sqrt( eMod / self.I_norm )
        return eMod

    def finish(self, modes):
        out = {}
        out['support'] = self.S.get()
        out['I']       = self.Imap(modes).get()

        if 'B' in modes.keys() :
            out['background'] = modes['B'].get()**2
            out['r_av']       = self.r_av
        return out

    def l2norm(self, delta, array0):
        num = 0.
        den = 0.
        for k in delta.keys():
            num += cl.array.sum( (delta[k] * delta[k].conj()).real ).get()[()]
            den += cl.array.sum( (array0[k] * array0[k].conj()).real ).get()[()] 
        return np.sqrt(num / den)

def choose_N_highest_pixels(array, N, tol = 1.0e-5, maxIters=1000, support=None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N

    then return (array_i > x) a boolean mask

    This is faster than using percentile (surprising)

    If support is not None then values outside the support
    are ignored. 
    """
    s0 = cl.array.max(array).get()[()]
    s1 = cl.array.min(array).get()[()]
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = cl.array.sum( (array > s), dtype=np.dtype(np.int64)).get()[()] - N
          
        if np.abs(e) < tol :
            break

        if e < 0 :
            s0 = s
        else :
            s1 = s
        
    S = (array > s) 
    #print 'number of pixels in support:', np.sum(support), i, s, e
    return S

