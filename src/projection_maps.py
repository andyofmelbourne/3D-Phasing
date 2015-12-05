import numpy as np
from scipy import ndimage

class Proj():
    
    def __init__(self):
        pass

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
        out  = good_pix * psi * amp / (np.abs(psi) + alpha)
        out += ~good_pix * psi
        return out

    def Pmod(self, x, amp, good_pix):
        y = np.fft.fftn(x)
        y = self._Pmod(y, amp, good_pix)
        y = np.fft.ifftn(y)
        return y

    def Psup(self, x, support, real = True, pos = True):
        # apply support
        y = x * support
        
        # apply reality
        if real :
            y.imag = 0.0
        
        if real and pos :
            # apply positivity
            y[np.where(y<0)] = 0.0
        return y
