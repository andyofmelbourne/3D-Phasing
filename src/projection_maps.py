import numpy as np
from scipy import ndimage

def Pmod(amp, psi, good_pix):
    psi           = np.fft.fftn(psi)
    phase         = np.angle(psi)
    #phase         = ndimage.gaussian_filter(phase, 0.5)
    psi           = psi * (1 - good_pix)
    psi           = psi + amp * np.exp(1J * phase) * good_pix 
    psi           = np.fft.ifftn(psi)
    return psi


def ERA(psi, support, good_pix, amp):
    psi_sup = psi * support
    psi     = Pmod(amp, psi_sup.copy(), good_pix) 
    
    delta = psi - psi_sup
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



