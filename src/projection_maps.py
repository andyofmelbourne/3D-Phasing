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
    
    #psi_sup -= psi
    #mod_err  = np.sum( (psi_sup * psi_sup.conj()).real ) / float(psi.size)
    mod_err = calc_modulus_err(psi, support, good_pix, amp)
    return psi, mod_err


def DM(psi, support, good_pix, amp):
    temp = psi * (2 * support - 1)

    psi += Pmod(amp, temp, good_pix) - psi * support
    
    mod_err = calc_modulus_err(psi, support, good_pix, amp)
    return psi, mod_err

def calc_modulus_err(psi, support, good_pix, amp):
    dummy_comp  = psi.copy() * support
    dummy_comp  = np.fft.fftn(dummy_comp)
    dummy_real  = dummy_comp.__abs__() - amp 
    dummy_real *= good_pix
    mod_err     = np.sum( dummy_real * dummy_real) / float(psi.size)
    return np.sqrt(mod_err)
