import numpy as np
import scipy.ndimage

def overlap_cm(psi1, psi2):
    cm1 = scipy.ndimage.measurements.center_of_mass(psi1)
    cm2 = scipy.ndimage.measurements.center_of_mass(psi2)
    psi_out = psi2.copy()
    for i in range(len(cm1)):
        psi_out = np.roll(psi_out, -int(np.rint(cm2[i].real - cm1[i].real)), i)
    return psi_out


def merge_recons(psis, return_all=False):
    print '\n Merging', len(psis), 'reconstructions:'
    psi_out = psis[0].copy()
    I0      = np.abs(np.fft.fftn(psi_out))
    if return_all:
        psis_out = []

    def error(a, b):
        It = np.abs(np.fft.fftn(0.5 * (a + b)))
        e  = np.sum((I0 - It)**2)
	return e
    
    for i in range(1, len(psis), 1):
        print '\n overlaping recon:', i
        # unflipped
	psi0   = overlap_cm(psis[0], psis[i])
	e0    = error(psis[0], psi0)
	
        # negative?
	e0_m  = error(psis[0], -psi0)
	
        # flipped?
	psi1  = overlap_cm(psis[0], psis[i][::-1, ::-1, ::-1])
	e1    = error(psis[0], psi1[::-1, ::-1, ::-1])
	
        # negative?
	e1_m  = error(psis[0], -psi1)
	
	print '\n errors unflipped, neg, flipped, neg:', e0, e0_m, e1, e1_m
	es = np.array([e0, e0_m, e1, e1_m])
	if e0 == es.min():
	    print '\n Unflipped and positive'
	    psi = psi0
	elif e0_m == es.min():
	    print '\n Unflipped and negative'
	    psi = -psi0
	elif e1 == es.min():
	    print '\n flipped and positive'
	    psi = psi1
	elif e1_m == es.min():
	    print '\n flipped and negative'
	    psi = -psi1
	
        if return_all:
            psis_out.append(psi.copy())
	
        psi_out += psi
    print '\n renormalising'
    psi_out = psi_out/float(len(psis))
    if return_all:
        return psis_out
    else :
        return psi_out


