import numpy as np
import sys, os

sys.path.append(os.path.abspath('.'))
from utils import io_utils
from utils.l2norm import l2norm
from utils.progress_bar import update_progress
from utils.support import shrinkwrap
from src import projection_maps as pm


def _ERA(psi, Pmod, Psup):
    psi = Pmod(Psup(psi))
    return psi

def _HIO(psi, Pmod, Psup, beta):
    out = Pmod(psi)
    out = psi + beta * Psup( (1.+1./beta)*out - 1./beta * psi ) - beta * out  
    return out

def _Pmod(psi, amp, good_pix, alpha = 1.0e-10):
    out  = good_pix * psi * amp / (np.abs(psi) + alpha)
    out += ~good_pix * psi
    return out


def phase(I, support, params, good_pix = None, sample_known = None):
    amp = np.sqrt(I)
    
    if good_pix is None :
        good_pix = I > -1
    good_pix = good_pix.astype(np.bool)
    support  = support.astype(np.bool)
    
    def Pmod(x):
        y = np.fft.fftn(x)
        y = _Pmod(y, amp, good_pix)
        y = np.fft.ifftn(y)
        return y

    def Psup(x):
        # apply support
        y = x * support
        
        # apply reality
        y.imag = 0.0
        
        # apply positivity
        y[np.where(y<0)] = 0.0
        return y

    # initial support
    autoc   = np.fft.ifftn(I)
    support = shrinkwrap(autoc, 32*32*32, 24 * 28 * 32, 10, 0)
    
    x = np.random.random(support.shape) + np.random.random(support.shape) * 1J
    x = Psup(x)
    
    ERA = lambda x : _ERA(x, Pmod, Psup)
    HIO = lambda x : _HIO(x, Pmod, Psup, params['phasing']['beta'])

    emod = []
    efid = []
    index     = 0
    index_max = params['phasing']['outerloop'] * (params['phasing']['hio'] + params['phasing']['era'])
    print params['phasing']['outerloop'] , params['phasing']['hio'] , params['phasing']['era']

    for i in range(params['phasing']['outerloop']):
        for j in range(params['phasing']['hio']):
            x = HIO(x)
            
            emod.append( l2norm(x, Pmod(Psup(x))) )
            if sample_known is not None :
                efid.append( l2norm(x, sample_known) )
            update_progress(index / max(1.0, float(index_max-1)), 'HIO', index, emod[-1], efid[-1])
            index += 1
        
        for k in range(params['phasing']['era']):
            x = ERA(x)
            
            emod.append( l2norm(x, Pmod(Psup(x))) )
            if sample_known is not None :
                efid.append( l2norm(x, sample_known) )
            update_progress(index / max(1.0, float(index_max-1)), 'ERA', index, emod[-1], efid[-1])
            index += 1

        support = shrinkwrap(x, 32*32*32, 24 * 28 * 32, 10, i)
        print '\n\nshrinking...', np.sum(support), 24 * 28 * 32

    M = np.abs(np.fft.fftn(Psup(x)))**2
    return x, M, emod, efid

if __name__ == "__main__":
    args = io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = io_utils.read_input_h5(args.input)
    
    sample_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, sample_known = sample_known)
    
    # write the h5 file 
    io_utils.write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, sample_known, sample_ret, emod, efid)
