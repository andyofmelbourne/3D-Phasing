import numpy as np
import sys, os

sys.path.append(os.path.abspath('.'))
from utils import io_utils
from utils.l2norm import l2norm
from utils.progress_bar import update_progress
from utils.support import shrinkwrap
from src import projection_maps as pm


def phase(I, support, params, good_pix = None, sample_known = None):
    amp = np.sqrt(I)
    
    if good_pix is None :
        good_pix = I > -1
    good_pix = good_pix.astype(np.bool)
    support  = support.astype(np.bool)

    # define the projections
    #-----------------------
    projs = pm.Proj()
    Pmod = lambda x: projs.Pmod(x, amp, good_pix)
    Psup = lambda x : projs.Psup(x, support, real = True, pos = True)
    
    ERA = lambda x : projs._ERA(x, Pmod, Psup)
    if params['phasing']['beta'] == 1 :
        HIO = lambda x : projs._HIO_beta1(x, Pmod, Psup)
    else :
        HIO = lambda x : projs._HIO_beta1(x, Pmod, Psup, params['phasing']['beta'])

    # initial support
    #----------------
    autoc   = np.fft.ifftn(I)
    support = shrinkwrap(autoc, params['shrinkwrap']['start_pix'], params['shrinkwrap']['stop_pix'], params['shrinkwrap']['steps'], 0)
    
    # initial guess
    #--------------
    x = np.random.random(support.shape) + np.random.random(support.shape) * 1J
    x = Psup(x)


    emod      = []
    index     = 0
    index_max = params['phasing']['outerloop'] * (params['phasing']['hio'] + params['phasing']['era']) + params['phasing']['era_final']
    print params['phasing']['outerloop'] , params['phasing']['hio'] , params['phasing']['era']

    # track emod
    #-----------
    def errs(emod, index, index_max):
        emod.append( l2norm(x, Pmod(Psup(x))) )
        update_progress(index / max(1.0, float(index_max-1)), 'HIO', index, emod[-1], -1)
        index += 1
        return emod, index

    for i in range(params['phasing']['outerloop']):
        for j in range(params['phasing']['hio']):
            x = HIO(x)
            emod, index = errs(emod, index, index_max)
            
        for k in range(params['phasing']['era']):
            x = ERA(x)
            emod, index = errs(emod, index, index_max)

        support = shrinkwrap(x, params['shrinkwrap']['start_pix'], params['shrinkwrap']['stop_pix'], params['shrinkwrap']['steps'], i)
        print '\n\nshrinking...', np.sum(support), params['shrinkwrap']['stop_pix']

    # final ERA iterations
    i += 1
    for k in range(params['phasing']['era_final']):
        x = ERA(x)
        emod, index = errs(emod, index, index_max)
        
    support = shrinkwrap(x, params['shrinkwrap']['start_pix'], params['shrinkwrap']['stop_pix'], params['shrinkwrap']['steps'], i)
    print '\n\nshrinking...', np.sum(support), params['shrinkwrap']['stop_pix']

    M = np.abs(np.fft.fftn(Psup(x)))**2
    return x, M, emod, None


if __name__ == "__main__":
    args = io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = io_utils.read_input_h5(args.input)
    
    sample_ret, diff_ret, emod, efid = phase(diff, support, params, \
                                good_pix = good_pix, sample_known = sample_known)
    
    # write the h5 file 
    io_utils.write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, sample_known, sample_ret, emod, efid)
