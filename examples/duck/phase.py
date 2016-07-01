import numpy as np
import sys, os

import phasing_3d
import phasing_3d.utils as utils

def centre(array):
    # get the centre of mass of |P|^2
    import scipy.ndimage
    a  = (array.conj() * array).real
    cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int)# - np.array(a.shape)/2
    
    # centre array
    array = np.roll(array, -cm[0], 0)
    array = np.roll(array, -cm[1], 1)
    array = np.roll(array, -cm[2], 2)
    #array = era.multiroll(array, -cm)
    return array

def phase(I, support, params, good_pix = None, sample_known = None):
    if params['phasing']['support'] == 'highest_N':
        support = params['highest_N']['n']

    if params['phasing']['background'] is not None :
        background = True
    else :
        background = None
    
    # repeats
    xs    = []
    eMods = []
    eCons = []
    info  = {}
    info['plan'] = info['queue'] = None
    for j in range(params['phasing']['repeats']):
        if params['phasing']['repeats'] > 1 :
            print '\n\nLoop:', j
            print '----------'
            print '----------'
        
        x    = None
        eMod = []
        eCon = []
        
        # Error reduction algorithm
        #--------------------------
        if params['phasing']['era_init'] > 0 :
            x, info = phasing_3d.ERA(I, params['phasing']['era_init'], support, mask = good_pix, O = x, \
                      background = background, \
                      method = None, hardware = params['compute']['hardware'], alpha = 1.0e-10, \
                      dtype = 'double', plan = info['plan'], queue = info['queue'], full_output = True)
            eMod += info['eMod']
            eCon += info['eCon']
            if info.has_key('background'):
                background = info['background']

        for i in range(params['phasing']['outerloop']):
            
            # Difference Map
            #---------------
            if params['phasing']['dm'] > 0 :
                x, info = phasing_3d.DM(I, params['phasing']['dm'], support, mask = good_pix, O = x, \
                          background = background, \
                          method = None, hardware = params['compute']['hardware'], alpha = 1.0e-10, \
                          dtype = 'double', plan = info['plan'], queue = info['queue'], full_output = True)
                eMod += info['eMod']
                eCon += info['eCon']
                if info.has_key('background'):
                    background = info['background']
            
            # Error reduction algorithm
            #--------------------------
            if params['phasing']['era'] > 0 :
                x, info = phasing_3d.ERA(I, params['phasing']['era'], support, mask = good_pix, O = x, \
                          background = background, \
                          method = None, hardware = params['compute']['hardware'], alpha = 1.0e-10, \
                          dtype = 'double', plan = info['plan'], queue = info['queue'], full_output = True)
                eMod += info['eMod']
                eCon += info['eCon']
                if info.has_key('background'):
                    background = info['background']

        # Error reduction algorithm
        #--------------------------
        if params['phasing']['era_final'] > 0 :
            x, info = phasing_3d.ERA(I, params['phasing']['era_final'], support, mask = good_pix, O = x, \
                      background = background, \
                      method = None, hardware = params['compute']['hardware'], alpha = 1.0e-10, \
                      dtype = 'double', plan = info['plan'], queue = info['queue'], full_output = True)
            eMod += info['eMod']
            eCon += info['eCon']
            if info.has_key('background'):
                background = info['background']

        xs.append(centre(x))
        eMods.append(eMod)
        eCons.append(eCon)

    if params['phasing']['repeats'] > 1 :
        xs = np.array(xs)
        x, T, T_rav = merge_sols(xs)
        info['I'] = np.abs(np.fft.fftn(x))**2
        info['transmission'] = T
        info['transmission radial average'] = T_rav
        xs = [x]
    else :
        info['transmission'] = None
        info['transmission radial average'] = None

    if info.has_key('r_av'):
        r_av = info['r_av']
    else :
        r_av = None
    
    return np.array(xs), info['I'], eMods, eCons, info['transmission'], info['transmission radial average'], r_av


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = utils.io_utils.read_input_h5(args.input)
    
    samples_ret, diff_ret, emods, econs, T, T_rav, B_rav = phase(diff, support, params, \
                                good_pix = good_pix, sample_known = sample_known)
    
    # write the h5 file 
    utils.io_utils.write_output_h5(params['output']['path'], diff, diff_ret, support, \
                    support, good_pix, sample_known, samples_ret, emods, econs, None, T, T_rav, B_rav)
