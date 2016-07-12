import numpy as np
import sys, os
import re
import copy

import phasing_3d
import phasing_3d.utils as utils

def config_iters_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def phase(I, support, params, good_pix = None, sample_known = None):
    d   = {'eMod' : [],         \
           'eCon' : [],         \
           'O'    : None,       \
           'background' : None, \
           'B_rav' : None, \
           'support' : None     \
            }
    out = []

    params['phasing_parameters']['O'] = None
    
    params['phasing_parameters']['mask'] = good_pix
    
    if params['phasing_parameters']['support'] is None :
        params['phasing_parameters']['support'] = support

    # Repeats
    #---------------------------------------------
    for j in range(params['phasing']['repeats']):
        out.append(copy.copy(d))
        
        alg_iters = config_iters_to_alg_num(params['phasing']['iters'])
        
        for alg, iters in alg_iters :

            if alg == 'ERA':
               O, info = phasing_3d.ERA(I, iters, **params['phasing_parameters'])
        
            if alg == 'DM':
               O, info = phasing_3d.DM(I,  iters, **params['phasing_parameters'])

            out[j]['O']           = params['phasing_parameters']['O']          = O
            out[j]['support']     = params['phasing_parameters']['support']    = info['support']
            out[j]['eMod']       += info['eMod']
            out[j]['eCon']       += info['eCon']
            
            if 'background' in info.keys():
                print 'background:'
                out[j]['background']  = params['phasing_parameters']['background'] = info['background']
                out[j]['B_rav']       = info['r_av']
    
    out_merge = out[0]
    out_merge['I']    = info['I']

    return out_merge


if __name__ == "__main__":
    args = utils.io_utils.parse_cmdline_args_phasing()
    
    # read the h5 file
    diff, support, good_pix, sample_known, params = utils.io_utils.read_input_h5(args.input)
    
    out = phase(diff, support, params, \
                        good_pix = good_pix, sample_known = sample_known)
    
    # write the h5 file 
    utils.io_utils.write_output_h5(params['output']['path'], diff, out['I'], support, out['support'], \
                                  good_pix, sample_known, out['O'], out['eMod'], out['eCon'], None, None, None, out['B_rav'])
