#!/usr/bin/env python

"""
This script just runs the pipline:
    Load the config file
    Run the forward simulation / input script
    Run the phasing script 
"""

import time
import sys, os
import configparser
import subprocess

from phasing import utils


if __name__ == "__main__":
    # Load the config file
    #--------------------------------------
    args = utils.io_utils.parse_cmdline_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    params = utils.io_utils.parse_parameters(config)
    
    # make input 
    #--------------------------------------
    if 'input' in params and 'script' in params['input']:
        runstr = "python " + params['input']['script'] + ' ' + args.config
        print('\n',runstr)
        subprocess.call([runstr], shell=True)

    # forward problem
    #--------------------------------------
    if 'simulation' in params and params['simulation']['sample'] == 'duck':
        diff, beamstop, background_circle, edges, support, solid_unit = utils.duck.generate_diff(params)
        
        # write to file
        utils.io_utils.write_input_h5(params['output']['path'], diff, support, \
                beamstop * edges, solid_unit, args.config)
    
    if args.input :
        sys.exit()

    # inverse problem
    #--------------------------------------
    runstr = "python " + params['phasing']['script'] + ' ' + \
                     os.path.join(params['output']['path'],'input.h5')
    print('\n',runstr)
    subprocess.call([runstr], shell=True)
