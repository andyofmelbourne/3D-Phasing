#!/usr/bin/env python

import time
import sys, os
import ConfigParser
import subprocess

sys.path.append('.')
from utils import io_utils
from utils import duck


if __name__ == "__main__":
    args = io_utils.parse_cmdline_args()
    
    config = ConfigParser.ConfigParser()
    config.read(args.config)
    
    params = io_utils.parse_parameters(config)
    
    # make input 
    if params.has_key('input') and  params['input'].has_key('script'):
        runstr = "python " + params['input']['script'] + ' ' + args.config
        print '\n',runstr
        subprocess.call([runstr], shell=True)

    # forward problem
    if params.has_key('simulation') and params['simulation']['sample'] == 'duck':
        diff, beamstop, background_circle, edges, support, solid_unit = duck.generate_diff(params)
        
        # write to file
        io_utils.write_input_h5(params['output']['path'], diff, support, \
                beamstop * edges, solid_unit, args.config)
    
    # inverse problem
    runstr = "python " + params['phasing']['script'] + ' ' + \
                     os.path.join(params['output']['path'],'input.h5')
    print '\n',runstr
    subprocess.call([runstr], shell=True)
