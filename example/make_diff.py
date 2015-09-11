
# import from parant directory
import os, sys, getopt, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir  = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utils import io_utils

import ConfigParser
import numpy as np

if __name__ == "__main__":
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    params = io_utils.parse_parameters(config)
    
    # make the real-space object
    psi = np.zeros((params['cube']['n'], params['cube']['n'], \
                    params['cube']['n']), np.complex128)
    
    # fill the bounding box
    s = [params['cube']['a'], params['cube']['b'], params['cube']['c']]
    psi[(psi.shape[0]-s[0])/2 : (psi.shape[0]+s[0])/2, \
        (psi.shape[1]-s[1])/2 : (psi.shape[1]+s[1])/2, \
        (psi.shape[2]-s[2])/2 : (psi.shape[2]+s[2])/2] = 1.0

    # real-space support
    support = psi.real > 0.
    
    # add noise
    i       = np.where(psi == 1.0) 
    psi[i] += np.random.random((len(i[0]), )) * params['cube']['noise']
    if params['cube']['real'] is False :
        psi[i] += np.random.random((len(i[0]), )) * params['cube']['noise'] * 1J 
    
    # make the diffraction data
    diff = np.fft.fftn(psi)
    diff = np.abs(diff)**2
    diff = np.fft.fftshift(diff)
    
    # output
    shapestr = str(params['cube']['n']) + 'x' + str(params['cube']['n']) + 'x' + str(params['cube']['n']) 
    diff.astype(np.float32).tofile('diff_example_'+shapestr+'_float32.bin')
    psi.astype(np.complex64).tofile('obj_example_'+shapestr+'_complex64.bin')
    support.tofile('support_example_'+shapestr+'_bool.bin')

