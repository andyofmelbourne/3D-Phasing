import numpy as np
import sys


def parse_parameters(config):
    """
    Parse values from the configuration file and sets internal parameter accordingly
    The parameter dictionary is made available to both the workers and the master nodes

    The parser tries to interpret an entry in the configuration file as follows:

    - If the entry starts and ends with a single quote, it is interpreted as a string
    - If the entry is the word None, without quotes, then the entry is interpreted as NoneType
    - If the entry is the word False, without quotes, then the entry is interpreted as a boolean False
    - If the entry is the word True, without quotes, then the entry is interpreted as a boolean True
    - If non of the previous options match the content of the entry, the parser tries to interpret the entry in order as:

        - An integer number
        - A float number
        - A string

      The first choice that succeeds determines the entry type
    """

    monitor_params = {}

    for sect in config.sections():
        monitor_params[sect]={}
        for op in config.options(sect):
            monitor_params[sect][op] = config.get(sect, op)
            if monitor_params[sect][op].startswith("'") and monitor_params[sect][op].endswith("'"):
                monitor_params[sect][op] = monitor_params[sect][op][1:-1]
                continue
            if monitor_params[sect][op] == 'None':
                monitor_params[sect][op] = None
                continue
            if monitor_params[sect][op] == 'False':
                monitor_params[sect][op] = False
                continue
            if monitor_params[sect][op] == 'True':
                monitor_params[sect][op] = True
                continue
            try :
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    pass

    return monitor_params

def binary_out(array, fnam, endianness='little', appendType=True, appendDim=True):
    """Write a n-d array to a binary file."""
    arrayout = np.array(array)
    
    if appendDim == True :
        fnam_out = fnam + '_'
        for i in arrayout.shape[:-1] :
            fnam_out += str(i) + 'x' 
        fnam_out += str(arrayout.shape[-1]) + '_' + str(arrayout.dtype) + '.bin'
    else :
        fnam_out = fnam
    
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    
    arrayout.tofile(fnam_out)

def binary_in(fnam, ny = None, nx = None, dtype = None, endianness='little', dimFnam = True):
    """Read a n-d array from a binary file."""
    if dimFnam :
        # grab the dtype from the '_float64.bin' at the end
        tstr = fnam[:-4].split('_')[-1]
        if dtype is None :
            dtype = np.dtype(tstr)
        
        # get the dimensions from the 'asfasfs_89x12x123_' bit
        b    = fnam[:fnam.index(tstr)-1].split('_')[-1]
        dims = b.split('x')
        dims = np.array(dims, dtype=np.int)
        dims = tuple(dims)
        
        arrayout = np.fromfile(fnam, dtype=dtype).reshape( dims )
    
    else :
        arrayout = np.fromfile(fnam, dtype=dtype).reshape( (ny,nx) )
    
    if sys.byteorder != endianness:
        arrayout.byteswap(True)
    
    return arrayout
