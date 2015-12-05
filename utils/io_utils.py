import numpy as np
import sys

def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'reconstruct.py', description='phase a merged 3D diffraction volume')
    parser.add_argument('config', type=str, \
                        help="file name of the configuration file")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.config):
        raise NameError('config file does not exist: ' + args.config)
    return args


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
            try:
                monitor_params[sect][op] = int(monitor_params[sect][op])
                continue
            except :
                try :
                    monitor_params[sect][op] = float(monitor_params[sect][op])
                    continue
                except :
                    # attempt to pass as an array of ints e.g. '1, 2, 3'
                    try :
                        l = monitor_params[sect][op].split(',')
                        monitor_params[sect][op] = np.array(l, dtype=np.int)
                        continue
                    except :
                        pass

    return monitor_params


def parse_cmdline_args_phasing():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'phase.py', description='phase a merged 3D diffraction volume')
    parser.add_argument('input', type=str, \
                        help="h5 file name of the input file")
    args = parser.parse_args()
    return args


def write_output_h5(path, diff, diff_ret, support, support_ret, \
        good_pix, solid_unit, solid_unit_ret, emod, efid):
    import os, h5py
    fnam = os.path.join(path, 'output.h5')
    if_exists_del(fnam)
    
    f = h5py.File(fnam, 'w')
    f.create_dataset('data', data = diff)
    f.create_dataset('data retrieved', data = diff_ret)
    f.create_dataset('sample support', data = support.astype(np.int16))
    f.create_dataset('sample support retrieved', data = support_ret.astype(np.int16))
    f.create_dataset('good pixels', data = good_pix.astype(np.int16))
    f.create_dataset('modulus error', data = emod)
    if efid is not None :
        f.create_dataset('fidelity error', data = efid)
    else :
        f.create_dataset('fidelity error', data = -np.ones_like(emod))
    f.create_dataset('sample init', data = solid_unit)
    f.create_dataset('sample retrieved', data = solid_unit_ret)

    # read the config file and dump it into the h5 file
    """
    g = open(config).readlines()
    h = ''
    for line in g:
        h += line
    f.create_dataset('config file', data = np.array(h))
    """
    f.close()
    return 


def read_output_h5(path):
    import os, h5py
    f = h5py.File(path, 'r')
    diff           = f['data'].value
    diff_ret       = f['data retrieved'].value
    support        = f['sample support'].value.astype(np.bool)
    support_ret    = f['sample support retrieved'].value.astype(np.bool)
    good_pix       = f['good pixels'].value.astype(np.bool)
    emod           = f['modulus error'].value
    efid           = f['fidelity error'].value
    solid_unit     = f['sample init'].value
    solid_unit_ret = f['sample retrieved'].value
    #config_file    = f['config file'].value

    f.close()

    # read then pass the config file
    """
    import ConfigParser
    import StringIO
    config_file = StringIO.StringIO(config_file)

    config = ConfigParser.ConfigParser()
    config.readfp(config_file)
    params = parse_parameters(config)
    """
    
    return diff, diff_ret, support, support_ret, \
        good_pix, solid_unit, solid_unit_ret, emod, efid


def write_input_h5(path, diff, support, good_pix, solid_known, config):
    import os, h5py
    fnam = os.path.join(path, 'input.h5')
    if_exists_del(fnam)
    
    f = h5py.File(fnam, 'w')
    f.create_dataset('data', data = diff)
    f.create_dataset('sample support', data = support.astype(np.int16))
    f.create_dataset('good pixels', data = good_pix.astype(np.int16))
    if solid_known is not None :
        f.create_dataset('sample', data = solid_known)
    # read the config file and dump it into the h5 file
    g = open(config).readlines()
    h = ''
    for line in g:
        h += line
    f.create_dataset('config file', data = np.array(h))
    f.close()
    return 


def read_input_h5(fnam):
    import h5py
    
    f = h5py.File(fnam, 'r')
    diff     = f['data'].value
    support  = f['sample support'].value.astype(np.bool)
    good_pix = f['good pixels'].value.astype(np.bool)
    
    if 'sample' in f.keys():
        solid_known = f['sample'].value
    else :
        solid_known = None

    config_file = f['config file'].value

    f.close()

    # read then pass the config file
    import ConfigParser
    import StringIO
    config_file = StringIO.StringIO(config_file)

    config = ConfigParser.ConfigParser()
    config.readfp(config_file)
    params = parse_parameters(config)
    return diff, support, good_pix, solid_known, params


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


def if_exists_del(fnam):
    import os
    # check that the directory exists and is a directory
    output_dir = os.path.split( os.path.realpath(fnam) )[0]
    if os.path.exists(output_dir) == False :
        raise ValueError('specified path does not exist: ', output_dir)
    
    if os.path.isdir(output_dir) == False :
        raise ValueError('specified path is not a path you dummy: ', output_dir)
    
    # see if it exists and if so delete it 
    # (probably dangerous but otherwise this gets really anoying for debuging)
    if os.path.exists(fnam):
        print '\n', fnam ,'file already exists, deleting the old one and making a new one'
        os.remove(fnam)
