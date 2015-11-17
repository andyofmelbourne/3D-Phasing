import numpy as np

def rad_av(diff, rs = None):
    if rs is None :
        i = np.arange(diff.shape[0]) - diff.shape[0] / 2
        j = np.arange(diff.shape[1]) - diff.shape[1] / 2
        k = np.arange(diff.shape[2]) - diff.shape[2] / 2
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs = np.sqrt(i**2 + j**2 + k**2).astype(np.int16).ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, diff.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    return r_av
