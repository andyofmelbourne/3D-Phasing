import numpy as np

def expand_region_by(mask, frac):
    import scipy.ndimage
    
    N = np.sum(mask)
    for sig in range(1, 20):
        # convolve mask with a gaussian
        mask_out = scipy.ndimage.filters.gaussian_filter(mask.copy().astype(np.float), sig, mode = 'constant')

        # progressively truncate until the sesired number of pixels is reached
        M = frac * N
        threshs = np.linspace(mask_out.max(), mask_out.min(), 100)
        for thresh in threshs:
            s = np.sum(mask_out > thresh)
            #print thresh, s
            if s > M :
                return (mask_out > thresh)

        # we did not find a good candidate 
