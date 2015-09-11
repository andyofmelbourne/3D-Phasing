import numpy as np

def zero_pad_to_nearest_pow2(diff, shape_new = None):
    """
    find the smallest power of 2 that 
    fits each dimension of the diffraction
    pattern then zero pad keeping the zero
    pixel centred
    """
    if shape_new is None :
        shape_new = []
        for s in diff.shape:
            n = 0
            while 2**n < s :
                n += 1
            shape_new.append(2**n)

    print '\n reshaping:', diff.shape, '-->', shape_new
    diff_new = np.zeros(tuple(shape_new), dtype=diff.dtype)
    diff_new[:diff.shape[0], :diff.shape[1], :diff.shape[2]] = diff

    # roll the axis to keep N / 2 at N'/2
    for i in range(len(shape_new)):
        diff_new = np.roll(diff_new, shape_new[i]/2 - diff.shape[i] / 2, i)
    return diff_new

def mk_circle(shape, rad):
    i, j, k = np.indices(shape)
    r       = (i-shape[0]/2)**2 + (j-shape[1]/2)**2 + (k-shape[2]/2)**2 
    circle  = r < rad**2
    return circle
