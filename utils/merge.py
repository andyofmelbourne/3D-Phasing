import numpy as np
import h5py
from itertools import product
from noise import rad_av

def centre2(O):
    import scipy.ndimage
    a  = (O * O.conj()).real
    a  = np.fft.fftshift(a)

    # a is 'cut' on the side of the array when the centre of mass is 
    # in unstable equilibrium

    cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int)# - np.array(a.shape)/2
    print 'centre of mass after shifting and fft rolling', cm
    if not np.allclose(cm, np.array(a.shape) / 2.) :
        print 'bad centering, rolling a bit and retrying:'
        out = multiroll(O, (np.random.random((3,)) * np.array(a.shape)).astype(np.int) )
        out = centre(out)
        out = centre2(out)
    else :
        out = O
    return out



def merge_sols(Os):
    """
    grab the solutions, align the phases, un-flip and centre
    then average them.

    Also return the radial average of the merged farfield 
    diffraction pattern.
    """
    print '\n Merging solutions'
    print ' centering...'
    for i in range(len(Os)):
        Os[i] = centre2(Os[i])
        
    print ' aligning phases...'
    for i in range(len(Os)):
        s     = np.sum(Os[i])
        phase = np.arctan2(s.imag, s.real)
        Os[i] = Os[i] * np.exp(- 1J * phase)
        print '\t sum(imag) after alignment:', np.sum(Os[i].imag)

    print '\n flipping with respect to Os[0]'
    O = Os[0]
    for i in range(1, len(Os)):
        Ot  = Os[i].copy()
        Ot  = O - Ot
        Ot  = (Ot * Ot.conj()).real
        er1 = np.sum( Ot )
    
        Ot  = Os[i].copy()
        Ot  = Ot[::-1, ::-1, ::-1]
        Ot  = multiroll(Ot, [1,1,1])
        Ot2  = O - Ot
        Ot2 = (Ot2 * Ot2.conj()).real
        er2 = np.sum( Ot2 )

        print ''
        print '\t error un-flipped:', er1
        print '\t error    flipped:', er2
        if er2 < er1 :
            print '\t flipping...'
            Os[i] = Ot

    O = np.sum(Os, axis = 0) / float(Os.shape[0])

    T, rav_T = transmission(Os)
    return O, T, rav_T

def transmission(Os):
    Os = np.fft.fftn(Os, axes = (-3,-2,-1))
    Os = np.angle(Os)
    T  = np.abs(np.average(np.exp(1J * Os), axis=0))
    rav_T = rad_av(T, is_fft_shifted = True)
    return T, rav_T

def radial_average_from_sol(O):
    Oh = np.fft.fftn(O)
    rad_av_intensity = rad_av(np.abs(Oh)**2, is_fft_shifted = True)
    rad_av_phase     = rad_av(np.angle(Oh),  is_fft_shifted = True)
    return rad_av_intensity, rad_av_phase

def centre(array):
    # get the centre of mass of |P|^2
    import scipy.ndimage
    a  = (array.conj() * array).real
    cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int)# - np.array(a.shape)/2
    #print '\ncentre of mass before shifting', cm
    
    # centre array
    array = np.roll(array, -cm[0], 0)
    array = np.roll(array, -cm[1], 1)
    array = np.roll(array, -cm[2], 2)
    #array = era.multiroll(array, -cm)

    # double check
    #a  = (array.conj() * array).real
    #cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int)# - np.array(a.shape)/2
    #print 'centre of mass after shifting', cm

    #a  = np.fft.fftshift(a)
    #cm = np.rint(scipy.ndimage.measurements.center_of_mass(a)).astype(np.int)# - np.array(a.shape)/2
    #print 'centre of mass after shifting and fft rolling', cm
    return array
    
def multiroll(x, shift, axis=None):
    """Roll an array along each axis.
    Thanks to: Warren Weckesser, 
    http://stackoverflow.com/questions/30639656/numpy-roll-in-several-dimensions
    
    
    Parameters
    ----------
    x : array_like
        Array to be rolled.
    shift : sequence of int
        Number of indices by which to shift each axis.
    axis : sequence of int, optional
        The axes to be rolled.  If not given, all axes is assumed, and
        len(shift) must equal the number of dimensions of x.
    Returns
    -------
    y : numpy array, with the same type and size as x
        The rolled array.
    Notes
    -----
    The length of x along each axis must be positive.  The function
    does not handle arrays that have axes with length 0.
    See Also
    --------
    numpy.roll
    Example
    -------
    Here's a two-dimensional array:
    >>> x = np.arange(20).reshape(4,5)
    >>> x 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    Roll the first axis one step and the second axis three steps:
    >>> multiroll(x, [1, 3])
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])
    That's equivalent to:
    >>> np.roll(np.roll(x, 1, axis=0), 3, axis=1)
    array([[17, 18, 19, 15, 16],
           [ 2,  3,  4,  0,  1],
           [ 7,  8,  9,  5,  6],
           [12, 13, 14, 10, 11]])
    Not all the axes must be rolled.  The following uses
    the `axis` argument to roll just the second axis:
    >>> multiroll(x, [2], axis=[1])
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])
    which is equivalent to:
    >>> np.roll(x, 2, axis=1)
    array([[ 3,  4,  0,  1,  2],
           [ 8,  9,  5,  6,  7],
           [13, 14, 10, 11, 12],
           [18, 19, 15, 16, 17]])
    """
    x = np.asarray(x)
    if axis is None:
        if len(shift) != x.ndim:
            raise ValueError("The array has %d axes, but len(shift) is only "
                             "%d. When 'axis' is not given, a shift must be "
                             "provided for all axes." % (x.ndim, len(shift)))
        axis = range(x.ndim)
    else:
        # axis does not have to contain all the axes.  Here we append the
        # missing axes to axis, and for each missing axis, append 0 to shift.
        missing_axes = set(range(x.ndim)) - set(axis)
        num_missing = len(missing_axes)
        axis = tuple(axis) + tuple(missing_axes)
        shift = tuple(shift) + (0,)*num_missing

    # Use mod to convert all shifts to be values between 0 and the length
    # of the corresponding axis.
    shift = [s % x.shape[ax] for s, ax in zip(shift, axis)]

    # Reorder the values in shift to correspond to axes 0, 1, ..., x.ndim-1.
    shift = np.take(shift, np.argsort(axis))

    # Create the output array, and copy the shifted blocks from x to y.
    y = np.empty_like(x)
    src_slices = [(slice(n-shft, n), slice(0, n-shft))
                  for shft, n in zip(shift, x.shape)]
    dst_slices = [(slice(0, shft), slice(shft, n))
                  for shft, n in zip(shift, x.shape)]
    src_blks = product(*src_slices)
    dst_blks = product(*dst_slices)
    for src_blk, dst_blk in zip(src_blks, dst_blks):
        y[dst_blk] = x[src_blk]

    return y



if __name__ == '__main__':
    import h5py
    import pyqtgraph as pg
    f  = h5py.File('spi/ivan/output.h5', 'r')
    Os = f['sample retrieved'].value
    O  = merge_sols(Os.copy())
    pg.show(np.fft.fftshift(O.real))

    rad_av_data = rad_av(f['data'].value, is_fft_shifted = True)
    rad_av_in, rad_av_phase = radial_average_from_sol(O)
    plot = pg.plot(rad_av_data)
    plot.plot(rad_av_in, pen=pg.mkPen('b'))
    pg.plot(rad_av_phase)
