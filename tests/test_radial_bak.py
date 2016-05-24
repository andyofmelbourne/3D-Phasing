import numpy as np
import afnumpy
import time

afnumpy.arrayfire.set_device(2)
#afnumpy.arrayfire.set_device(1)

def radial_symetry_af2(background, inds = None, tots = None, is_fft_shifted = True):
    if (inds is None) or (tots is None) :
        print 'initialising...'
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = afnumpy.array(rs.ravel())
        
        # store a list of indexes and totals
        inds = []
        tots = []
        for i in range(0, afnumpy.max(rs)+1):
            m = (rs == i)
            j = afnumpy.where(m)
            inds.append(j)
            tots.append(afnumpy.float(afnumpy.sum(m)))
    
    out = background.ravel()
    for ind, tot in zip(inds[:5], tots[:5]) :
        out[ind] = 2.
        #a = afnumpy.sum(background[ind])
        #out[ind] = afnumpy.sum(background[ind]) #/ tot
    
    return out, inds, tots

def radial_symetry_af(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = afnumpy.array(rs)
    
    out = background
    for i in range(0, afnumpy.max(rs)+1):
        m     = (rs == i)
        msum  = afnumpy.sum(m)
        if msum > 1 :
            out[m] = afnumpy.sum(background[m]) / float(msum)
    
    return out, rs, None


def radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
    
    out = background.copy()
    for i in range(0, np.max(rs)+1):
        m     = (rs == i)
        msum  = np.sum(m)
        if msum > 1 :
            out[m] = np.sum(background[m]) / float(msum)
    
    return out, rs, None

def _radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int16)
        
        if is_fft_shifted is False :
            rs = np.fft.fftshift(rs)
        rs = rs.ravel()
    
    ########### Find the radial average
    # get the r histogram
    r_hist = np.bincount(rs)
    # get the radial total 
    r_av = np.bincount(rs, background.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)

    ########### Make a large background filled with the radial average
    background = r_av[rs].reshape(background.shape)
    return background, rs, r_av

if __name__ == '__main__':
    shape = (128,128,128)
    sig   = 20.

    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    k = np.fft.fftfreq(shape[2]) * shape[2]
    i, j, k = np.meshgrid(i, j, k, indexing='ij')
    rs0     = np.rint(np.sqrt(i**2 + j**2 + k**2)).astype(np.int16)
    
    a = np.exp(-rs0**2 / (2. * sig**2))
    a *= np.random.random(shape)


    rs = None
    a = afnumpy.array(a)
    d0 = time.time()
    for i in range(10):
        bak, rs, r_av = _radial_symetry(np.array(a), rs, is_fft_shifted = True)
        bak = afnumpy.array(bak)
    d1 = time.time()
    print d1 - d0

    rs = None
    a = np.random.random(shape)
    d0 = time.time()
    for i in range(10):
        bak, rs, r_av = _radial_symetry(a, rs, is_fft_shifted = True)
        bak = bak
    d1 = time.time()
    print d1 - d0

    rs = None
    a = afnumpy.array(a)
    d0 = time.time()
    for i in range(10):
        bak, rs, r_av = radial_symetry_af(a, rs, is_fft_shifted = True)
        bak = bak
    d1 = time.time()
    print d1 - d0
