import numpy as np
import afnumpy
import afnumpy.fft
import sys

afnumpy.arrayfire.set_device(2)

def ERA(I, iters, support, mask = 1, O = None, background = None, method = None, hardware = 'cpu', alpha = 1.0e-10, dtype = 'single', queue = None, plan = None, full_output = True):
    """
    Find the phases of 'I' given O using the Error Reduction Algorithm.
    
    Parameters
    ----------
    I : numpy.ndarray, (N, M, K)
        Merged diffraction patterns to be phased. 
    
        N : the number of pixels along slowest scan axis of the detector
        M : the number of pixels along slow scan axis of the detector
        K : the number of pixels along fast scan axis of the detector
    
    O : numpy.ndarray, (N, M, K) 
        The real-space scattering density of the object such that:
            I = |F[O]|^2
        where F[.] is the 3D Fourier transform of '.'.     
    
    iters : int
        The number of ERA iterations to perform.
    
    support : (numpy.ndarray, None or int), (N, M, K)
        Real-space region where the object function is known to be zero. 
        If support is an integer then the N most intense pixels will be kept at
        each iteration.
    
    mask : numpy.ndarray, (N, M, K), optional, default (1)
        The valid detector pixels. Mask[i, j, k] = 1 (or True) when the detector pixel 
        i, j, k is valid, Mask[i, j, k] = 0 (or False) otherwise.
    
    method : (None, 1, 2, 3, 4), optional, default (None)
        method = None :
            Automattically choose method 1, 2 based on the contents of 'background'.
            if   'background' == None then method = 1
            elif 'background' != None then method = 2
        method = 1 :
            Just update 'O'
        method = 2 :
            Update 'O' and 'background'
    
    hardware : ('cpu', 'gpu'), optional, default ('cpu') 
        Choose to run the reconstruction on a single cpu core ('cpu') or a single gpu
        ('gpu'). The numerical results should be identical.
    
    alpha : float, optional, default (1.0e-10)
        A floating point number to regularise array division (prevents 1/0 errors).
    
    dtype : (None, 'single' or 'double'), optional, default ('single')
        Determines the numerical precision of the calculation. If dtype==None, then
        it is determined from the datatype of I.
    
    full_output : bool, optional, default (True)
        If true then return a bunch of diagnostics (see info) as a python dictionary 
        (a list of key : value pairs).
    
    Returns
    -------
    O : numpy.ndarray, (U, V, K) 
        The real-space object function after 'iters' iterations of the ERA algorithm.
    
    info : dict, optional
        contains diagnostics:
            
            'I'     : the diffraction pattern corresponding to object above
            'eMod'  : the modulus error for each iteration:
                      eMod_i = sqrt( sum(| O_i - Pmod(O_i) |^2) / I )
            'eCon'  : the convergence error for each iteration:
                      eCon_i = sqrt( sum(| O_i - O_i-1 |^2) / sum(| O_i |^2) )
        
    Notes 
    -----
    The ERA is the simplest iterative projection algorithm. It proceeds by 
    progressive projections of the exit surface waves onto the set of function that 
    satisfy the:
        modulus constraint : after propagation to the detector the exit surface waves
                             must have the same modulus (square root of the intensity) 
                             as the detected diffraction patterns (the I's).
        
        support constraint : the exit surface waves (W) must be separable into some object 
                                 and probe functions so that W_n = O_n x P.
    
    The 'projection' operation onto one of these constraints makes the smallest change to the set 
    of exit surface waves (in the Euclidean sense) that is required to satisfy said constraint.
    Examples 
    --------
    """
    method = 1
    
    if dtype is None :
        dtype   = I.dtype
        c_dtype = (I[0,0,0] + 1J * I[0, 0, 0]).dtype
    
    elif dtype == 'single':
        dtype   = np.float32
        c_dtype = np.complex64

    elif dtype == 'double':
        dtype   = np.float64
        c_dtype = np.complex128

    if O is None :
        O = np.random.random((I.shape)).astype(c_dtype)
    
    O    = O.astype(c_dtype)
    
    I_norm    = np.sum(mask * I)
    amp       = np.sqrt(I).astype(dtype)
    eMods     = []
    eCons     = []

    if background is not None :
        if background is True :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            background = np.sqrt(background)
        rs = None
        background = afnumpy.array(background)

    # send arrays to the gpu
    amp = afnumpy.array(amp)
    O   = afnumpy.array(O)
    mask = afnumpy.array(mask)
    support = afnumpy.array(support)


    # method 1
    #---------
    if method == 1 :
        if iters > 0 :
            print '\n\nalgrithm progress iteration convergence modulus error'
        for i in range(iters) :
            O0 = O.copy()
            
            # modulus projection 
            if background is not None :
                #print 'background'
                O, background  = pmod_7(amp, background, O, mask, alpha = alpha)
            else :
                O = pmod_1(amp, O, mask, alpha = alpha)
            
            O1 = O.copy()
            
            # support projection 
            if type(support) is int :
                print 'highest N'
                S = choose_N_highest_pixels( (O * O.conj()).real, support)
                S = afnumpy.array(choose_N_highest_pixels( (O * O.conj()).real, support))
            else :
                S = support
            O = O * S

            if background is not None :
                #print 'background'
                background, rs, r_av = radial_symetry(np.array(background), rs = rs)
                background = afnumpy.array(background)
            
            # metrics
            O2 = O.copy()
            
            O2    -= O0
            eCon   = afnumpy.sum( (O2 * O2.conj()).real ) / afnumpy.sum( (O0 * O0.conj()).real )
            eCon   = afnumpy.sqrt(eCon)
            
            O1    -= O0
            eMod   = afnumpy.sum( (O1 * O1.conj()).real ) / I_norm
            eMod   = afnumpy.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            O = np.array(O)
            info = {}
            info['plan'] = info['queue'] = None
            info['I']     = np.abs(np.fft.fftn(O))**2
            if background is not None :
                background, rs, r_av = radial_symetry(background**2, rs = rs)
                info['background'] = background
                info['r_av']       = r_av
                info['I']         += info['background']
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O, info
        else :
            return O


def update_progress(progress, algorithm, i, emod, esup):
    barLength = 15 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}: [{1}] {2}% {3} {4} {5} {6} {7}".format(algorithm, "#"*block + "-"*(barLength-block), int(progress*100), i, emod, esup, status, " " * 5) # this last bit clears the line
    sys.stdout.write(text)
    sys.stdout.flush()

def choose_N_highest_pixels(array, N):
    percent = (1. - float(N) / float(array.size)) * 100.
    thresh  = np.percentile(array, percent)
    support = array > thresh
    # print '\n\nchoose_N_highest_pixels'
    # print 'percentile         :', percent, '%'
    # print 'intensity threshold:', thresh
    # print 'number of pixels in support:', np.sum(support)
    return support

def pmod_1(amp, O, mask = 1, alpha = 1.0e-10):
    O = afnumpy.fft.fftn(O)
    O = Pmod_1(amp, O, mask = mask, alpha = alpha)
    O = afnumpy.fft.ifftn(O)
    return O
    
def Pmod_1(amp, O, mask = 1, alpha = 1.0e-10):
    out  = mask * O * amp / (afnumpy.abs(O) + alpha)
    out += (1 - mask) * O
    return out

def pmod_7(amp, background, O, mask = 1, alpha = 1.0e-10):
    O = afnumpy.fft.fftn(O)
    O, background = Pmod_7(amp, background, O, mask = mask, alpha = alpha)
    O = afnumpy.fft.ifftn(O)
    return O, background
    
def Pmod_7(amp, background, O, mask = 1, alpha = 1.0e-10):
    M = mask * amp / afnumpy.sqrt((O.conj() * O).real + background**2 + alpha)
    out         = O * M
    background *= M
    out += (1 - mask) * O
    return out, background

def radial_symetry(background, rs = None, is_fft_shifted = True):
    """
    Use arrayfire's histogram to calculate the radial averages.
    """
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
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

def _radial_symetry(background, rs = None, is_fft_shifted = True):
    if rs is None :
        i = np.fft.fftfreq(background.shape[0]) * background.shape[0]
        j = np.fft.fftfreq(background.shape[1]) * background.shape[1]
        k = np.fft.fftfreq(background.shape[2]) * background.shape[2]
        i, j, k = np.meshgrid(i, j, k, indexing='ij')
        rs      = np.sqrt(i**2 + j**2 + k**2).astype(np.int16)
        
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
    
