import numpy as np
import sys

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
    if hardware == 'gpu':
        from era_gpu import ERA_gpu
        return ERA_gpu(I, iters, support, mask, O, background, method, hardware, alpha, dtype, queue, plan, full_output)
    
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

    # method 1
    #---------
    if method == 1 :
        if iters > 0 :
            print '\n\nalgrithm progress iteration convergence modulus error'
        for i in range(iters) :
            O0 = O.copy()
            
            # modulus projection 
            O = pmod_1(amp, O, mask, alpha = alpha)
            
            O1 = O.copy()
            
            # support projection 
            if type(support) is int :
                S = choose_N_highest_pixels( (O * O.conj()).real, support)
            else :
                S = support
            O = O * S
            
            # metrics
            O2 = O.copy()
            
            O2    -= O0
            eCon   = np.sum( (O2 * O2.conj()).real ) / np.sum( (O0 * O0.conj()).real )
            eCon   = np.sqrt(eCon)
            
            O1    -= O0
            eMod   = np.sum( (O1 * O1.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )
            
            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['I']     = np.abs(np.fft.fftn(O))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            return O, info
        else :
            return O

    # method 2
    #---------
    # update the object with background retrieval
    elif method == 2 :
        if background is None :
            background = np.random.random((I.shape)).astype(dtype)
        else :
            temp       = np.empty(I.shape, dtype = dtype)
            temp[:]    = np.sqrt(background)
            background = temp
        
        print 'algrithm progress iteration convergence modulus error'
        for i in range(iters) :
            if update == 'O' : bak = O.copy()
            if update == 'P' : bak = P.copy()
            if update == 'OP': bak = np.hstack((O.ravel().copy(), P.ravel().copy()))
            
            E_bak        = exits.copy()
            
            # modulus projection 
            exits, background  = pmod_7(amp, background, exits, mask, alpha = alpha)
            
            E_bak       -= exits
            
            # consistency projection 
            if update == 'O': O, P_heatmap = psup_O_1(exits, P, R, O.shape, P_heatmap, alpha = alpha)
            if update == 'P': P, O_heatmap = psup_P_1(exits, O, R, O_heatmap, alpha = alpha)
            if update == 'OP':
                for j in range(OP_iters):
                    O, P_heatmap = psup_O_1(exits, P, R, O.shape, None, alpha = alpha)
                    P, O_heatmap = psup_P_1(exits, O, R, None, alpha = alpha)

            background[:] = np.mean(background, axis=0)
            
            exits = make_exits(O, P, R, exits)
            
            # metrics
            if update == 'O' : temp = O
            if update == 'P' : temp = P
            if update == 'OP': temp = np.hstack((O.ravel(), P.ravel()))
            
            bak   -= temp
            eCon   = np.sum( (bak * bak.conj()).real ) / np.sum( (temp * temp.conj()).real )
            eCon   = np.sqrt(eCon)
            
            eMod   = np.sum( (E_bak * E_bak.conj()).real ) / I_norm
            eMod   = np.sqrt(eMod)
            
            update_progress(i / max(1.0, float(iters-1)), 'ERA', i, eCon, eMod )

            eMods.append(eMod)
            eCons.append(eCon)
        
        if full_output : 
            info = {}
            info['exits'] = exits
            info['I']     = np.abs(np.fft.fftn(exits, axes = (-2, -1)))**2
            info['eMod']  = eMods
            info['eCon']  = eCons
            info['heatmap']  = P_heatmap
            if update == 'O' : return O, background**2, info
            if update == 'P' : return P, background**2, info
            if update == 'OP': return O, P, background**2, info
        else :
            if update == 'O':  return O, background**2
            if update == 'P':  return P, background**2
            if update == 'OP': return O, P, background**2


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
    O = np.fft.fftn(O)
    O = Pmod_1(amp, O, mask = mask, alpha = alpha)
    O = np.fft.ifftn(O)
    return O
    
def Pmod_1(amp, O, mask = 1, alpha = 1.0e-10):
    O  = mask * O * amp / (abs(O) + alpha)
    O += (1 - mask) * O
    return O

def pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10):
    exits = np.fft.fftn(exits, axes = (-2, -1))
    exits, background = Pmod_7(amp, background, exits, mask = mask, alpha = alpha)
    exits = np.fft.ifftn(exits, axes = (-2, -1))
    return exits, background
    
def Pmod_7(amp, background, exits, mask = 1, alpha = 1.0e-10):
    M = mask * amp / np.sqrt((exits.conj() * exits).real + background**2 + alpha)
    exits      *= M
    background *= M
    exits += (1 - mask) * exits
    return exits, background

