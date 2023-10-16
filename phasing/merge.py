import argparse
import sys

description = "Phase a far-field diffraction volume using iterative projection algorithms."
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-u', '--update_freq', type=int, default=0,  \
                    help="write intermediate results to output every 'update_freq' iterations")
parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                    help="Python pickle file containing a dictionary with keys 'object'")
parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout.buffer, \
                    help="Python pickle output file. The result is written as a dictionary with the key 'object'")
args = parser.parse_args()

import numpy as np
import pyopencl as cl
import pyopencl.array 
import reikna.cluda as cluda
import reikna.fft
import tqdm
import pickle

import warnings
warnings.simplefilter("error")
warnings.simplefilter("ignore", DeprecationWarning)

def step_downhill(x, err, f_calc, d, fd, min_step, maxiters=100):
    for j in range(maxiters):
        alpha   = -np.sign(fd) * 2**j * min_step 
        err_new = f_calc(x + alpha * d)
        if err_new > err :
            alpha = -np.sign(fd) * 2**(j-1) * min_step 
            #print('negative curvature, steping downhill', j)
            return x + alpha * d, alpha
        else :
            err = err_new

def test_step(x, er, f, alpha, d, curv, grad, etol, max_iters=10):
    for j in range(max_iters):
        t  = x + alpha * d
        # 
        er_t = f(t)
        er_m = er + alpha * grad + alpha**2/2 * curv
        # if the difference is too great halve the step size
        if np.abs(er_t-er_m) > etol :
            alpha = alpha/2.
        else :
            if j > 0 :
                #print('bad local error model, reducing stepsize', j)
                pass
            return t, alpha

def line_search_with_trust(x, f, d, fd, dfd, iters=10, etol=1):
    """
    Do a newton raphson line search
    
    model: 
    f(x + alpha * d) ~ f(x) + alpha * f'(x) . d + alpha^2 / 2 * dT . f''(x) . d
    """
    #alphas, xs = [0], [x.copy()]
    for i in range(iters):
        er   = f(x)
        grad = fd(x, d)
        curv = dfd(x, d)
        #
        # if the curvature is negative then we will end up 
        # searching for the local maximum (bad). In which case
        # we are better off making a small step down hill
        if curv < 0 :
            x, alpha = step_downhill(x, er, f, d, grad, 0.1, maxiters=100)
        #
        # test actual error vs model
        else :
            alpha = - grad / curv 
            x, alpha = test_step(x, er, f, alpha, d, curv, grad, etol, max_iters=10)
            
        #xs.append(x.copy())
        #alphas = alphas + [alphas[-1] + alpha]
        #print(alphas[-1])
    #return xs, alphas
    return x, True

class Cgls(object):
    """Minimise the function f using the nonlinear cgls algorithm.
    
    """
    def __init__(self, x0, f, df, fd, dfd = None, Md = None, imax = 10**5, e_tol = 1.0e-10, x_tol=1e-2):
        self.f     = f
        self.df    = df
        self.fd    = fd
        self.iters = 0
        self.imax  = imax
        self.e_tol = e_tol
        self.x_tol = x_tol
        self.errors = []
        self.x     = x0
        self.Md    = Md 

        self.dfd = dfd
        self.line_search = lambda x, d: line_search_with_trust(x, self.f, d, self.fd, self.dfd, iters=5, etol=1)
        self.cgls = self.cgls_Ploak_Ribiere

    def cgls_Ploak_Ribiere(self, iterations = None):
        """
        All of the vectors are 'selfed' so that the iterations may continue 
        when called again."""
        if self.iters == 0 :
            self.r         = - self.df(self.x)
            self.r_old     = self.r.copy()
            self.d         = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            self.delta_0   = self.delta_new.copy()
        # 
        if iterations == None :
            iterations = self.imax
        #
        x_old = self.x.copy()
        # 
        t = tqdm.trange(iterations, desc='cgls err:', file=sys.stderr)
        #t = range(iterations)
        for i in t:
            #
            # perform a line search of f along d
            self.x, status = self.line_search(self.x, self.d)
            # 
            self.r         = - self.df(self.x)
            delta_old      = self.delta_new
            delta_mid      = np.sum(self.r * self.r_old)
            self.r_old     = self.r.copy()
            self.delta_new = np.sum(self.r**2)
            #
            # Polak-Ribiere formula
            beta           = (self.delta_new - delta_mid)/ delta_old
            #
            self.d         = self.r + beta * self.d
            #
            # reset the algorithm 
            if (self.iters % self.x.size == 0) or (status == False) or beta <= 0.0 :
                self.d = self.r
            #
            # calculate the error
            self.errors.append(self.f(self.x))
            self.iters = self.iters + 1
            if self.iters > self.imax or (self.errors[-1] < self.e_tol) or np.max(np.abs(x_old - self.x)) < self.x_tol :
                break
            t.set_description("cgls err: {:.2e}".format(self.errors[-1]))
            x_old = self.x.copy()
        #
        #
        return self.x

def make_qs(shape):
    qs = []
    for i in range(len(shape)):
        q = np.fft.fftfreq(shape[i])
        s = [None for j in range(len(shape))]
        s[i] = slice(None)
        qs.append(q[tuple(s)].copy())
    return qs

def mkramp(r, q):
    out = np.exp(2J * np.pi * np.sum(np.array([r[i]*q[i] for i in range(len(r))], dtype=object)))
    return out

def error(a, b, r, q):
    return np.sum( np.abs(a - b * mkramp(r, q))**2 )

def grad(a, b, r, q):
    t  = -4J * np.pi * b * (a.conj() * mkramp(r, q) - b.conj())
    out = np.array([np.sum( qq * t.real ) for qq in q])
    return out

def grad_dot(d, a, b, r, q):
    g = grad(a, b, r, q)
    return np.dot(g, d)

def dHd(d, a, b, r, q):
    t = 8 * np.pi**2 * a.conj() * b * mkramp(r, q)
    H = []
    for i in range(len(r)):
        for j in range(len(r)):
            H.append(np.sum(q[i] * q[j] * t.real))
    H = np.array(H).reshape((len(r), len(r)))
    out = np.dot(d, np.dot(H, d))
    return out

def cgls_align(Oth, oh):
    a = Oth.copy()
    b = oh.copy()
    norm = np.abs(a).max()
    a /= norm
    b /= norm
    
    q = make_qs(a.shape)
    
    x0 = np.zeros((len(a.shape),))
    cgls = Cgls(x0, lambda x: error(a, b, x, q), lambda x: grad(a, b, x, q), lambda x, d: grad_dot(d, a, b, x, q), lambda x, d: dHd(d, a, b, x, q))
    cgls.cgls(100)
        
    return norm * b * mkramp(cgls.x, q)


def merge(Oth, PRTF, O, index):
    oh = np.fft.fftn(O)
    
    if Oth is not None :
        # use convolution to get approximate allignment
        # also check for inversion
        C     = np.abs(np.fft.ifftn(oh * Oth.conj()))
        dx    = np.unravel_index(np.argmax(C), C.shape)
        C_max = C[dx]
        
        C     = np.abs(np.fft.ifftn(oh.conj() * Oth.conj()))
        dx_inv= np.unravel_index(np.argmax(C), C.shape)
        C_max_inv = C[dx_inv]
        
        if C_max_inv > C_max :
            dx = dx_inv
            oh = oh.conj()
        
        # calculate phase ramp for any number of dimensions
        xdq = np.zeros(C.shape, dtype=float)
        for i in range(len(C.shape)):
            q = np.fft.fftfreq(C.shape[i])
            s = [None for j in range(len(C.shape))]
            s[i] = slice(None)
            xdq += dx[i] * q[tuple(s)]
        
        oh *= np.exp(2J * np.pi * xdq)
        
        # set the mean phase value to zero
        oh *= np.exp(-1J * np.angle(oh).ravel()[0])
        
        # now refine with cgls
        oh = cgls_align(Oth/index, oh)
    
    else :
        Oth  = np.zeros(O.shape, dtype=np.complex128)
        PRTF = np.zeros(O.shape, dtype=np.complex128)
        
        # set the mean phase value to zero
        oh *= np.exp(-1J * np.angle(oh).ravel()[0])
    
    # add to total
    PRTF  += np.exp(1J * np.angle(oh))
    Oth   += oh
    
    return Oth, PRTF

def conv_metric(Oth, PRTF, index):
    return np.sum(np.abs(Oth) * np.abs(PRTF)) / np.sum(np.abs(Oth)) / index

def output(Oth, PRTF, tots, index):
    out = {}
    out['convergence_metric'] = conv_metric(Oth, PRTF, index)
    out['object']             = np.fft.ifftn(Oth/index)
    out['PRTF']               = np.abs(PRTF)/index
    for name in tots:
        out[name]             = tots[name] / index
    
    pickle.dump(out, args.output)
    args.output.flush()

import time
if __name__ == "__main__":
    Oth   = None 
    PRTF  = None
    index = 0

    tots = {}
    
    while True :
        try :
            # must be dict
            package = pickle.load(args.input)
            
            Oth, PRTF = merge(Oth, PRTF, package.pop('object'), index)
            
            # just average everything else
            for name in package.keys():
                if name not in tots :
                    tots[name] = package[name]
                else : 
                    tots[name] += package[name]
            
            index += 1
            
            if args.update_freq != 0 and index % args.update_freq == 0:
                output(Oth, PRTF, tots, index)
        
        except EOFError :
            break
        
        #except Exception as e :
        #    print(e, file=sys.stderr)
    
    output(Oth, PRTF, tots, index)
