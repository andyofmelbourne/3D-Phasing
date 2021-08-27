import argparse
import sys

if __name__ == '__main__':
    description = "Phase a far-field diffraction volume using iterative projection algorithms."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--repeats', type=int, default=1, \
                        help="repeat the iteration sequence this many times")
    parser.add_argument('-r', '--reality', action='store_true', \
                        help="Enforce reality of the object at each iteration")
    parser.add_argument('--iters', default=["100DM", "100ERA"], nargs='*', \
                        help="Iteration sequence for the algorith")
    parser.add_argument('-u', '--update_freq', type=int, default=0, \
                        help="write intermediate results to output every 'update_freq' iterations")
    parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                        help="Python pickle file containing a dictionary with keys 'intensity' and 'support'")
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
import re
import time

#    Difference Map
#       modes += mapper.Pmod(modes_sup * 2 - modes) - modes_sup
#    1.
#    O2 = O * S
#    O2 = (O2 + O2.conj())/2.
#    O  = O - O2
#    O2 = O2 - O
#    2.
#    O2 = O2 * amp / (amb(O2) + alpha)
#    3. 
#    O = O + O2
#
#    Error reduction
#    O2 = O * S
#    O2 = (O2 + O2.conj())/2.

prgs_code = r"""//CL//
#include <pyopencl-complex.h>


__kernel void DM_support (
    __global cfloat_t *O, 
    __global cfloat_t *O2, 
    __global const char *S 
    )
{
int i = get_global_id(0);

O2[i].x = O[i].x * S[i];
O2[i].y = O[i].y * S[i];

O[i].x -= O2[i].x;
O[i].y -= O2[i].y;

O2[i].x -= O[i].x;
O2[i].y = -O[i].y;
}


__kernel void DM_support_real (
    __global cfloat_t *O, 
    __global cfloat_t *O2, 
    __global const char *S 
    )
{
int i = get_global_id(0);

O2[i].x = O[i].x * S[i];
O2[i].y = 0.;

O[i].x -= O2[i].x;

O2[i].x -= O[i].x;
O2[i].y = -O[i].y;
}


__kernel void Pmod (
    __global cfloat_t *O, 
    __global const float *amp 
    )
{
int i = get_global_id(0);

float angle = atan2(O[i].y, O[i].x);

O[i].x = amp[i] * cos(angle);
O[i].y = amp[i] * sin(angle);

}



__kernel void DM_update (
    __global cfloat_t *O, 
    __global const cfloat_t *O2
    )
{
int i = get_global_id(0);

O[i].x += O2[i].x;
O[i].y += O2[i].y;
}


__kernel void ERA_support (
    __global cfloat_t *O, 
    __global const char *S 
    )
{
int i = get_global_id(0);

O[i].x = O[i].x * S[i];
O[i].y = O[i].y * S[i];
}

__kernel void ERA_support_real (
    __global cfloat_t *O, 
    __global const char *S 
    )
{
int i = get_global_id(0);

O[i].x = O[i].x * S[i];
O[i].y = 0.;
}
"""

def iters_string_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters



def phase(I, S, iters="100DM 100ERA", reality=False, callback=None):
    ## Step #1. Obtain an OpenCL platform.
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.GPU)
        if len(devices) > 0:
            platform = p
            device   = devices[0]
            break
    
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
    queue   = cl.CommandQueue(context)
    
    api = cluda.ocl_api()
    thr = api.Thread(queue)
    
    prgs_build = cl.Program(context, prgs_code).build()
    
    #cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    O   = cl.array.to_device(queue, np.ascontiguousarray(np.empty(I.shape, dtype=np.complex64)))
    O2  = cl.array.to_device(queue, np.ascontiguousarray(np.empty(I.shape, dtype=np.complex64)))
    amp = cl.array.to_device(queue, np.ascontiguousarray(np.sqrt(I).astype(np.float32)))
    S   = cl.array.to_device(queue, np.ascontiguousarray(S.astype(np.int8)))
    
    # initialise fft routine
    fft  = reikna.fft.FFT(O)
    cfft = fft.compile(thr)
    
    # initialise random object
    O.set(amp.get() * np.exp(2J * np.pi * np.random.random(I.shape)).astype(np.complex64))
    cfft(O, O, 1)

    # define projections
    if reality :
        DM_support  = prgs_build.DM_support_real 
        ERA_support = prgs_build.ERA_support_real 
    else :
        DM_support  = prgs_build.DM_support 
        ERA_support = prgs_build.ERA_support
    
    I_norm = np.sum(I)
    
    # Difference Map
    #---------------
    def DM(DM_iters):
        it = tqdm.tqdm(range(DM_iters), desc='IPA DM', file=sys.stderr)
        for i in it:
            launch = DM_support(queue, (O.size,), None, O.base_data, O2.base_data, S.base_data)
            
            cfft(O2, O2)
            
            # print error
            err = np.sqrt(cl.array.sum( (amp - abs(O2))**2 ).get()[()]/I_norm)
            it.set_description('IPA DM {:.2e}'.format(err))
            
            launch = prgs_build.Pmod(queue, (O.size,), None, O2.base_data, amp.base_data)
            
            cfft(O2, O2, 1)
            
            launch = prgs_build.DM_update(queue, (O.size,), None, O.base_data, O2.base_data)
            
            queue.finish()

            if callback :
                callback(O.get() * np.sqrt(I.size), i)

    # Error Reduction
    #----------------
    def ERA(ERA_iters):
        it = tqdm.tqdm(range(ERA_iters), desc='IPA ERA', file=sys.stderr)
        for i in it:
            launch = ERA_support(queue, (O.size,), None, O.data, S.data)
            
            cfft(O, O)
            
            # print error
            err = np.sqrt(np.sum( (amp.get() - np.abs(O.get()))**2 )/I_norm)
            it.set_description('IPA ERA {:.2e}'.format(err))
            
            launch = prgs_build.Pmod(queue, (O.size,), None, O.data, amp.data)
            
            cfft(O, O, 1)
            queue.finish()
            
            if callback :
                callback(O.get() * np.sqrt(I.size), i)
	
    # parse iteration sequence
    seq = iters_string_to_alg_num(iters)
    for s in seq:
        if s[0] == 'ERA':
            ERA(s[1])
        elif s[0] == 'DM':
            DM(s[1])
        else :
            raise ValueError('Could not parse iteration sequence string:' + iters)
    
    return O.get() * np.sqrt(I.size)
      
if __name__ == '__main__':
    # 1. read in electron density from stdin
    pipe = pickle.load(args.input)
    I = pipe['intensity']
    S = pipe['support']
    
    def pipe(O, i):
        if i % args.update_freq == 0:
            pickle.dump({'object_partial': O}, args.output)
    
    if args.update_freq == 0 :
        pipe = None
    
    for i in range(args.repeats):
        O = phase(I, S, ' '.join(args.iters), reality=args.reality, callback=pipe)
        
        pickle.dump({'object': O}, args.output)
