import argparse
import sys

if __name__ == '__main__':
    description = "Phase a far-field diffraction volume using iterative projection algorithms."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--repeats', type=int, default=1, \
                        help="repeat the iteration sequence this many times")
    parser.add_argument('-r', '--reality', action='store_true', \
                        help="Enforce reality of the object at each iteration")
    parser.add_argument('-v', '--voxel_number', type=int, \
                        help="Use the voxel number support projection with given number of voxels")
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

def choose_N_highest_pixels(array, N, tol = 1.0e-5, maxIters=1000, support=None):
    """
    Use bisection to find the root of
    e(x) = \sum_i (array_i > x) - N
    then return (array_i > x) a boolean mask
    This is faster than using percentile (surprising)
    If support is not None then values outside the support
    are ignored. 
    """
    s0 = array.max()
    s1 = array.min()

    if support is not None :
        a = array[support > 0]
    else :
        a = array
        support = 1
    
    for i in range(maxIters):
        s = (s0 + s1) / 2.
        e = np.sum(a > s) - N
    
        if np.abs(e) < tol :
            break

        if e < 0 :
            s0 = s
        else :
            s1 = s
        
    S = (array > s) * support
    print(s, array.min(), array.max(), S.size, np.argmax(array))
    #print 'number of pixels in support:', np.sum(support), i, s, e
    return S

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

__kernel void voxel_number_mask (
    __global const cfloat_t* O, 
    __global char* S, 
    const int NV,
    const int N
    )
{
int i = get_global_id(0);
int n, j;
int dn = N/max_workers + 1;
float t, a;

local float minmax[max_workers];
local int temp[max_workers];
local int tot;
local float s, s0, s1;

// find the minimum and maximum value of |O|^2
minmax[i] = 0.;
for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    t = O[n].x * O[n].x + O[n].y * O[n].y;
    if (t>minmax[i])
        minmax[i]=t;
}

barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

if (i==0){
    s1 = 0.;
    for (n = 0; n < max_workers; n++){ 
        if (minmax[n]>s1)
            s1 = minmax[n];
    }
}

barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

minmax[i] = FLT_MAX;
for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    t = O[n].x * O[n].x + O[n].y * O[n].y;
    if (t<minmax[i])
        minmax[i]=t;
}

barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

if (i==0){
    s0 = FLT_MAX;;
    for (n = 0; n < max_workers; n++){ 
        if (minmax[n]<s0)
            s0 = minmax[n];
    }
}


barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

for (j=0; j<100; j++){
    if (i==0)
        s = (s0 + s1)/2.;
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    // sum |O|^2 > s
    // each worker sums a bit
    temp[i] = 0.;
    for (n = i*dn; n < min((i+1)*dn, N); n++){ 
        t = O[n].x * O[n].x + O[n].y * O[n].y;
        if (t>s)
            temp[i]++;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    // worker 0 sums the sums
    if (i==0){
        tot = -NV;
        for (n = 0; n < max_workers; n++){ 
            tot += temp[n];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    //if (i==0)
    //    printf("%i %f %f %f %i\n", j, s0, s, s1, tot);
    
    if (tot==0) {
        // Fill S
        for (n = i*dn; n < min((i+1)*dn, N); n++){ 
            t = O[n].x * O[n].x + O[n].y * O[n].y;
            if (t>s) 
                S[n] = 1;
            else 
                S[n] = 0;
        }
        break;
    }
    
    if (i==0){
        if (tot>0)
            s0 = s;
        else 
            s1 = s;
    }
        
}
}


__kernel void amp_err (
    __global const cfloat_t *O, 
    __global const float *amp, 
    const float I_norm, 
    const int N, 
    __global float *result
    )
{
int i = get_global_id(0);
int n;
int dn = N/max_workers + 1;
float t = 0.;

local float temp[max_workers];

temp[i] = 0.;

// each worker sums a bit
for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    t = amp[n] - sqrt(O[n].x * O[n].x + O[n].y * O[n].y);
    temp[i] = t*t;
}

barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

// worker 0 sums the sums
if (i==0){
t = 0.;
for (n = 0; n < max_workers; n++){ 
    t += temp[n];
}
result[0] = sqrt(t/I_norm);
}


}

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
    __global const float *amp,
    __global const char *mask
    )
{
int i = get_global_id(0);

if (mask[i] == 1){
float angle = atan2(O[i].y, O[i].x);

O[i].x = amp[i] * cos(angle);
O[i].y = amp[i] * sin(angle);
}

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



def phase(I, S=None, mask=None, iters="100DM 100ERA", reality=False, repeats=1, voxel_number = None, callback=None, callback_finished=None):
    ## Step #1. Obtain an OpenCL platform.
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.GPU)
        if len(devices) > 0:
            platform = p
            device   = devices[0]
            break
    max_workers = device.max_work_item_sizes[0]
    
    ## Step #3. Create a context for the selected device.
    context = cl.Context([device])
    queue   = cl.CommandQueue(context)
    
    api = cluda.ocl_api()
    thr = api.Thread(queue)
    
    # add this definition to allow for variable length arrays
    prgs_build = cl.Program(context, f"#define max_workers {max_workers}\n" + prgs_code).build()
    
    #cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
    Oc  = np.ascontiguousarray(np.empty(I.shape, dtype=np.complex64))
    O   = cl.array.to_device(queue, np.ascontiguousarray(np.empty(I.shape, dtype=np.complex64)))
    O2  = cl.array.to_device(queue, np.ascontiguousarray(np.empty(I.shape, dtype=np.complex64)))
    amp = cl.array.to_device(queue, np.ascontiguousarray(np.sqrt(I).astype(np.float32)))
    err = np.ascontiguousarray(np.empty((1,), dtype=np.float32))
    errg= cl.array.to_device(queue, err)
    
    if voxel_number :
        S   = cl.array.to_device(queue, np.ascontiguousarray(np.zeros(I.shape, dtype=np.int8)))
    else :
        S   = cl.array.to_device(queue, np.ascontiguousarray(S.astype(np.int8)))
    
    if mask :
        mask   = cl.array.to_device(queue, np.ascontiguousarray(mask.astype(np.int8)))
    else :
        mask   = cl.array.to_device(queue, np.ascontiguousarray(np.ones(I.shape, dtype=np.int8)))
        
    
    # initialise fft routine
    fft  = reikna.fft.FFT(O)
    cfft = fft.compile(thr)
    
    # define projections
    if reality :
        DM_support  = prgs_build.DM_support_real 
        ERA_support = prgs_build.ERA_support_real 
    else :
        DM_support  = prgs_build.DM_support 
        ERA_support = prgs_build.ERA_support
    
    I_norm = np.sum(I)

    def calc_amp_err(o):
        launch = prgs_build.amp_err(queue, 
                                   (max_workers,), (max_workers,), 
                                   o.data, amp.data, 
                                   np.float32(I_norm), np.int32(o.size), 
                                   errg.data)
        launch.wait()
        cl.enqueue_copy(queue, err, errg.data)
        return err[0]
    
    # Difference Map
    #---------------
    def DM(DM_iters):
        it = tqdm.tqdm(range(DM_iters), desc='IPA DM', file=sys.stderr)
        for i in it:
            if voxel_number :
                l = prgs_build.voxel_number_mask(queue, (max_workers,), (max_workers,), O.data, S.data, np.int32(voxel_number), np.int32(O.size))
            
            launch = DM_support(queue, (O.size,), None, O.data, O2.data, S.data)
            
            cfft(O2, O2)
            
            amp_err = calc_amp_err(O2)
            it.set_description('IPA DM {:.2e}'.format(amp_err))
            
            launch = prgs_build.Pmod(queue, (O.size,), None, O2.data, amp.data, mask.data)
            
            cfft(O2, O2, 1)
            
            launch = prgs_build.DM_update(queue, (O.size,), None, O.data, O2.data)
            
            queue.finish()
            
            if callback :
                cl.enqueue_copy(queue, Oc, O.data)
                callback(Oc * np.sqrt(I.size), amp_err, i)
        return amp_err

    # Error Reduction
    #----------------
    def ERA(ERA_iters):
        it = tqdm.tqdm(range(ERA_iters), desc='IPA ERA', file=sys.stderr)
        for i in it:
            if voxel_number :
                l = prgs_build.voxel_number_mask(queue, (max_workers,), (max_workers,), O.data, S.data, np.int32(voxel_number), np.int32(O.size))
            
            launch = ERA_support(queue, (O.size,), None, O.data, S.data)
            
            cfft(O, O)
            
            amp_err = calc_amp_err(O)
            it.set_description('IPA ERA {:.2e}'.format(amp_err))
            
            launch = prgs_build.Pmod(queue, (O.size,), None, O.data, amp.data, mask.data)
            
            cfft(O, O, 1)
            queue.finish()
            
            if callback :
                cl.enqueue_copy(queue, Oc, O.data)
                callback(Oc * np.sqrt(I.size), amp_err, i)
        return amp_err
	
    for r in range(repeats):
        # initialise random object
        Oc[:] = np.sqrt(I) * np.exp(2J * np.pi * np.random.random(I.shape)).astype(np.complex64)
        cl.enqueue_copy(queue, O.data, Oc)
        cfft(O, O, 1)

        # parse iteration sequence
        seq = iters_string_to_alg_num(iters)
        for s in seq:
            if s[0] == 'ERA':
                amp_err = ERA(s[1])
            elif s[0] == 'DM':
                amp_err = DM(s[1])
            else :
                raise ValueError('Could not parse iteration sequence string:' + iters)
        
        if callback_finished :
            cl.enqueue_copy(queue, Oc, O.data)
            callback_finished(Oc * np.sqrt(I.size), amp_err)
    
    return O.get() * np.sqrt(I.size)
      
if __name__ == '__main__':
    # 1. read in electron density from stdin
    pipe = pickle.load(args.input)
    I = pipe['intensity']
    if 'support' in pipe :
        S = pipe['support']
    else :
        S = None
    
    if 'mask' in pipe :
        mask = pipe['mask']
    else :
        mask = None
    
    def pipe(O, err, i):
        if i % args.update_freq == 0:
            pickle.dump({'object_partial': O, 'error_partial': err}, args.output)
    
    def output(O, err):
        pickle.dump({'object': O, 'error': err}, args.output)
    
    if args.update_freq == 0 :
        pipe = None
        
    O = phase(I, S=S, iters=' '.join(args.iters), reality=args.reality, repeats=args.repeats, voxel_number = args.voxel_number, callback=pipe, callback_finished=output)
