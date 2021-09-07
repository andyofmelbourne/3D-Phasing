import argparse
import sys

if __name__ == '__main__':
    description = "Phase a far-field diffraction volume using iterative projection algorithms."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--repeats', type=int, default=1, \
                        help="repeat the iteration sequence this many times")
    parser.add_argument('-r', '--reality', action='store_true', \
                        help="Enforce reality of the object at each iteration")
    parser.add_argument('-b', '--radial_background_correction', action='store_true', \
                        help="Include radial background correction")
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
import time

import phasing.phase_routines

from phasing.phase_routines import Opencl_init, Support_projection, Data_projection, generator_from_iters_string

def phase(I, S=None, mask=None, iters="100DM 100ERA", reality=False, radial_background_correction = False, voxel_number = None, callback=None, callback_finished=None, update_freq=None):
    # initialise opencl context, device, queue and reikna thread
    opencl_stuff = Opencl_init()
    
    # modes += mapper.Pmod(modes_sup * 2 - modes) - modes_sup
    cl_code = cl.Program(opencl_stuff.context, r"""
        #include <pyopencl-complex.h>
        // O2 = Psup(O)
        __kernel void DM1 (
            __global cfloat_t *O, 
            __global cfloat_t *O2
            )
        {
        int i = get_global_id(0);
        
        O[i].x -= O2[i].x;
        O[i].y -= O2[i].y;
        O2[i].x -= O[i].x;
        O2[i].y -= O[i].y;
        }
        
        // O2 = Pmod(2*Psup(O) - O)
        __kernel void DM2 (
            __global cfloat_t *O, 
            __global const cfloat_t *O2
            )
        {
        int i = get_global_id(0);
        
        O[i].x += O2[i].x;
        O[i].y += O2[i].y;
        }

        __kernel void DM1_bak (
            __global float *bak, 
            __global float *bak2
            )
        {
        int i = get_global_id(0);
        
        bak[i] -= bak2[i];
        bak2[i] -= bak[i];
        }
        
        __kernel void DM2_bak (
            __global float *bak, 
            __global const float *bak2
            )
        {
        int i = get_global_id(0);
        
        bak[i] += bak2[i];
        }
    """).build()
	
    # initialise random object
    O  = cl.array.empty(opencl_stuff.queue, I.shape, dtype=np.complex64)
    Oc = np.sqrt(I) * np.exp(2J * np.pi * np.random.random(I.shape))
    cl.enqueue_copy(opencl_stuff.queue, O.data, np.ascontiguousarray(Oc.astype(np.complex64)))

    if radial_background_correction :
        bak = cl.array.empty(opencl_stuff.queue, I.shape, dtype=np.float32)
        bak.fill(0.)
    else :
        bak = cl.array.empty(opencl_stuff.queue, (1,), dtype=np.float32)
    
    # initialise projections
    support_projection = Support_projection(opencl_stuff, I.shape, 
                                            S, voxel_number, reality,      
                                            radial_background_correction)
    
    data_projection = Data_projection(opencl_stuff, I, O, mask,  
                                      radial_background_correction)
                                      #False)
    
    data_projection.cfft(O, O, 1)

    # initialise DM arrays
    if 'DM' in iters :
        O2   = cl.array.empty_like(O)
        bak2 = cl.array.empty_like(bak)
    
    # parse iteration sequence
    seq_gen, total = generator_from_iters_string(iters)
    
    it = tqdm.tqdm(seq_gen, total = total, desc='IPA', file=sys.stderr)
    iteration = 0
    for alg in it:
        if alg == 'ERA':
            support_projection(O, O, bak, bak)
            
            data_projection(O, bak)
            
            opencl_stuff.queue.finish()
            it.set_description('IPA ERA {:.2e}'.format(data_projection.amp_err))
        
        elif alg == 'DM':
            support_projection(O, O2, bak, bak2)
            
            cl_code.DM1(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
            cl_code.DM1_bak(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)
            
            data_projection(O2, bak)
            
            cl_code.DM2(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
            cl_code.DM2_bak(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)
             
            opencl_stuff.queue.finish()
            it.set_description('IPA DM {:.2e}'.format(data_projection.amp_err))
        
        # output results 
        iteration += 1
        if (update_freq and iteration % update_freq == 0) or iteration == (total-1) :
            out = {'object': O.get() * np.sqrt(I.size), 
                   'error': data_projection.amp_err, }
            
            if radial_background_correction :
                 out['radial_background'] = bak.get()
            
            if voxel_number :
                 out['support'] = support_projection.S.get()
            
            yield out
   
      
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

    for out in phase(
                I, S=S, iters=' '.join(args.iters), 
                reality=args.reality, 
                radial_background_correction = args.radial_background_correction, 
                voxel_number = args.voxel_number, 
                callback=pipe, 
                callback_finished=output,
                update_freq=args.update_freq):
        
        pickle.dump(out, args.output)
