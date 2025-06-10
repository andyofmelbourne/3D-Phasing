import argparse
import sys

if __name__ == '__main__':
    description = "Phase a far-field diffraction volume using iterative projection algorithms."
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-p', '--repeats', type=int, default=1, \
                        help="repeat the iteration sequence this many times")
    parser.add_argument('-r', '--reality', action='store_true', \
                        help="Enforce reality of the object at each iteration")
    parser.add_argument('-s', '--psup_out', action='store_true', \
                        help="apply support projection before output")
    parser.add_argument('-b', '--radial_background_correction', action='store_true', \
                        help="Include radial background correction")
    parser.add_argument('-v', '--voxel_number', type=int, \
                        help="Use the voxel number support projection with given number of voxels")
    parser.add_argument('-t', '--threshold', type=float, \
                        help="use a threshold support projection")
    parser.add_argument('--iters', default=["100DM", "100ERA"], nargs='*', \
                        help="Iteration sequence for the algorith")
    parser.add_argument('-u', '--update_freq', type=int, default=0, \
                        help="write intermediate results to output every 'update_freq' iterations")
    parser.add_argument('--inversion_symmetry', default=False, action='store_true', \
                        help="enforce inversion symmetry (real Fourier values) at each iteration")
    parser.add_argument('--D6', default=False, action='store_true', \
                        help="enforce voxel perfect symmetry opperations from"
                        "the point group D6 at each iteration")
    parser.add_argument('--fftshift', default=False, action='store_true', \
                        help="fftshift output")
    parser.add_argument('--positive', default=False, action='store_true', \
                        help="enforce positivity")
    parser.add_argument('-c', '--centre', default=True, action='store_true', \
                        help="Centre the object accourding to the centre-of-mass of the support before output")
    parser.add_argument('--no-centre', dest='centre', action='store_false')
    parser.add_argument('--HIO_beta', nargs=2, type=float, default=[1., 1.], \
                        help="Feedback parameter for HIO")
    parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                        help="Python pickle file containing a dictionary with keys 'intensity' and 'support'")
    parser.add_argument('--shrink', nargs=3, type=float, \
                        help="Shrinkwrap parameters to apply during ERA"
                        "iterations. arguement is sig_start sig_stop threshold.  e.g. --shrink 2 0.5 0.3")
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout.buffer, \
                        help="Python pickle output file. The result is written as a dictionary with the key 'object'")
    args = parser.parse_args()


import numpy as np
import pyopencl as cl
import pyopencl.array
import tqdm
import pickle

import phasing.phase_routines
from phasing.phase_routines import (Opencl_init,
                                    Support_projection,
                                    Data_projection,
                                    generator_from_iters_string,
                                    centre_object,
                                    shrinkwrap)


def phase(
    I, O_in=None, S=None, mask=None, iters="100DM 100ERA",
    reality=False, radial_background_correction = False,
    voxel_number = None, update_freq=None, repeats=1,
    centre = False, beta=1., threshold=None, apply_support_before_output=False,
    shrink_sig_start = None, shrink_sig_stop = None, shrink_thresh = None,
    inversion_symmetry = False, fftshift=None, positive=False, D6=None
    ):

    # initialise opencl context, device, queue and reikna thread
    opencl_stuff = Opencl_init()

    # DM: modes += Pmod(modes_sup * 2 - modes) - modes_sup
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

        __kernel void HIO (
            __global cfloat_t *O,
            __global cfloat_t *Om,
            __global const char *S,
            float beta
            )
        {
        int i = get_global_id(0);

        O[i].x = S[i] * Om[i].x;
        O[i].y = S[i] * Om[i].y;

        O[i].x -= beta * (1-S[i]) * Om[i].x;
        O[i].y -= beta * (1-S[i]) * Om[i].y;

        // copy O to Om
        Om[i].x = O[i].x;
        Om[i].y = O[i].y;
        }

        __kernel void copyO (
            __global const cfloat_t *O,
            __global cfloat_t *O2
            )
        {
        int i = get_global_id(0);

        O2[i].x = O[i].x;
        O2[i].y = O[i].y;
        }

        __kernel void copyb (
            __global const float *bak,
            __global float *bak2
            )
        {
        int i = get_global_id(0);

        bak2[i] = bak[i];
        }
    """).build()

    # initialise object
    O = cl.array.empty(opencl_stuff.queue, I.shape, dtype=np.complex64)

    if O_in is not None :
        cl.enqueue_copy(opencl_stuff.queue, O.data, np.ascontiguousarray(O_in.astype(np.complex64)))

    # initialise background (even if not used)
    if radial_background_correction :
        bak = cl.array.empty(opencl_stuff.queue, I.shape, dtype=np.float32)
        bak.fill(0.)
    else :
        bak = cl.array.empty(opencl_stuff.queue, (1,), dtype=np.float32)

    # initialise projections
    support_projection = Support_projection(
        opencl_stuff, I.shape, S, voxel_number,
        threshold, reality, positive, radial_background_correction, D6=D6
    )

    data_projection = Data_projection(opencl_stuff, I, O, mask,
                                      radial_background_correction,
                                      real=inversion_symmetry,
                                      D6=False)

    # initialise DM arrays
    if ('DM' in iters) or ('HIO' in iters):
        O2 = cl.array.empty_like(O)

    if 'RAAR' in iters:
        O2 = cl.array.empty_like(O)
        O3 = cl.array.empty_like(O)

    if ('DM' in iters):
        bak2 = cl.array.empty_like(bak)

    for r in range(repeats):
        # parse iteration sequence
        seq_gen, total = generator_from_iters_string(iters)

        # get total number of shrinkwrap iterations
        shrink_iterations = iters.count('SHRINK')
        shrink_iteration = 0
        if shrink_iterations > 0:
            sigmas = np.linspace(
                shrink_sig_start, shrink_sig_stop, shrink_iterations
            )

        # get total number of RAAR iterations
        raar_iterations = iters.count('RAAR')
        raar_iteration = 0
        if raar_iterations > 0:
            betas = np.linspace(
                beta[0], beta[1], raar_iterations+1
            )
            print(f'{betas=}', file=sys.stderr)

        # initialise random object
        if O_in is None:
            Oc = np.sqrt(I) * np.exp(2J * np.pi * np.random.random(I.shape))
            cl.enqueue_copy(opencl_stuff.queue, O.data, np.ascontiguousarray(Oc.astype(np.complex64)))
        data_projection.cfft(O, O, 1)
        bak.fill(0.)

        # initialise O2 for HIO
        if 'HIO' in iters:
            cl_code.copyO(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
            data_projection(O2, bak)

        errs = []

        it = tqdm.tqdm(seq_gen, total=total, desc='IPA', file=sys.stderr)
        iteration = 0
        ERA_iterations = 0
        DM_iterations = 0
        last_alg = None
        for alg in it:
            if alg != last_alg:
                if alg == 'DM' or alg == 'HIO':
                    cl_code.copyO(opencl_stuff.queue, (O.size,), None, O.data, O2.data)

                if alg == 'DM':
                    cl_code.copyb(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)

            # hack
            if alg == 'ERA' or alg == 'ERAnosym':
                ERA_iterations += 1

                # if alg != ERA then P222 symmetry will be enforced instead of
                # D6
                support_projection(O, O, bak, bak, alg=alg)
                # support_projection(O, O, bak, bak, alg=None)

                data_projection(O, bak)

            elif alg == 'HIO':
                support_projection(O, O2, bak, bak, update_Oout=False)

                cl_code.HIO(opencl_stuff.queue, (O.size,), None, 
                            O.data, O2.data, support_projection.S.data, 
                            np.float32(beta))

                data_projection(O2, bak)

            # RAAR
            # r <-- [b/2 (Rs Rm + I) + (1 - b) Pm] r
            # Rs = 2 Ps - 1
            #
            # O2 = Pm O
            # O3 = (1-b) O2  # (1-b)Pm
            # O3 += (b/2) O  # b/2 I
            # O2 = Rm O = 2 O2 - O
            # O3 -= b/2 O2
            # O2 = Ps O2
            # O3 += b O2
            # O = O3
            elif alg == 'RAAR':
                beta_i = betas[raar_iteration]
                if alg != last_alg:
                    raar_iteration += 1
                print(f'{beta_i=}', file=sys.stderr)

                # O2 = Pm O
                O2[:] = O
                data_projection(O2, bak)

                # O3 = (1-b) O2  # (1-b)Pm
                O3[:] = (1-beta_i) * O2

                # O3 += (b/2) O  # b/2 I
                O3[:] += beta_i/2 * O

                # O2 = Rm O = 2 O2 - O
                O2[:] = 2 * O2 - O

                # O3 -= b/2 O2
                O3 -= beta_i/2 * O2

                # O2 = Ps O2
                support_projection(O2, O2, bak, bak)

                # O3 += b O2
                O3 += beta_i * O2

                # O = O3
                O[:] = O3

            elif alg == 'DM':
                DM_iterations += 1

                support_projection(O, O2, bak, bak2, vox=False)
                
                cl_code.DM1(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
                cl_code.DM1_bak(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)
                
                data_projection(O2, bak)

                cl_code.DM2(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
                cl_code.DM2_bak(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)

            elif alg == 'SHRINK':
            
            # if shrink_sig is not None and alg == 'ERA' and ERA_iterations % shrink_update == 0 :
            # if shrink_sig is not None and alg == 'DM' and DM_iterations % shrink_update == 0 :
                data_projection(O, bak)

                #print(f'\nshrinkwrap iteration = {iteration} shrinkwrap_index {shrinkwrap_index}\n', file=sys.stderr)
                St = support_projection.S.get()

                sig = sigmas[shrink_iteration]
                shrink_iteration += 1
                shrinkwrap(O.get(), St, sig = sig, thresh = shrink_thresh, iteration = iteration)
                
                support_projection.S.set(St) 

                #cl_code.copyO(opencl_stuff.queue, (O.size,), None, O.data, O2.data)
                #cl_code.copyb(opencl_stuff.queue, (bak.size,), None, bak.data, bak2.data)
                 
            opencl_stuff.queue.finish()
             
            it.set_description('IPA {} {:.2e}'.format(alg, data_projection.amp_err))
            
            errs.append(data_projection.amp_err)

            last_alg = alg
            
            # output results 
            iteration += 1
            if (update_freq and iteration % update_freq == 0) or iteration == total :
                print('sending output...', file=sys.stderr)
                if apply_support_before_output :
                    support_projection(O, O, bak, bak)
                    
                Oc    = O.get() * np.sqrt(I.size)
                Sc    = support_projection.S.get()
                
                if centre :
                    Oc, Sc = centre_object(Oc, Sc)

                if fftshift:
                    Oc = np.fft.fftshift(Oc)
                    Sc = np.fft.fftshift(Sc)

                # testing
                # out = {'object': Oc.T,
                #        'error': np.array(errs), }
                out = {'object': Oc,
                       'error': np.array(errs), }

                if radial_background_correction :
                    out['radial_background'] = bak.get()**2

                    if fftshift:
                        out['radial_background'] = np.fft.fftshift(out['radial_background'])

                #if voxel_number :
                #     out['support'] = Sc
                out['support'] = Sc.astype(bool)

                errs = []
                yield out


if __name__ == '__main__':
    # 1. read in electron density from stdin
    pipe = pickle.load(args.input)
    
    I = pipe['intensity']
    
    if 'support' in pipe :
        S = pipe['support']
    else :
        S = None

    if 'object' in pipe :
        O = pipe['object']
    else :
        O = None
    
    if 'mask' in pipe :
        print('loading intensity mask from input', file=sys.stderr)
        mask = pipe['mask']
    else :
        print('no intensity mask detected', file=sys.stderr)
        mask = None

    if args.shrink is not None :
        sig_start, sig_stop, thresh = args.shrink

    else :
        sig_start, sig_stop, thresh = [None, None, None]
        
    if args.inversion_symmetry :
        # be sure to centre the support since we have lost translational inveriance
        if S is not None :
            S, _ = centre_object(S, S)
            S = np.fft.ifftshift(S)

    phasor = phase(
                I, O, S=S, mask = mask, iters=' '.join(args.iters),
                reality = args.reality,
                radial_background_correction = args.radial_background_correction,
                voxel_number = args.voxel_number,
                update_freq = args.update_freq,
                repeats = args.repeats,
                centre = args.centre,
                beta = args.HIO_beta,
                threshold = args.threshold,
                apply_support_before_output = args.psup_out,
                shrink_sig_start = sig_start,
                shrink_sig_stop = sig_stop,
                shrink_thresh = thresh,
                inversion_symmetry = args.inversion_symmetry,
                fftshift = args.fftshift,
                positive = args.positive,
                D6 = args.D6
    )

    for out in phasor:
        pickle.dump(out, args.output)
