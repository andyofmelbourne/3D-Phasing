import pyopencl as cl
import pyopencl.array 
import reikna.cluda as cluda
import reikna.fft
import numpy as np
import re

from reikna.algorithms import Reduce, Predicate, predicate_sum
from reikna.cluda import Snippet
from reikna.core import Annotation, Type, Transformation, Parameter

def centre_object(im, S):
    dims = range(len(im.shape))
    
    # convert image to probability density
    out = S / np.sum(S)
    
    # convert xyz coords to phases on the unit circle
    s = [slice(None),] + [None for i in dims]
    s = tuple(s)
    
    xyz = 2 * np.pi * np.indices(im.shape) / np.array(im.shape)[s]
    
    centres = list(dims)
    for dim in dims:
        # calculate circular centre of mass
        mean = np.angle(np.sum(out * np.exp(1J * xyz[dim])))
        if mean < 0 : 
            mean += 2*np.pi
        
        # map back to data coordinates
        centres[dim] = np.rint(mean * im.shape[dim] / (2 * np.pi)).astype(int)
        
    # roll data 
    im = np.roll(im, -np.array(centres), axis = tuple(dims))
    S  = np.roll(S, -np.array(centres), axis = tuple(dims))
    im = np.fft.fftshift(im)
    S  = np.fft.fftshift(S)
    return im, S

class Opencl_init():
    def __init__(self):
        # find an opencl device (preferably a GPU) in one of the available platforms
        for p in cl.get_platforms():
            devices = p.get_devices(cl.device_type.GPU)
            if len(devices) > 0:
                break
            
        if len(devices) == 0 :
            for p in cl.get_platforms():
                devices = p.get_devices()
                if len(devices) > 0:
                    break
        
        self.context = cl.Context(devices)
        self.queue   = cl.CommandQueue(self.context)
        
        # for the reikna module
        self.api = cluda.ocl_api()
        self.thr = self.api.Thread(self.queue)
        
    
class Support_projection():
    def __init__(self, opencl_stuff, shape, S, voxel_number, threshold, reality, radial_background_correction):
        self.queue = opencl_stuff.queue
        
        self.cl_code = cl.Program(opencl_stuff.context, r"""        
        #include <pyopencl-complex.h>
        __kernel void support (
            __global const cfloat_t *Oin, 
            __global cfloat_t *Oout, 
            __global const char *S 
            )
        {
        int i = get_global_id(0);
        
        Oout[i].x = Oin[i].x * S[i];
        Oout[i].y = Oin[i].y * S[i];
        }
        
        __kernel void support_real (
            __global const cfloat_t *Oin, 
            __global cfloat_t *Oout, 
            __global const char *S 
            )
        {
        int i = get_global_id(0);
        
        Oout[i].x = Oin[i].x * S[i];
        Oout[i].y = 0.;
        }

        __kernel void threshold_support (
            __global const cfloat_t *Oin, 
            __global char *S,
            float threshold
            )
        {
        int i = get_global_id(0);
        
        float t = Oin[i].x * Oin[i].x + Oin[i].y * Oin[i].y;
        //printf("%f %f\n", t, threshold);
        if (t > threshold) {
            S[i] = 1;
        } else {
            S[i] = 0;
        }
        }
        """).build()
        
        if S is not None :
            self.S = cl.array.to_device(self.queue, np.ascontiguousarray(S.astype(np.int8)))
        else :
            self.S = cl.array.empty(self.queue, shape, dtype=np.int8)
        
        if voxel_number :
            self.voxsup = VoxSup(opencl_stuff, shape, voxel_number)
        
        if radial_background_correction :
            self.radav = Radial_average(opencl_stuff, shape)

        if reality :
            self.support_proj = self.cl_code.support_real
        else :
            self.support_proj = self.cl_code.support

        if threshold :
            self.threshold_support = self.cl_code.threshold_support
            self.threshold_support.set_scalar_arg_dtypes([None, None, np.float32])
            # for fft factor
            self.threshold = threshold / np.prod(shape)**0.5
        else :
            self.threshold = threshold 
        
        self.voxel_number = voxel_number
        self.radial_background_correction = radial_background_correction 
        
    def __call__(self, Oin, Oout, bakin, bakout, update_Oout=True):
        if self.voxel_number:
            self.voxsup(Oin, self.S, tol=1)
        
        if self.threshold :
            import sys
            print(self.threshold, file=sys.stderr)
            self.threshold_support(self.queue, (Oin.size,), None, Oin.data, self.S.data, self.threshold)
            print(np.sum(self.S.get()), file=sys.stderr)
            print(np.max(np.abs(Oin.get())**2), file=sys.stderr)
        
        if update_Oout :
            self.support_proj(self.queue, (Oin.size,), None, Oin.data, Oout.data, self.S.data)
        
        if self.radial_background_correction :
            self.radav(bakin)
            self.radav.broadcast(bakout)

class Data_projection():
    def __init__(self, opencl_stuff, I, o, mask, radial_background_correction):
        self.cl_code = cl.Program(opencl_stuff.context, r"""        
        #include <pyopencl-complex.h>
        
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
         
        __kernel void Pmod_bak (
            __global cfloat_t *O, 
            __global float    *bak, 
            __global const float *amp,
            __global const char *mask
            )
        {
        int i = get_global_id(0);
        
        if (mask[i] == 1){
        float phi = atan2(sqrt(O[i].y*O[i].y + O[i].x*O[i].x), bak[i]);
        float theta = atan2(O[i].y, O[i].x);
        
        O[i].x = amp[i] * cos(theta) * sin(phi);
        O[i].y = amp[i] * sin(theta) * sin(phi);
        bak[i] = amp[i] * cos(phi);
        }
        }
        
        __kernel void amp_err (
            __global const cfloat_t *O, 
            __global const float *amp,
            __global const char *mask,
            __global float *diff
            )
        {
        int i = get_global_id(0);
        
        if (mask[i] == 1){
            diff[i] = pown( amp[i] - sqrt(O[i].x*O[i].x + O[i].y*O[i].y), 2);
        }
        }

        __kernel void amp_err_bak (
            __global const cfloat_t *O, 
            __global const float *bak,
            __global const float *amp,
            __global const char *mask,
            __global float *diff
            )
        {
        int i = get_global_id(0);
        
        if (mask[i] == 1){
            diff[i] = pown( amp[i] - sqrt(O[i].x*O[i].x + O[i].y*O[i].y + bak[i]*bak[i]), 2);
        }
        }
        
        """).build()

        if mask :
            self.mask = cl.array.to_device(opencl_stuff.queue,  
                                          np.ascontiguousarray(mask, dtype=np.int8))
        else :
            self.mask = cl.array.to_device(opencl_stuff.queue,  
                                          np.ascontiguousarray(np.ones(I.shape, dtype=np.int8)))
        
        if radial_background_correction :
            self.Pmod = self.Pmod_bak
        else :
            self.Pmod = self.Pmod_nobak
        
        self.amp = cl.array.to_device(opencl_stuff.queue,  
                                      np.ascontiguousarray(np.sqrt(I), dtype=np.float32))
        self.I_norm = np.sum(I)
            
        self.diff = cl.array.empty(opencl_stuff.queue, I.shape, dtype=np.float32)
        
        self.radial_background_correction = radial_background_correction
        
        # compile reikna fft class
        self.cfft = reikna.fft.FFT(o).compile(opencl_stuff.thr)

        # sum routine for data error calc
        self.rsum = Reikna_sum(opencl_stuff.thr,  self.amp)
        
        self.queue = opencl_stuff.queue
    
    def __call__(self, O, bak):
        self.cfft(O, O)
        self.Pmod(O, bak)
        self.cfft(O, O, 1)
        
        events = self.rsum(self.diff)
        [e.wait() for e in events]
        self.amp_err = (self.rsum.out.get()/self.I_norm)**0.5

    def Pmod_nobak(self, O, bak=None):
        self.cl_code.amp_err(self.queue, (O.size,), None, 
                             O.data, self.amp.data, 
                             self.mask.data, self.diff.data)
        #
        self.cl_code.Pmod(self.queue, (O.size,), None, 
                        O.data, self.amp.data, 
                        self.mask.data)
        

    def Pmod_bak(self, O, bak):
        self.cl_code.amp_err_bak(self.queue, (O.size,), None, 
                                 O.data, bak.data, self.amp.data, 
                                 self.mask.data, self.diff.data)
        #
        self.cl_code.Pmod_bak(self.queue, (O.size,), None, 
                            O.data, bak.data, 
                            self.amp.data, self.mask.data)

class Reikna_sum():
    def __init__(self, thr, arr, axes=None):
        reikna_sum = Reduce(arr, predicate_sum(arr.dtype), axes)
        self.creikna_sum = reikna_sum.compile(thr)
        self.out = thr.empty_like(reikna_sum.parameter.output)
    
    def __call__(self, in_arr):
        return self.creikna_sum(self.out, in_arr)

class Reikna_minmax():
    def __init__(self, thr, arr):
        # Minmax data type and the corresponding structure.
        # Note that the dtype must be aligned before use on a GPU.
        mmc_dtype = cluda.dtypes.align(np.complex64)
        mmc_c_decl = cluda.dtypes.ctype_module(mmc_dtype)
        
        empty = np.array([np.finfo(arr.dtype).max + 1J*np.finfo(arr.dtype).min,], mmc_dtype)[0]

        # Reduction predicate for the minmax.
        # v1 and v2 get the names of two variables to be processed.
        predicate = Predicate(
            Snippet.create(lambda v1, v2: """
                ${ctype} result = ${v1};
                if (${v2}.x < result.x)
                    result.x = ${v2}.x;
                if (${v2}.y > result.y)
                    result.y = ${v2}.y;
                return result;
                """,
                render_kwds=dict(ctype=mmc_c_decl)),
            empty)

        # A transformation that creates initial minmax structures for the given array of integers
        to_mmc = Transformation(
            [Parameter('output', Annotation(Type(mmc_dtype, arr.shape), 'o')),
            Parameter('input', Annotation(arr, 'i'))],
            """
            ${output.ctype} res;
            res.x = ${input.load_same};
            res.y = ${input.load_same};
            ${output.store_same}(res);
            """)
        
        
        # Create the reduction computation and attach the transformation above to its input.
        reduction = Reduce(to_mmc.output, predicate)
        reduction.parameter.input.connect(to_mmc, to_mmc.output, new_input=to_mmc.input)
        self.creduction = reduction.compile(thr)
        
        self.out = thr.empty_like(reduction.parameter.output)

    def __call__(self, arr):
        # Run the computation
        self.creduction(self.out, arr)


class VoxSup():

    def __init__(self, opencl_stuff, shape, voxels):
        queue = opencl_stuff.queue
        # array to hold |O|^2
        self.I    = cl.array.empty(queue, shape, np.float32)
        self.mask = cl.array.empty(queue, shape, np.float32)
        self.s    = cl.array.empty(queue, (1,), np.float32)
        self.voxels = cl.array.to_device(queue, np.array([voxels,], dtype=np.float32))
        self.queue = opencl_stuff.queue
        
        self.code = cl.Program(queue.context, r"""
        #include <pyopencl-complex.h>
        
        __kernel void intensity (
            __global const cfloat_t *O, 
            __global float *I
            )
        {
        
        int i = get_global_id(0);
        
        I[i] = O[i].x*O[i].x + O[i].y * O[i].y;
        }

        __kernel void mask_greater (
            __global const float *f, 
            __global float *mask, 
            __global const float *val
            )
        {

        int i = get_global_id(0);

        if (f[i] > val[0])
            mask[i] = 1;
        else 
            mask[i] = 0;
        }

        __kernel void div_interval (
            __global float2 *s01, 
            __global float *s
            )
        {
        s[0] = (s01[0].x + s01[0].y) / 2;
        }

        __kernel void update_interval (
            __global float2 *s01, 
            __global float *s,
            __global const float *sum,
            __global const float *voxels
            )
        {
        if (sum[0] > voxels[0])
            s01[0].x = s[0];
        else 
            s01[0].y = s[0];
        
        s[0] = (s01[0].x + s01[0].y) / 2;
        }

        __kernel void copy_mask_to_support (
            __global const float *mask, 
            __global char *S
            )
        {

        int i = get_global_id(0);

        S[i] = (char)mask[i];
        }
        """).build()

        self.rminmax = Reikna_minmax(opencl_stuff.thr, self.I)
        self.rsum    = Reikna_sum(opencl_stuff.thr, self.mask)

    def __call__(self, arr, S, tol=10):
        # calculate intensity
        launch = self.code.intensity(self.queue, (arr.size,), None, arr.data, self.I.data)
        
        # now get the max and min value
        self.rminmax(self.I)
        
        # now divide
        launch = self.code.div_interval(self.queue, (1,), (1,), self.rminmax.out.data, self.s.data)
        
        sum_last = None
        for i in range(100):
            # now creat float mask
            launch = self.code.mask_greater(self.queue, (arr.size,), None, self.I.data, self.mask.data, self.s.data)
            
            # now sum the mask
            self.rsum(self.mask)
            
            # now update interval
            launch = self.code.update_interval(self.queue, (1,), (1,), self.rminmax.out.data, self.s.data, self.rsum.out.data, self.voxels.data)
            
            sum = self.rsum.out.get()
            if sum_last is not None and np.abs(sum-sum_last)<tol :
                break
            sum_last = sum
            
        # copy float mask to support
        launch = self.code.copy_mask_to_support(self.queue, (S.size,), None, self.mask.data, S.data)


class Radial_average():
    def __init__(self, opencl_stuff, shape):
        queue = opencl_stuff.queue
        prgs_code = r"""//CL//
        #include <pyopencl-complex.h>
        
        // remap array 
        // 477 it/s on intel HD
        __kernel void remap (
            __global const float *f, 
            __global const uint *r,
            __global const uint *rindex,
            __global float *ravmap,
            uint I
            )
        {
        uint i = get_global_id(0);
        
        ravmap[r[i]*I + rindex[i]] = f[i];
        }
        
        __kernel void normalise (
            __global float *rav,
            __global const float *rcount
            )
        {
        uint i = get_global_id(0);
        
        rav[i] /= rcount[i];
        }
        
        __kernel void broadcast (
            __global const float *rav,
            __global const uint  *r,
            __global float *f
            )
        {
        uint i = get_global_id(0);
        
        f[i] = rav[r[i]];
        }
        """
        
        self.prgs_build = cl.Program(queue.context, prgs_code).build()
        
        # store radial values
        dim   = len(shape)

        if dim==1 :
            x    = np.fft.fftfreq(shape[0], 1/shape[0])
            r    = np.rint(np.sqrt(x**2)).astype(np.uint32)
        elif dim==2 :
            x    = np.fft.fftfreq(shape[0], 1/shape[0])[:, None]
            y    = np.fft.fftfreq(shape[1], 1/shape[1])[None, :]
            r    = np.rint(np.sqrt(x**2 + y**2)).astype(np.uint32)
        elif dim==3 :
            x    = np.fft.fftfreq(shape[0], 1/shape[0])[:, None, None]
            y    = np.fft.fftfreq(shape[1], 1/shape[1])[None, :, None]
            z    = np.fft.fftfreq(shape[2], 1/shape[2])[None, None, :]
            r    = np.rint(np.sqrt(x**2 + y**2 + z**2)).astype(np.uint32)
        
        r  = np.ascontiguousarray(r)
        
        # make a unique counter for pixels with the same r value
        # and store the number of pixels with a given r value
        rindex = np.zeros_like(r)
        rcount = np.ascontiguousarray(np.zeros((r.max(),), dtype=np.float32))
        m = np.ones(shape, dtype=bool)
        for val in range(r.max()):
            m[:] = r == val
            msum = np.sum(m)
            rindex[m] = np.arange(msum)
            rcount[val] = msum
         
        ravmap = np.ascontiguousarray(np.zeros((r.max(), rindex.max()), dtype=np.float32))
        
        self.I = np.uint32(ravmap.shape[1])
        self.r = cl.array.to_device(queue, r)
        self.rindex = cl.array.to_device(queue, rindex)
        self.ravmap = cl.array.to_device(queue, ravmap)
        self.rcount = cl.array.to_device(queue, rcount)
        self.rsum   = Reikna_sum(opencl_stuff.thr, self.ravmap, axes=1)
        self.queue  = opencl_stuff.queue
        
    def __call__(self, arr):
        launch = self.prgs_build.remap(
                    self.queue, (arr.size,), None, 
                    arr.data, self.r.data, self.rindex.data, 
                    self.ravmap.data, np.uint32(self.I))
        self.rsum(self.ravmap)
        launch = self.prgs_build.normalise(
                    self.queue, (self.rsum.out.size,), None, 
                    self.rsum.out.data, self.rcount.data)
        self.out = self.rsum.out
    
    def broadcast(self, arr):
        launch = self.prgs_build.broadcast(
                    self.queue, (arr.size,), None, 
                    self.rsum.out.data, self.r.data, arr.data)


def iters_string_to_alg_num(string):
    # split a string like '100ERA 200DM 50ERA' with the numbers
    steps = re.split('(\d+)', string)   # ['', '100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # get rid of empty strings
    steps = [s for s in steps if len(s)>0] # ['100', 'ERA ', '200', 'DM ', '50', 'ERA']
    
    # pair alg and iters
    # [['ERA', 100], ['DM', 200], ['ERA', 50]]
    alg_iters = [ [steps[i+1].strip(), int(steps[i])] for i in range(0, len(steps), 2)]
    return alg_iters

def alg_num_to_generator(alg_iters):
    for s in alg_iters :
        for i in range(s[1]):
            yield s[0]

def generator_from_iters_string(iters):
    alg_iters = iters_string_to_alg_num(iters)
    total_iterations = np.sum([s[1] for s in alg_iters])
    return alg_num_to_generator(alg_iters), total_iterations

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
    __global const float *bak, 
    __global const float *amp, 
    __global const char *mask, 
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
    t = mask[n] * (amp[n] - sqrt(O[n].x * O[n].x + O[n].y * O[n].y + bak[n] * bak[n]));
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

//  modes += mapper.Pmod(modes_sup * 2 - modes) - modes_sup
__kernel void DM_support_bak (
    __global float *bak_sup, 
    __global float *bak
    )
{
int i = get_global_id(0);

bak[i] -= bak_sup[i];

bak_sup[i] = bak_sup[i] - bak[i];
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

__kernel void Pmod_bak (
    __global cfloat_t *O, 
    __global float    *bak, 
    __global const float *amp,
    __global const char *mask
    )
{
int i = get_global_id(0);

if (mask[i] == 1){
float scale = amp[i] * rsqrt(1e-5 + O[i].x*O[i].x + O[i].y*O[i].y + bak[i]*bak[i]);

O[i].x *= scale;
O[i].y *= scale;
bak[i] *= scale;
}

}


__kernel void DM_update_bak (
    __global float *bak2, 
    __global float *bak
    )
{
int i = get_global_id(0);

bak[i] += bak2[i];
bak2[i] = bak[i];
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
