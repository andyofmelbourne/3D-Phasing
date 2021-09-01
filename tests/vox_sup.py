import numpy as np
import pyopencl as cl
import pyopencl.array 
import reikna.cluda as cluda
import reikna.fft
import tqdm
import pickle
import re
import time

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
float s;
float s0 = 0.;
float s1 = FLT_MAX;
//float s1 = 2;

// find the minimum and maximum value of |O|^2
minmax[i] = 0.;
for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    t = O[n].x * O[n].x + O[n].y * O[n].y;
    if (t>minmax[i])
        minmax[i]=t;
}

if (i==0){
    s1 = 0.;
    for (n = 0; n < max_workers; n++){ 
        if (minmax[i]>s1)
            s1 = minmax[i];
    }
}

minmax[i] = FLT_MAX;
for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    t = O[n].x * O[n].x + O[n].y * O[n].y;
    if (t<minmax[i])
        minmax[i]=t;
}

if (i==0){
    s0 = FLT_MAX;;
    for (n = 0; n < max_workers; n++){ 
        if (minmax[i]<s0)
            s0 = minmax[i];
    }
}


barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

for (j=0; j<200; j++){
    s = (s0 + s1)/2.;
    
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
    
    
    if (tot>0)
        s0 = s;
    else 
        s1 = s;
        
}
}
"""



## Step #1. Obtain an OpenCL platform.
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if len(devices) > 0:
        platform = p
        device   = devices[0]
        break

max_workers = device.max_work_item_sizes[0]
#max_workers = 9
print('max workers:', max_workers)

## Step #3. Create a context for the selected device.
context = cl.Context([device])
queue   = cl.CommandQueue(context)

api = cluda.ocl_api()
thr = api.Thread(queue)

prgs_build = cl.Program(context, f"#define max_workers {max_workers}\n" + prgs_code).build()

#cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
Oc   = np.ascontiguousarray(np.random.random((128,128,128)).astype(np.complex64))
O    = cl.array.to_device(queue, Oc)
Sc   = np.ascontiguousarray(np.zeros(Oc.shape, dtype=np.int8))
S    = cl.array.to_device(queue, Sc)

for i in tqdm.tqdm(range(10)):
    l = prgs_build.voxel_number_mask(queue, (max_workers,), (max_workers,), O.data, S.data, np.int32(5000), np.int32(O.size))
    l.wait()
cl.enqueue_copy(queue, Sc, S.data)
queue.finish()
print(np.sum(Sc))
#print(np.abs(Oc)**2)

#import time
#t0 = time.time()
#for i in range(100):
#    err0 = np.sum(Oc)
#print(err0, time.time()-t0)

#t0 = time.time()
#for i in range(100):
    #launch = prgs_build.sum1(queue, (1,), (1,), O.data, err1.data, np.int32(Oc.size))
#    launch = prgs_build.sum2(queue, (max_workers,), (max_workers,), O.data, err1.data, np.int32(Oc.size))

#queue.finish()
#cl.enqueue_copy(queue, err1c, err1.data)
#print(err1.get()[0], time.time()-t0)
#print(err1c[0], time.time()-t0)
