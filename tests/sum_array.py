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

// use 1 worker to sum the entire array
__kernel void sum1 (
    __global const float *O, 
    __global float *err, 
    const int N
    )
{
int n;
float tt = 0;

for (n = 0; n < N; n++){ 
    tt += O[n];
}
err[0] = tt;
}

// use 1 group and many workers to sum array
__kernel void sum2 (
    __global const float *O, 
    __global float *err, 
    const int N
    )
{

int n;
int i = get_global_id(0);
int w = get_local_size(0);
int dn = (N+1)/w;

local float tt[32];

tt[i] = 0.;

for (n = i*dn; n < min((i+1)*dn, N); n++){ 
    tt[i] += O[n];
    //if (i==0)
    //    printf(" %i %i %i %f \n", i, w, dn, tt[i]);
}

barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

if (i==0){
float ttt = 0.;
for (n = 0; n < w; n++){ 
    ttt += tt[n];
}
err[0] = ttt;
}
}

// test function call overhead
__kernel void sum3 (
    __global const float *O, 
    __global float *err, 
    const int N
    )
{
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
#max_workers = 1
print('max workers:', max_workers)

## Step #3. Create a context for the selected device.
context = cl.Context([device])
queue   = cl.CommandQueue(context)

api = cluda.ocl_api()
thr = api.Thread(queue)

prgs_build = cl.Program(context, prgs_code).build()

#cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
Oc   = np.ascontiguousarray(np.random.random((128,128,128)).astype(np.float32))
#Oc   = np.ascontiguousarray(np.arange(32).astype(np.float32))
O    = cl.array.to_device(queue, Oc)
err1c = np.ascontiguousarray(np.empty((1,), dtype=np.float32))
err1  = cl.array.to_device(queue, err1c)

import time
t0 = time.time()
for i in range(100):
    err0 = np.sum(Oc)
print(err0, time.time()-t0)

t0 = time.time()
for i in range(100):
    #launch = prgs_build.sum1(queue, (1,), (1,), O.data, err1.data, np.int32(Oc.size))
    launch = prgs_build.sum2(queue, (max_workers,), (max_workers,), O.data, err1.data, np.int32(Oc.size))

queue.finish()
cl.enqueue_copy(queue, err1c, err1.data)
#print(err1.get()[0], time.time()-t0)
print(err1c[0], time.time()-t0)
