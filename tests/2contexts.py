import numpy as np
import pyopencl as cl
import pyopencl.array 
import reikna.cluda as cluda
import reikna.fft
import tqdm
import pickle
import re
import time

## Step #1. Obtain an OpenCL platform.
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if len(devices) > 0:
        platform = p
        device1   = devices[0]
        break

## Step #1. Obtain an OpenCL platform.
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.CPU)
    if len(devices) > 0:
        platform = p
        device2   = devices[0]
        break

max_workers = device1.max_work_item_sizes[0]
print('max workers:', max_workers)
max_workers = device2.max_work_item_sizes[0]
print('max workers:', max_workers)

#max_workers = 1

## Step #3. Create a context for the selected device.
context1= cl.Context([device1])
context2= cl.Context([device2])
queue1  = cl.CommandQueue(context1)
queue2  = cl.CommandQueue(context2)

api = cluda.ocl_api()
thr1 = api.Thread(queue1)
thr2 = api.Thread(queue2)

#prgs_build = cl.Program(context, f"#define max_workers {max_workers}\n" + prgs_code).build()

#cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
Oc   = np.ascontiguousarray(np.random.random((128,128,128)).astype(np.complex64))
O1   = cl.array.to_device(queue1, Oc)
O2   = cl.array.to_device(queue2, Oc)

# initialise fft routine
cfft1 = reikna.fft.FFT(O1).compile(thr1)
cfft2 = reikna.fft.FFT(O1).compile(thr2)

N = 10
t0 = time.time()
for i in range(N):
    #cfft1(O1, O1)
    cfft1(O1, O1)
    #cfft2(O2, O2)
    cfft2(O2, O2)
queue2.finish()
queue1.finish()
print('{:.2e}'.format( (time.time()-t0)))
