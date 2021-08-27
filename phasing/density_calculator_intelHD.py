#!/usr/bin/env python3
# write an opencl kernel to calculate density
# each work item calculates the density at a point in space for all atoms

import prody
import numpy as np
import os, sys
import tqdm
from .refdata import cromer_mann_params
import h5py
import pyopencl
import argparse
import pickle

render_kernel = \
r"""
__kernel void calculate_density(
    __global const float  *atom_coords_x,
    __global const float  *atom_coords_y,
    __global const float  *atom_coords_z,
    __global const float  *x_grid,
    __global const float  *y_grid,
    __global const float  *z_grid,
    __global const float  *occupancies,
    __global const float  *bfactors,
    __global const float  *crom_B,
    __global const float  *crom_w,
    __global float  *out,
    const int  atoms,
    const int  index_offset
    )
{                                                       
    float PI  = 4*3.14159265359;
    float PI2 = -4. * 3.14159265359 * 3.14159265359;

    float x, y, z, r2, x_atom, y_atom, z_atom, occ, BB, B;
    int i, j, work_group;
    
    // assume one work item per work group
    //work_group = get_group_id(1) + index_offset;
    work_group = get_global_id(1) + index_offset;
    //printf(" %i \n", work_group );
    
    // get my x y z coordinate
    x = x_grid[work_group];
    y = y_grid[work_group];
    z = z_grid[work_group];
    
    // loop over atoms
    for (i=0; i<atoms; i++){
        
        // atom coordinate
        x_atom = atom_coords_x[i];
        y_atom = atom_coords_y[i];
        z_atom = atom_coords_z[i];
        
        // occupancy
        occ = occupancies[i];
        
        // B-factor
        B = bfactors[i];

        // relative poistion of atom
        r2 = (x-x_atom)*(x-x_atom)+(y-y_atom)*(y-y_atom)+(z-z_atom)*(z-z_atom);
        
        if (r2 < 100.) {
            // loop cromer man coefs
            for (j=0; j<5; j++){
                // cromer mann coefs (gausian widths)
                BB = B + crom_B[5 * i + j];
                // printf(" %f %f %f \n", BB, r2, PI2 );
                out[work_group] += crom_w[5 * i + j] * occ * pow( PI / BB, (float)1.5) * exp(PI2 * r2 / BB);
                //printf(" %f %f %f %f \n", crom_w[5 * i + j], B , crom_B[5 * i + j], BB);
            }
        }
    }
}
"""

##################################################################
# OpenCL crap -- compile the kernel
##################################################################
import os
import pyopencl as cl
## Step #1. Obtain an OpenCL platform.
# with a gpu device
device = None
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
    if len(devices) > 0:
        platform = p
        device   = devices[0]
        break

# with a gpu device
if device is None :
    for p in cl.get_platforms():
        devices = p.get_devices(cl.device_type.CPU)
        if len(devices) > 0:
            platform = p
            device   = devices[0]
            break

## Step #3. Create a context for the selected device.
context = cl.Context([device])
queue   = cl.CommandQueue(context)

# load and compile the opencl code
program     = cl.Program(context, render_kernel).build('-cl-fast-relaxed-math -cl-mad-enable')
kernel      = program.calculate_density

kernel.set_scalar_arg_dtypes( 11*[None] + [np.int32, np.int32])


def render_molecule_opencl(xyz, occ, B, names, x, y, z, cromer_mann_params, B_offset=0.):
    mf = cl.mem_flags
    atom_coords_x = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(xyz[:, 0].astype(np.float32)))
    atom_coords_y = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(xyz[:, 1].astype(np.float32)))
    atom_coords_z = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(xyz[:, 2].astype(np.float32)))
    x_grid        = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(x.ravel().astype(np.float32)))
    y_grid        = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(y.ravel().astype(np.float32)))
    z_grid        = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(z.ravel().astype(np.float32)))
    crom_B        = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(np.array([[0.] + cromer_mann_params[name][4:-1] for name in names]).astype(np.float32)))
    crom_w        = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(np.array([[cromer_mann_params[name][8]] + cromer_mann_params[name][:4] for name in names]).astype(np.float32)))
    occ           = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(occ.astype(np.float32)))
    B             = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(B_offset + B.astype(np.float32)))
    S             = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = np.ascontiguousarray(np.zeros(x.shape, dtype=np.float32)))


    it = tqdm.tqdm(range(x.shape[0]), desc='updating pixel map', file=sys.stderr)
    for i in it:
        kernel(queue, (1, x.shape[1]*x.shape[2]), None, 
                atom_coords_x,
                atom_coords_y,
                atom_coords_z,
                x_grid,
                y_grid,
                z_grid,
                occ,
                B,
                crom_B,
                crom_w,
                S,
                len(xyz), 
                i*x.shape[1]*x.shape[2])
        queue.finish()
    
    out = np.empty(x.shape, dtype=np.float32)
    cl.enqueue_copy(queue, out, S)
    return out

    
    
# get a pdb file
# build the atoms in reciprocal space
# take the radial average

# 1. make an electron density map of the unit-cell
#   1. read pdb coords and b-factors (prody?)


def parse_pdb(pdb_fnam):
    pdb = prody.parsePDB(pdb_fnam)
    # get the atomic coordinates (N, 3) --> (atom no., x/y/z in Ang)
    #   I think this is just one mol.
    xyz = pdb.getCoords()
    occ = pdb.getOccupancies()
    B   = pdb.getBetas()
    # I'm getting fractional Zs from 1sx4 for some reason
    #Z   = np.rint(pdb.getMasses()).astype(np.int)
    names = pdb.getElements()
    return xyz, occ, B, names

def define_grid(xyz, vox_size = 1., domain='auto', pad=10.):
    if domain=='auto':
        xmin, xmax = np.min(xyz[:, 0]), np.max(xyz[:, 0])
        ymin, ymax = np.min(xyz[:, 1]), np.max(xyz[:, 1])
        zmin, zmax = np.min(xyz[:, 2]), np.max(xyz[:, 2])
        xmax += pad
        ymax += pad
        zmax += pad
        xmin -= pad
        ymin -= pad
        zmin -= pad
        X = np.array([xmax-xmin, ymax-ymin, zmax-zmin])
        x = np.arange(xmin, xmax, vox_size)
        y = np.arange(ymin, ymax, vox_size)
        z = np.arange(zmin, zmax, vox_size)
    
    # define the realspace grid
    x, y, z = np.meshgrid(x, y, z, indexing='ij')
    return x, y, z

def render_molecule_from_pdb(pdb, vox, B_offset=1.):
    xyz, occ, B, names = parse_pdb(pdb)
    
    x, y, z = define_grid(xyz, vox)
    
    print('array size:', x.shape, file=sys.stderr)
    S = render_molecule_opencl(xyz, occ, B, names, x, y, z, cromer_mann_params, B_offset)
    
    # normalise e-'s / A^3 --> e-'s / voxel 
    # S *= vox**3
    return S






