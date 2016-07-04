import os
import numpy as np
import noise
import support
import circle
import os

def generate_diff(config):
    solid_unit = make_3D_duck(shape = config['sample']['shape'])
    
    Solid_unit = np.fft.fftn(solid_unit, config['detector']['shape'])
    solid_unit_expanded = np.fft.ifftn(Solid_unit)

    diff  = np.abs(Solid_unit)**2

    # add noise
    if config['detector']['photons'] is not None :
        diff, edges = noise.add_noise_3d(diff, config['detector']['photons'], \
                                      remove_courners = config['detector']['cut_courners'],\
                                      unit_cell_size = config['sample']['diameter'])
    else :
        edges = np.ones_like(diff, dtype=np.bool)

    # add circle
    if config['detector']['add_circle'] is not None :
        #print 'adding circular background:'
        background_circle = np.max(diff) * 0.0001 * ~circle.make_beamstop(diff.shape, config['detector']['add_circle'])
        diff += background_circle 
    else :
        background_circle = None

    # define the solid_unit support
    if config['sample']['support_frac'] is not None :
        support = support.expand_region_by(solid_unit_expanded > 0.1, config['sample']['support_frac'])
    else :
        support = solid_unit_expanded > (solid_unit_expanded.min() + 1.0e-5)
    
    # add a beamstop
    if config['detector']['beamstop'] is not None :
        beamstop = circle.make_beamstop(diff.shape, config['detector']['beamstop'])
        diff    *= beamstop
    else :
        beamstop = np.ones_like(diff, dtype=np.bool)

    return diff, beamstop, background_circle, edges, support, solid_unit_expanded


def interp_3d(array, shapeout):
    from scipy.interpolate import griddata
    ijk = np.indices(array.shape)
    
    points = np.zeros((array.size, 3), dtype=np.float)
    points[:, 0] = ijk[0].ravel()
    points[:, 1] = ijk[1].ravel()
    points[:, 2] = ijk[2].ravel()
    values = array.astype(np.float).ravel()

    gridout  = np.mgrid[0: array.shape[0]-1: shapeout[0]*1j, \
                        0: array.shape[1]-1: shapeout[1]*1j, \
                        0: array.shape[2]-1: shapeout[2]*1j]
    arrayout = griddata(points, values, (gridout[0], gridout[1], gridout[2]), method='nearest')
    return arrayout
    

def make_3D_duck(shape = (12, 25, 30)):
    fnam = os.path.dirname(os.path.realpath(__file__))
    fnam = os.path.join(fnam, 'duck_300_211_8bit.raw')
    # call in a low res 2d duck image
    duck = np.fromfile(fnam, dtype=np.int8).reshape((211, 300))
    
    # convert to bool
    duck = duck < 50

    # make a 3d volume
    duck3d = np.zeros( (100,) + duck.shape , dtype=np.bool)

    # loop over the third dimension with an expanding circle
    i, j = np.mgrid[0 :duck.shape[0], 0 :duck.shape[1]]

    origin = [150, 150]

    r = np.sqrt( ((i-origin[0])**2 + (j-origin[1])**2).astype(np.float) )

    rs = range(50) + range(50, 0, -1)
    rs = np.array(rs) * 200 / 50.
    
    circle = lambda ri : r < ri
    
    for z in range(duck3d.shape[0]):
        duck3d[z, :, :] = circle(rs[z]) * duck

    # now interpolate the duck onto the required grid
    duck3d = interp_3d(duck3d, shape)

    # get rid of the crap
    duck3d[np.where(duck3d < 0.1)] = 0.0
    return duck3d
        

if __name__ == '__main__':
    duck3d = make_3D_duck()
