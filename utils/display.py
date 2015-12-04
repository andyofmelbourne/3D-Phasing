import pyqtgraph as pg
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
import signal
import sys

import io_utils

def show_vol(map_3d):
    import pyqtgraph.opengl as gl
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)
    ex  = Show_vol(map_3d)
    sys.exit(app.exec_())

class Show_vol(QtGui.QWidget):
    def __init__(self, map_3d):

        super(Show_vol, self).__init__()
        # 3D plot for psi
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.show()

        # layout
        vlayout = QtGui.QVBoxLayout() 
        vlayout.addWidget(self.w)

        data = map_3d
        d = np.empty(data.shape + (4,), dtype=np.ubyte)

        # white scale
        dis   = 255. #(data.astype(np.float) * (255./data.max())).astype(np.ubyte)
        alpha = (data.astype(np.float) * (255./data.max())).astype(np.ubyte)

        d[..., 0] = dis
        d[..., 1] = dis
        d[..., 2] = dis
        d[..., 3] = alpha #((data/data.max())**2 * 255.).astype(np.ubyte)

        # show the x-y-z axis
        d[:, 0, 0] = [255,0,0,100]
        d[0, :, 0] = [0,255,0,100]
        d[0, 0, :] = [0,0,255,100] 
        self.v = gl.GLVolumeItem(d)
        self.v.translate(-data.shape[0]/2,-data.shape[1]/2,-data.shape[2]/2)
        self.w.addItem(self.v)
        ax = gl.GLAxisItem()
        self.w.addItem(ax)

        self.setLayout(vlayout)
        self.resize(800,800)
        self.show()


class Application():

    def __init__(self, diff, diff_ret, support, support_ret, \
                 good_pix, solid_unit, solid_unit_ret,       \
                 emod, efid):
        
        solid_unit_ret = np.fft.ifftshift(solid_unit_ret.real)
        duck_plots = (np.sum(solid_unit_ret, axis=0),\
                      np.sum(solid_unit_ret, axis=1),\
                      np.sum(solid_unit_ret, axis=2))
        duck_plots = np.hstack(duck_plots)

        support_ret   = np.fft.ifftshift(support_ret)
        support_plots = (np.sum(support_ret, axis=0),\
                         np.sum(support_ret, axis=1),\
                         np.sum(support_ret, axis=2))
        support_plots = np.hstack(support_plots)

        
        diff_ret_plots = np.hstack((np.fft.ifftshift(diff_ret[0, :, :]), \
                                np.fft.ifftshift(diff_ret[:, 0, :]), \
                                np.fft.ifftshift(diff_ret[:, :, 0])))
        diff_ret_plots = diff_ret_plots**0.2

        diff_plots = np.hstack((np.fft.ifftshift(diff[0, :, :]), \
                                np.fft.ifftshift(diff[:, 0, :]), \
                                np.fft.ifftshift(diff[:, :, 0])))
        diff_plots = diff_plots**0.2
        
        # Always start by initializing Qt (only once per application)
        app = QtGui.QApplication([])

        # Define a top-level widget to hold everything
        w = QtGui.QWidget()

        # 2D projection images for the sample
        self.duck_plots = pg.ImageView()

        # 2D projection images for the sample support
        self.support_plots = pg.ImageView()

        # 2D slices for the diffraction volume
        self.diff_plots = pg.ImageView()

        # 2D slices for the retrieved diffraction volume
        self.diff_ret_plots = pg.ImageView()

        # line plots of the error metrics
        self.plot_emod = pg.PlotWidget()
        self.plot_efid = pg.PlotWidget()
         
        Vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical) 

        # ducks
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        Hsplitter.addWidget(self.duck_plots)
        Hsplitter.addWidget(self.support_plots)
        Vsplitter.addWidget(Hsplitter)
        
        # diffs
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        Hsplitter.addWidget(self.diff_plots)
        Hsplitter.addWidget(self.diff_ret_plots)
        Vsplitter.addWidget(Hsplitter)

        # errors
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        Hsplitter.addWidget(self.plot_emod)
        Hsplitter.addWidget(self.plot_efid)
        Vsplitter.addWidget(Hsplitter)
        
        hlayout_tot = QtGui.QHBoxLayout()
        hlayout_tot.addWidget(Vsplitter)

        w.setLayout(hlayout_tot)

        self.duck_plots.setImage(duck_plots.T)
        self.support_plots.setImage(support_plots.T)
        self.diff_plots.setImage(diff_plots.T)
        self.diff_ret_plots.setImage(diff_ret_plots.T)
        self.plot_emod.plot(emod)
        self.plot_emod.setTitle('Modulus error l2norm:')
        self.plot_efid.plot(efid)
        self.plot_efid.setTitle('Fidelity error l2norm:')
        
        ## Display the widget as a new window
        w.show()

        ## Start the Qt event loop
        app.exec_()
        

class Show_input():

    def __init__(self, diff, support, good_pix, solid_unit):
        
        solid_unit = np.fft.ifftshift(solid_unit.real)
        duck_plots = (np.sum(solid_unit, axis=0),\
                      np.sum(solid_unit, axis=1),\
                      np.sum(solid_unit, axis=2))

        duck_plots = np.hstack(duck_plots)
        
        diff_plots = np.hstack((np.fft.ifftshift(diff[0, :, :]), \
                                np.fft.ifftshift(diff[:, 0, :]), \
                                np.fft.ifftshift(diff[:, :, 0])))
        diff_plots = diff_plots**0.2
        
        # Always start by initializing Qt (only once per application)
        app = QtGui.QApplication([])

        # Define a top-level widget to hold everything
        w = QtGui.QWidget()

        # 2D projection images for the sample
        self.duck_plots = pg.ImageView()

        # 2D slices for the diffraction volume
        self.diff_plots = pg.ImageView()

        # line plots of the error metrics
        #self.plot_emod = pg.PlotWidget()
        #self.plot_efid = pg.PlotWidget()
         
        Vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical) 

        # ducks
        Vsplitter.addWidget(self.duck_plots)
        
        # diffs
        Vsplitter.addWidget(self.diff_plots)
        
        # errors
        #Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        #Hsplitter.addWidget(self.plot_emod)
        #Hsplitter.addWidget(self.plot_efid)
        #Vsplitter.addWidget(Hsplitter)
        
        hlayout_tot = QtGui.QHBoxLayout()
        hlayout_tot.addWidget(Vsplitter)

        w.setLayout(hlayout_tot)

        self.duck_plots.setImage(duck_plots.T)
        self.diff_plots.setImage(diff_plots.T)
        #self.plot_emod.plot(emod)
        #self.plot_emod.setTitle('Modulus error l2norm:')
        #self.plot_efid.plot(efid)
        #self.plot_efid.setTitle('Fidelity error l2norm:')
        
        ## Display the widget as a new window
        print 'showing...'
        w.show()

        ## Start the Qt event loop
        app.exec_()
        
def parse_cmdline_args():
    import argparse
    import os
    parser = argparse.ArgumentParser(prog = 'display.py', description='display the contents of output.h5 in a GUI')
    parser.add_argument('path', type=str, \
                        help="path to output.h5 file")
    parser.add_argument('inout', type=str, \
                        help="'input' or 'output'. Display as input or output of phasing algorithm")
    args = parser.parse_args()

    # check that args.ini exists
    if not os.path.exists(args.path):
        raise NameError('output h5 file does not exist: ' + args.path)
    return args



if __name__ == '__main__':
    args = parse_cmdline_args()
    
    if args.inout == 'input':
        # read the h5 file 
        diff, support, good_pix, solid_unit, params = io_utils.read_input_h5(args.path)

        ex  = Show_input(diff, support, good_pix, solid_unit)
    
    elif args.inout == 'output':
        # read the h5 file 
        diff, diff_ret, support, support_ret, \
        good_pix, solid_unit, solid_unit_ret, \
        emod, efid                             = io_utils.read_output_h5(args.path)

        ex  = Application(diff, diff_ret, support, support_ret, \
                          good_pix, solid_unit, solid_unit_ret, \
                          emod, efid)
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)

    sys.exit(app.exec_())
