# set-up watchdog to look for new files

# load the latest psi, support and error

# look at the xy, yz, xz axis of psi and support

# plot error

#!/usr/bin/env python

import numpy as np
import time
import sys, os
import ConfigParser
from utils import io_utils
import re
import signal

import fnmatch
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

psi_fnams     = []
support_fnams = []
mod_err_fnams = []
sup_err_fnams = []

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

class FnamEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        print event.src_path, event.is_directory
        if event.is_directory == False :
            if event.src_path[-3 :] == 'bin':
                fnam = os.path.basename(event.src_path)
                
                if fnam[: 3] == 'psi':
                    global psi_fnams
                    psi_fnams.append(str(event.src_path))
                    
                elif fnam[: 7] == 'support':
                    global support_fnams
                    support_fnams.append(str(event.src_path))
                
                elif fnam[: 3] == 'mod':
                    global mod_err_fnams
                    mod_err_fnams.append(str(event.src_path))


def fill_current_file_list(directory = './'):
    global psi_fnams, support_fnams, mod_err_fnams
    psi_fnams, support_fnams, mod_err_fnams = [], [], []
    for dirname, dirnames, filenames in os.walk(directory):
        for fnam in fnmatch.filter(filenames, '*.bin'):
            fnam_abs = os.path.join(dirname, fnam)
            
            if fnam[: 3] == 'psi':
                psi_fnams.append(str(fnam_abs))
                sort_nicely(psi_fnams)
                
            elif fnam[: 7] == 'support':
                support_fnams.append(str(fnam_abs))
                sort_nicely(support_fnams)
            
            elif fnam[: 3] == 'mod':
                mod_err_fnams.append(str(fnam_abs))
                sort_nicely(mod_err_fnams)

            elif fnam[: 3] == 'sup':
                sup_err_fnams.append(str(fnam_abs))
                sort_nicely(sup_err_fnams)

class Application(QtGui.QWidget):

    def __init__(self, params):
        super(Application, self).__init__()

        self.params = params
        
        # fill the files dict --> global variable files
        fill_current_file_list(self.params['output']['dir'])
        
        # set auto colour range for the first images
        self.auto = True

        # set the current filenames
        self.current_psi_fnam     = None
        self.current_support_fnam = None
        self.current_mod_err_fnam = None
        self.current_sup_err_fnam = None
        
        # set-up the gui
        self.refresh_rate = 3000 # milli seconds
        self.initUI()
        self.init_timer()
    

    def initUI(self):
        # 3D plot for psi
        #self.plot_psi = pg.ImageView()
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.show()
        
        # 3D plot for the support
        #self.plot_support = pg.ImageView()
        self.w2 = gl.GLViewWidget()
        self.w2.opts['distance'] = 200
        self.w2.show()

        # line plot for the error metric
        self.plot_err = pg.PlotWidget()

        hlayouts = []

        # cspad image layout
        hlayouts.append(QtGui.QHBoxLayout())
        Hsplitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        #Hsplitter.addWidget(self.plot_psi)
        Hsplitter.addWidget(self.w)
        Hsplitter.addWidget(self.w2)
        #Hsplitter.addWidget(self.plot_support)
        # histogram plot layout
        Vsplitter = QtGui.QSplitter(QtCore.Qt.Vertical)
        Vsplitter.addWidget(Hsplitter)
        Vsplitter.addWidget(self.plot_err)

        hlayouts[-1].addWidget(Vsplitter)

        # stack everything vertically 
        vlayout = QtGui.QVBoxLayout()
        for hlayout in hlayouts :
            vlayout.addLayout(hlayout)
        
        # display the image
        self.updateDisplay()
        
        self.setLayout(vlayout)
        self.resize(800,800)
        self.show()
        
        
    def init_timer(self):
        """ Update the image every milli_secs. """
        self.refresh_timer = QtCore.QTimer()
        self.refresh_timer.timeout.connect(self.updateDisplay)
        self.refresh_timer.start(self.refresh_rate)
    
    def updateDisplay(self):
        update = False
        
        # fill the files dict --> global variable files
        print 'checking...'
        fill_current_file_list(self.params['output']['dir'])
        
        # set the new data (if any)
        global psi_fnams, support_fnams, mod_err_fnams, sup_err_fnams
        if len(psi_fnams) > 0 :
            if psi_fnams[-1] != self.current_psi_fnam :
                self.current_psi_fnam = psi_fnams[-1]
                update = True

                # load the new psi
                psi = io_utils.binary_in(self.current_psi_fnam)
        else :
            self.current_psi_fnam = None
            
        if len(support_fnams) > 0 :
            if support_fnams[-1] != self.current_support_fnam :
                self.current_support_fnam = support_fnams[-1]
                update = True

                # load the new support
                support = io_utils.binary_in(self.current_support_fnam)
        else :
            self.current_support_fnam = None
        
        if len(mod_err_fnams) > 0 :
            if mod_err_fnams[-1] != self.current_mod_err_fnam :
                self.current_mod_err_fnam = mod_err_fnams[-1]
                update = True

                # load the new mod_err
                mod_err = io_utils.binary_in(self.current_mod_err_fnam)
        else :
            self.current_mod_err_fnam = None

        if len(sup_err_fnams) > 0 :
            if sup_err_fnams[-1] != self.current_sup_err_fnam :
                self.current_sup_err_fnam = sup_err_fnams[-1]
                update = True

                # load the new mod_err
                sup_err = io_utils.binary_in(self.current_sup_err_fnam)
        else :
            self.current_sup_err_fnam = None
        
        # display the new data (if any) 
        if update :
            if not self.auto :
                self.w.removeItem(self.v)
                self.w2.removeItem(self.v2)
            else :
                self.auto = False
            #self.plot_psi.setImage(np.abs(psi))
            #self.plot_psi.setCurrentIndex(psi.shape[0]/2)
            data = np.abs(psi)
            d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
            d2[..., 0] = 255
            d2[..., 1] = 100
            d2[..., 2] = (data.astype(np.float) * (255./data.max())).astype(np.ubyte)
            d2[..., 3] = ((data/data.max())**2 * 255.).astype(np.ubyte)
            d2[:, 0, 0] = [255,0,0,100]
            d2[0, :, 0] = [0,255,0,100]
            d2[0, 0, :] = [0,0,255,100] 
            self.v = gl.GLVolumeItem(d2)
            self.v.translate(-data.shape[0]/2,-data.shape[1]/2,-data.shape[2]/2)
            self.w.addItem(self.v)
            ax = gl.GLAxisItem()
            self.w.addItem(ax)
            
            #self.plot_support.setImage(support)
            #self.plot_support.setCurrentIndex(psi.shape[0]/2)
            data = support
            d3 = np.empty(data.shape + (4,), dtype=np.ubyte)
            d3[..., 0] = 255
            d3[..., 1] = 255
            d3[..., 2] = 255 #(data.astype(np.float) * (255./data.max())).astype(np.ubyte)
            d3[..., 3] = ((data.astype(np.float)/data.max())**2 * 50.).astype(np.ubyte)
            d3[:, 0, 0] = [255,0,0,100]
            d3[0, :, 0] = [0,255,0,100]
            d3[0, 0, :] = [0,0,255,100] 
            self.v2 = gl.GLVolumeItem(d3)
            self.v2.translate(-data.shape[0]/2,-data.shape[1]/2,-data.shape[2]/2)
            self.w2.addItem(self.v2)
            ax2 = gl.GLAxisItem()
            self.w2.addItem(ax2)
            
            self.plot_err.clear()
            self.plot_err.setTitle('Modulus + support (red) projection error')
            self.plot_err.plot(mod_err)
            self.plot_err.plot(sup_err, pen = (100, 0, 0))


if __name__ == '__main__':
    config = ConfigParser.ConfigParser()
    config.read(sys.argv[1])
    params = io_utils.parse_parameters(config)
    
    signal.signal(signal.SIGINT, signal.SIG_DFL)    # allow Control-C
    app = QtGui.QApplication(sys.argv)
    ex  = Application(params)
    sys.exit(app.exec_())
    """
    # display the support as a volumetric plot
    from pyqtgraph.Qt import QtCore, QtGui
    import pyqtgraph.opengl as gl
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 200
    w.show()
    w.setWindowTitle('pyqtgraph example: GLVolumeItem')
    
    #data = io_utils.binary_in('example/output/support_1000_128x128x128_int8.bin')
    data = np.abs(io_utils.binary_in('example/output/psi_1000_128x128x128_complex128.bin'))

    d2 = np.empty(data.shape + (4,), dtype=np.ubyte)
    d2[..., 0] = 255
    d2[..., 1] = 100
    d2[..., 2] = (data.astype(np.float) * (255./data.max())).astype(np.ubyte)
    d2[..., 3] = ((data/data.max())**2 * 150.).astype(np.ubyte)

    d2[:, 0, 0] = [255,0,0,100]
    d2[0, :, 0] = [0,255,0,100]
    d2[0, 0, :] = [0,0,255,100] 

    v = gl.GLVolumeItem(d2)
    v.translate(-data.shape[0]/2,-data.shape[1]/2,-data.shape[2]/2)
    w.addItem(v)

    ax = gl.GLAxisItem()
    w.addItem(ax)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
    """
