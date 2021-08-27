#!/usr/bin/env python3
import argparse
import sys

description = "Phase a far-field diffraction volume using iterative projection algorithms."
parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--pipe_through', action='store_true', \
                    help="pipe input to output before display")
parser.add_argument('-a', '--accumulate', action='store_true', \
                    help="Accumulate named data for display (as an image stack)")
parser.add_argument('-n', '--name', type=str, \
                    help="Only show datasets with the key 'name'")
parser.add_argument('-i', '--input', type=argparse.FileType('rb'), default=sys.stdin.buffer, \
                    help="Python pickle file containing data")
parser.add_argument('-o', '--output', type=argparse.FileType('wb'), default=sys.stdout.buffer, \
                    help="Python pickle output file if 'pipe_through' is True")
args = parser.parse_args()


# read data from input pipe 
# and display in pyqt5 gui running in another thread
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import pyqtgraph as pg 
import numpy as np
import pickle
import signal


class Get_piped_data(QObject):
    finished = pyqtSignal()
    data_recieved = pyqtSignal(str)
    
    data = {}
    
    def recurse_dict(self, d):
        for name, value in d.items():
            if isinstance(value, dict):
                self.recurse_dict(value)
            elif args.name is None or args.name == name :
                # accumulate 2D arrays (very specific...)
                if args.accumulate: 
                    if name in self.data and \
                           isinstance(self.data[name], np.ndarray) and \
                           len(self.data[name].shape) == 3:
                        self.data[name] = np.append(self.data[name], value[None, ...], axis=0)
                    else :
                        self.data[name] = value[None, ...]
                else :
                    self.data[name] = value
                
                self.data_recieved.emit(name)
    
    def run(self):
        """Long-running task."""
        while True :
            try :
                package = pickle.load(args.input)

                if args.pipe_through : 
                    pickle.dump(package, args.output)
                    sys.stdout.flush()
                
                # if data is a dictionary then recursively 
                # iterate over key, value pairs
                if isinstance(package, dict) : 
                    self.recurse_dict(package)
                
                elif args.name is None :
                    name = 'unamed data'
                    self.data[name] = package
                    self.data_recieved.emit(name)
                
            except EOFError :
                break
            
            except Exception as e :
                print(e, file=sys.stderr)
        self.finished.emit() 
        

class Main():
    def __init__(self):
        
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Get_piped_data()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        # Step 5: Connect signals and slots
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.data_recieved.connect(self.show_data)
        # Step 6: Start the thread
        self.thread.start()
        self.plots = {}
        
 
    def show_data(self, name):
        if isinstance(self.worker.data[name], np.ndarray): 
            if len(self.worker.data[name].shape) in (2,3):
                if name not in self.plots :
                    self.plots[name] = pg.ImageView(view = pg.PlotItem(title=name))
                    self.plots[name].ui.menuBtn.hide()
                    self.plots[name].ui.roiBtn.hide()
                    self.plots[name].show()
                
                # convert bool to uint8 for display
                dtype = self.worker.data[name].real.dtype
                if dtype == bool : 
                    dtype = np.uint8
                
                self.plots[name].setImage(self.worker.data[name].real.astype(dtype))

if __name__ == '__main__':
    # allow Control-C
    signal.signal(signal.SIGINT, signal.SIG_DFL) 
    
    # launch a GUI that will process incomming data
    app = QApplication([])
    
    m = Main()
    
    app.exec_()
