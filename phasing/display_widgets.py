
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout
from PyQt5.QtCore import QObject, pyqtSignal, QThread
import pyqtgraph as pg 
import numpy as np

class Default_1D(pg.PlotWidget):
    
    def __init__(self, name, data):
        super(Default_2D, self).__init__(title = name)
        self.show()

    def update(self, data):
        dtype = data.real.dtype
        
        # convert bool to uint8 for display
        if dtype == bool : 
            dtype = np.uint8
        
        self.plot(data.real.astype(dtype))

class Default_2D(pg.ImageView):
    
    def __init__(self, name, data):
        super(Default_2D, self).__init__(view = pg.PlotItem(title = name))
        
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.show()

    def update(self, data):
        dtype = data.real.dtype
        
        # convert bool to uint8 for display
        if dtype == bool : 
            dtype = np.uint8
        
        self.setImage(data.real.astype(dtype))

Default_3D = Default_2D


class D1(Default_2D):
    """
    Crop and project 3D images to display as 2D
    """
    
    def __init__(self, name, data):
        super().__init__(name, data)
        
    def update(self, data):
        super().update(self.project(data))
        
    def project(self, data):
        return np.vstack([np.sum(data, axis=a) for a in range(len(data.shape))])
    
