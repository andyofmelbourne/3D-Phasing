
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QSlider, QApplication
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QSize
import pyqtgraph as pg 
import numpy as np
import numbers

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance


class Default_1D(pg.PlotWidget):
    
    def __init__(self, name, data):
        super(Default_1D, self).__init__(title = name)
        self.show()

    def update_data(self, data):
        self.plotItem.clear()
        
        dtype = data.real.dtype
        
        # convert bool to uint8 for display
        if dtype == bool : 
            dtype = np.uint8
        
        self.plot(data.real.astype(dtype))
        
        QApplication.processEvents()

class Default_2D(pg.ImageView):
    
    def __init__(self, name, data):
        super(Default_2D, self).__init__(view = pg.PlotItem(title = name))
        
        self.ui.menuBtn.hide()
        self.ui.roiBtn.hide()
        self.show()

    def update_data(self, data):
        # convert bool to uint8 for display
        if data.dtype == bool : 
            t = data.astype(np.uint8)
        
        # convert complex to amp for display
        elif isinstance(data.ravel()[0], numbers.Complex):
            t = np.abs(data)
            
        else :
            t = data
        
        self.setImage(t, autoRange = False, autoLevels = False, autoHistogramRange = False)

Default_3D = Default_2D


class D1(Default_2D):
    """
    Crop and project 3D images to display as 2D
    """
    
    def __init__(self, name, data):
        super().__init__(name, data)
        
    def update_data(self, data):
        super().update_data(self.project(data))
        
    def project(self, data):
        return np.hstack([np.sum(data, axis=a) for a in range(len(data.shape))])

class D3(Default_2D):
    """
    Slice 3D images to display as 2D
    """
    
    def __init__(self, name, data):
        super().__init__(name, data)
        
    def update_data(self, data):
        super().update_data(self.project(data))
        
    def project(self, data):
        dims = range(len(data.shape))
        s = []
        for a in dims:
            t = [slice(None) for i in dims]
            t[a] = data.shape[a]//2
            s.append(t)
            
        return np.hstack([data[tuple(t)] for t in s])
    

import pyqtgraph.opengl as gl

class D2(QWidget):

    def __init__(self, name, data, parent = None):
        super(D2, self).__init__(parent)
        
        self.iso = Isosurface(name, data, parent=self, lvl=0.2)
        
        layout = QVBoxLayout()
        layout.addWidget(self.iso)
        
        self.sl = QSlider(Qt.Horizontal)
        self.sl.setMinimum(1)
        self.sl.setMaximum(9999)
        self.sl.setSingleStep(1)
        self.sl.setValue(2000)
        layout.addWidget(self.sl)
        self.sl.valueChanged.connect(self.lvlchange)
        
        self.setLayout(layout)
        self.setWindowTitle(name + ' isosurface')
        self.show()
        
    def lvlchange(self):
        self.iso.lvl = self.sl.value()/10000.
        self.iso.update_data()

    def update_data(self, data):
        self.iso.update_data(data)

    def sizeHint(self):
        return QSize(600, 900)

class Isosurface(gl.GLViewWidget):
    
    def __init__(self, name, data, lvl=0.2, parent=None):
        super(Isosurface, self).__init__(parent)

        self.data_init(data)
        
        self.setCameraPosition(distance=np.sqrt(np.sum(np.array(data.shape)**2)))
        
        # generate an isosurface from volumetric data
        verts, faces = pg.isosurface(self.data, lvl * self.max)
        
        # generate a mesh from verts and faces
        self.md = gl.MeshData(vertexes=verts, faces=faces)
        
        self.m1 = gl.GLMeshItem(meshdata=self.md, smooth=True, shader=self.make_shader2())
        self.m1.translate(-data.shape[0]/2, -data.shape[1]/2, -data.shape[2]/2)
        
        self.lvl = lvl
        self.addItem(self.m1)
        #self.show() 

    def data_init(self, data):
        if isinstance(data.ravel()[0], numbers.Complex):
            self.data = np.ascontiguousarray(np.abs(data))
        else:
            self.data = np.ascontiguousarray(data)
        
        self.max = np.max(self.data)
    
    def update_data(self, data = None):
        if data is not None :
            self.data_init(data) 
        
        verts, faces = pg.isosurface(self.data, self.lvl * self.max)
        
        self.m1.setMeshData(vertexes=verts, faces=faces)

    def make_shader2(self):
        # make my own pyqtgraph shader to move the light source to (1,1,1)
        from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader
        shader = ShaderProgram('shaded', [    
                    VertexShader(""" 
                        varying vec3 normal; 
                        void main() { 
                            // compute here for use in fragment shader 
                            normal = normalize(gl_NormalMatrix * gl_Normal); 
                            gl_FrontColor = gl_Color; 
                            gl_BackColor = gl_Color; 
                            gl_Position = ftransform(); 
                        } 
                    """), 
                    FragmentShader(""" 
                        varying vec3 normal; 
                        void main() { 
                            float p = dot(normal, normalize(vec3(1.0, 1.0, 1.0))); 
                            p = p < 0. ? 0. : p * 0.8; 
                            vec4 color = gl_Color; 
                            color.x = color.x * (0.2 + p); 
                            color.y = color.y * (0.2 + p); 
                            color.z = color.z * (0.2 + p); 
                            gl_FragColor = color; 
                        } 
                    """) 
                ]) 
        return shader
