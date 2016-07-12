# 3D-Phasing
Basic python code for phasing three-dimensional diffraction volumes

To install run:
```
$ git clone https://github.com/andyofmelbourne/3D-Phasing.git ~/.local/lib/python2.7/site-packages/phasing_3d
```

## Example command line
A basic example with a 3D duck:
```bash
$ cp -r ~/.local/lib/python2.7/site-packages/phasing_3d/examples .
$ cp ~/.local/lib/python2.7/site-packages/phasing_3d/reconstruct.py .
$ python reconstruct.py examples/duck/config.ini
```
See also: config_background.ini (radial background retrieval) and config_voxel_number_support.ini (unknown support) in examples/duck/

When complete, you may display the output:
```
$ python ~/.local/lib/python2.7/site-packages/phasing_3d/utils/display.py examples/duck/output.h5 output
```

## Example python 
Of course the above is all just padding around:
```python
import phasing_3d
import numpy as np
O = np.random.random((64,64,64))
S = np.zeros(O.shape, dtype=np.bool)
S[:16,:16,:16] = True
O *= S
I = np.abs(np.fft.fftn(O))**2
Oout, info = phasing_3d.DM(I, 100, support = S)
Oout, info = phasing_3d.ERA(I, 100, support = S, O = Oout)
```

Try it yourself, just copy the above text then:
```
$ ipython
>>> %paste
```
and watch the magic happen!
