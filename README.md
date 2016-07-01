# 3D-Phasing
Basic python code for phasing three-dimensional diffraction volumes

To install run:
```
$ git clone https://github.com/andyofmelbourne/3D-Phasing.git ~/.local/lib/python2.7/site-packages/
```

## Example
A basic example with a 3D duck:
```
$ cp -r ~/.local/lib/python2.7/site-packages/3D-Phasing/examples .
$ cp ~/.local/lib/python2.7/site-packages/3D-Phasing/reconstruct.py .
$ python reconstruct.py examples/duck/config.ini
```

When complete, you may display the output:
```
$ python ~/.local/lib/python2.7/site-packages/3D-Phasing/utils/display.py examples/duck/output.h5 output
```
