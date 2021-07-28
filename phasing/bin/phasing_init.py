#!/usr/bin/env python

# copy example files from the repo to a local directory

from pathlib import Path
import os, shutil
import phasing

if __name__ == '__main__':
    # get repo directory
    phasing_dir = Path(phasing.__file__).parent.resolve().parent
    
    # get example directory
    example_dir = phasing_dir / 'examples/duck'
    
    # create a phasing example directory
    if not (Path.cwd() / 'example').exists() :
        shutil.copytree(example_dir, Path.cwd() / 'example')

    # copy reconstruction script also
    shutil.copy(phasing_dir / 'phasing/reconstruct.py', Path.cwd() / 'example')
