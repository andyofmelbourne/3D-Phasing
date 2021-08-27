from setuptools import setup, find_packages
from pathlib import Path

scripts = ['phase.py', 'merge.py', 'pipe_h5.py', 'display.py', 'make_diffraction_volume.py', 'make_noisy_diffraction_volume.py', 'electron_density_from_pdb.py']

setup(
    name                 = "phasing",
    version              = "2021.0",
    packages             = find_packages(),
    scripts              = ['phasing/' + s for s in scripts]
    )
