from setuptools import setup, find_packages
from pathlib import Path

setup(
    name                 = "phasing",
    version              = "2021.0",
    packages             = find_packages(),
    scripts              = [str(p) for p in Path('.').glob('phasing/bin/*.py')]
    )
