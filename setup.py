"""setup.py for geoutils."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='dunkelflauten',
    version='1.0',
    description=('Library for analyzing dunkelflauten out of climate data.'),
    author='Felix Strnad',
    author_email='felix.strnad@uni-tuebingen.de',
    packages=find_packages('dunkelflauten'),
)
