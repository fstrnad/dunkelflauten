"""setup.py for connected extremes."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='conextremes',
    version='1.0',
    description=('Library for identifying and processing of dunkelflauten Extremes.'),
    author='Felix Strnad',
    author_email='felix.strnad@uni-tuebingen.de',
    packages=find_packages('dunkelflauten'),
)
