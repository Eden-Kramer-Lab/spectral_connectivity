#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='spectral_connectivity',
    version='0.1.0.dev0',
    license='GPL-3.0',
    description=('Frequency domain functional and directed'
                 'connectivity analysis tools for electrophysiological'
                 'data'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/spectral_connectivity',
    packages=find_packages(),
)
