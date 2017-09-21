#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='spectral',
    version='0.1.0.dev0',
    description=('Frequency domain functional and directed'
                 'connectivity analysis tools for electrophysiological'
                 'data'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    packages=find_packages(),
)
