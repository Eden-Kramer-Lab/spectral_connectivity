#!/usr/bin/env python3

from setuptools import setup, find_packages

INSTALL_REQUIRES = ['numpy >= 1.11', 'pandas >= 0.18.0', 'scipy', 'xarray',
                    'matplotlib']
TESTS_REQUIRE = ['pytest >= 2.7.1', 'nitime']

setup(
    name='spectral_connectivity',
    version='0.2.0.dev0',
    license='GPL-3.0',
    description=('Frequency domain functional and directed'
                 'connectivity analysis tools for electrophysiological'
                 'data'),
    author='Eric Denovellis',
    author_email='edeno@bu.edu',
    url='https://github.com/Eden-Kramer-Lab/spectral_connectivity',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
)
