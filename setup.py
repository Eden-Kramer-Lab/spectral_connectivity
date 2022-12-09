#!/usr/bin/env python3

import codecs
import os.path

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "numpy >= 1.11",
    "scipy",
    "xarray",
    "matplotlib",
]
TESTS_REQUIRE = ["pytest >= 2.7.1", "nitime"]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="spectral_connectivity",
    version=get_version("spectral_connectivity/__init__.py"),
    license="GPL-3.0",
    description=(
        "Frequency domain functional and directed"
        "connectivity analysis tools for electrophysiological"
        "data"
    ),
    author="Eric Denovellis",
    author_email="eric.denovellis@ucsf.edu",
    url="https://github.com/Eden-Kramer-Lab/spectral_connectivity",
    # long_description=open("README.md").read(),
    long_description_content_type="text/x-rst",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
    ],
    platforms="any",
    keywords=(
        "python neuroscience electrophysiology "
        "multitaper spectrogram frequency-domain"
    ),
    python_requires=">=3",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    project_urls={
        "Documentation": "https://spectral-connectivity.readthedocs.io/en/latest/",
        "Bug Reports": "https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues",
        "Source": "https://github.com/Eden-Kramer-Lab/spectral_connectivity",
    },
)
