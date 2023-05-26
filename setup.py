#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from pathlib import Path
import re

import numpy as np
from Cython.Build import cythonize

from setuptools import setup, find_packages
from setuptools.extension import Extension


parent_path = Path(__file__).parent

setup(
    name='EXtra',
    version='1.0.0',
    description='European XFEL toolkit for research and analysis',
    long_description=(parent_path / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/European-XFEL/EXtra',
    author='European XFEL',
    author_email='da@xfel.eu',
    license='BSD-3-Clause',

    package_dir={'': 'src'},
    packages=find_packages('src'),

    ext_modules=cythonize([
        Extension('extra.staging.ftd',
                  ['src/extra/staging/ftd.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=[
                      '-g0', '-O3', '-fpic', '-frename-registers',
                      '-ftree-vectorize']),
    ]),

    python_requires='>=3.6',
    install_requires=[
        'extra_data',
        'extra_geom',
        'karabo_bridge',
        'euxfel_bunch_pattern'
    ],
    extras_require={
        'test': ['pytest',],
    },

    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Physics',
    ]
)
