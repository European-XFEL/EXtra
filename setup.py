#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


setup(
    package_dir={'': 'src'},
    ext_modules=cythonize([
        Extension('extra.utils.ftd',
                  ['src/extra/utils/ftd.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=[
                      '-g0', '-O3', '-fpic', '-frename-registers',
                      '-ftree-vectorize']),
    ], language_level=3),
)
