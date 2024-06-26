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

cython_ext_kw = dict(
    include_dirs=[np.get_include()],
    extra_compile_args=['-g0', '-O3', '-fpic', '-frename-registers',
                        '-ftree-vectorize']
)

setup(
    package_dir={'': 'src'},
    ext_modules=cythonize([
        Extension('extra.signal._ftd', ['src/extra/signal/_ftd.pyx'],
                  **cython_ext_kw),
        Extension('extra.signal._interpolation',
                  ['src/extra/signal/_interpolation.pyx'],
                  **cython_ext_kw),
        Extension('extra.components._adq', ['src/extra/components/_adq.pyx'],
                  **cython_ext_kw),
    ], language_level=3),
)
