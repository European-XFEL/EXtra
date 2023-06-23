#!/usr/bin/env python

# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from pathlib import Path
import re

from setuptools import setup, find_packages


parent_path = Path(__file__).parent


with open(parent_path / 'src' / 'extra' / '__init__.py') as f:
    # As defined in PEP 440, Appendix B
    pattern = re.compile(r'^__version__ = \'([1-9][0-9]*!)?(0|[1-9][0-9]*)'
                         r'(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?'
                         r'(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?\'$')

    for line in f:
        m = pattern.search(line)

        if m is not None:
            version = line[m.start(2):m.end()-1]
            break
    else:
        raise RuntimeError('unable to find version string')


setup(
    name='EXtra',
    version=version,
    description='European XFEL toolkit for research and analysis',
    long_description=(parent_path / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/European-XFEL/EXtra',
    author='European XFEL',
    author_email='da@xfel.eu',
    license='BSD-3-Clause',

    package_dir={'': 'src'},
    packages=find_packages('src'),

    python_requires='>=3.9',
    install_requires=[
        'extra_data>=1.13',
        'extra_geom',
        'karabo_bridge',
        'euxfel_bunch_pattern'
    ],
    extras_require={
        'test': ['pytest',],
        'docs': [
            'black',
            'mkdocs-material',
            'mkdocstrings',
            'mkdocstrings-python',
            'pymdown-extensions'
          ],
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
