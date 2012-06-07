#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Azimuthal integration 
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from distutils.core import  Extension
from Cython.Distutils import build_ext

# for numpy
from numpy.distutils.misc_util import get_numpy_include_dirs


hist_ext = Extension("histogram",
                    include_dirs=get_numpy_include_dirs(),
                    sources=['histogram.c'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])

#
#halfsplit_ext = Extension("halfSplitPixel",
#                    include_dirs=get_numpy_include_dirs(),
#                    sources=['halfSplitPixel.c'],
#                    extra_compile_args=['-fopenmp'],
#                    extra_link_args=['-fopenmp'])


split_ext = Extension("splitPixel",
                    include_dirs=get_numpy_include_dirs(),
                    sources=['splitPixel.c'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])


relabel_ext = Extension("relabel",
                        include_dirs=get_numpy_include_dirs(),
                        sources=['relabel.pyx'])

bilinear_ext = Extension("bilinear",
                        include_dirs=get_numpy_include_dirs(),
                        sources=['bilinear.pyx'])
rebin_ext = Extension("fastrebin",
                        include_dirs=get_numpy_include_dirs(),
                        sources=['fastrebin.pyx', "slist.c"])

setup(name='histogram',
      version="0.3.0",
      author="Jerome Kieffer",
      author_email="jerome.kieffer@esrf.eu",
      description='test for azim int',
      ext_modules=[hist_ext, relabel_ext, split_ext, bilinear_ext, rebin_ext],
      cmdclass={'build_ext': build_ext},
      )
