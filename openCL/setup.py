#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Image Alignment
#
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jérôme Kieffer"
__copyright__ = "2012, ESRF"
__license__ = "LGPL"

import os
try:
    from distutils.core import setup
    from Cython.Distutils.extension import Extension
    from Cython.Distutils import build_ext
    ocl_azim = "ocl_azim.pyx"
except ImportError:
    from distutils.core import setup, Extension
    from distutils.command.build_ext import build_ext
    ocl_azim = "ocl_azim.cpp"
from numpy.distutils.misc_util import get_numpy_include_dirs

j = ""
openCL = []
for i in "openCL/ocl_tools/cLogger".split("/"):
    j = os.path.join(j, i)
    openCL.append(j)

ocl_azim_ext = Extension(name="ocl_azim",
                    sources=[ocl_azim, "ocl_base.cpp", "ocl_tools/ocl_tools.cc", "ocl_xrpd1d_fullsplit.cpp", "ocl_tools/cLogger/cLogger.c"],
                    include_dirs=get_numpy_include_dirs() + openCL,
                    language="c++",
                    libraries=["stdc++", "OpenCL"],
                    )


setup(name='ocl_azim',
      version="0.1.0",
      author="Jerome Kieffer",
      author_email="jerome.kieffer@esrf.eu",
      description='Cython wrapper for OpenCL implementation of azimuthal integration',
      ext_modules=[ocl_azim_ext],
      cmdclass={'build_ext': build_ext}
      )
