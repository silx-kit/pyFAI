#!/usr/bin/env python
# -*- coding: utf8 -*-
#
#    Project: Fast Azimuthal integration 
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
"""
Setup script for python Fast Azimuthal Integration
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/06/2012"
__status__ = "stable"

import os, sys, glob, shutil
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.sysconfig import get_python_lib
try:
    from Cython.Distutils import build_ext
except ImportError:
    build_ext = None

src = {}

if build_ext:
    ocl_azim = [os.path.join("openCL", i) for i in ("ocl_azim.pyx", "ocl_base.cpp", "ocl_tools/ocl_tools.cc", "ocl_tools/cLogger/cLogger.c", "ocl_xrpd1d_fullsplit.cpp")]
    for ext in ["histogram", "splitPixel", "splitBBox", "relabel", "bilinear"]:
        src[ext] = os.path.join("src", ext + ".pyx")
else:
    ocl_azim = [os.path.join("openCL", i) for i in ("ocl_azim.cpp", "ocl_base.cpp", "ocl_tools/ocl_tools.cc", "ocl_tools/cLogger/cLogger.c", "ocl_xrpd1d_fullsplit.cpp")]
    for ext in ["histogram", "splitPixel", "splitBBox", "relabel", "bilinear"]:
        src[ext] = os.path.join("src", ext + ".c")

installDir = os.path.join(get_python_lib(), "pyFAI")

hist_dic = dict(name="histogram",
                    include_dirs=get_numpy_include_dirs(),
                    sources=[src['histogram']],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'],
                    )

split_dic = dict(name="splitPixel",
                    include_dirs=get_numpy_include_dirs(),
                    sources=[src['splitPixel']],
#                    extra_compile_args=['-fopenmp'],
#                    extra_link_args=['-fopenmp'],
                    )

splitBBox_dic = dict(name="splitBBox",
                    include_dirs=get_numpy_include_dirs(),
                    sources=[src['splitBBox']],
#                    extra_compile_args=['-g'],
#                    extra_link_args=['-fopenmp'])
                    )
relabel_dic = dict(name="relabel",
                        include_dirs=get_numpy_include_dirs(),
                        sources=[src['relabel']])

bilinear_dic = dict(name="bilinear",
                        include_dirs=get_numpy_include_dirs(),
                        sources=[src['bilinear']])

ocl_azim_dict = dict(name="ocl_azim",
                    sources=ocl_azim,
                    include_dirs=["openCL","openCL/ocl_tools","openCL/ocl_tools/cLogger"] + get_numpy_include_dirs(),
                    language="c++",
                    libraries=["stdc++", "OpenCL"],
                    )

if sys.platform == "win32":
    data_files = [(installDir, [os.path.join("dll", "pthreadGC2.dll")])]
##    Remove OpenMP for windows
#    to_remove = ["extra_compile_args", "extra_link_args"]
#    for ext in [hist_dic, split_dic, splitBBox_dic, relabel_dic, bilinear_dic]:
#        for rem in to_remove:
#            if  rem in ext:
#                ext.pop(rem)
    root = os.path.dirname(os.path.abspath(__file__))
    tocopy_files = []
    script_files = []
    for i in os.listdir(os.path.join(root, "scripts")):
        if os.path.isfile(os.path.join(root, "scripts", i)):
            if i.endswith(".py"):
                script_files.append(os.path.join("scripts", i))
            else:
                tocopy_files.append(os.path.join("scripts", i))
    for i in tocopy_files:
        filein = os.path.join(root, i)
        if (filein + ".py") not in script_files:
            shutil.copyfile(filein, filein + ".py")
            script_files.append(filein + ".py")

else:
    data_files = []
    script_files = glob.glob("scripts/*")



version = [eval(l.split("=")[1]) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyFAI-src", "__init__.py")) if l.strip().startswith("version")][0]

setup(name='pyFAI',
      version=version,
      author="Jérôme Kieffer (python), Peter Boesecke (geometry), Manuel Sanchez del Rio (algorithm), Vicente Armando Sole (algorithm) and Dimitris Karkoulis (GPU ) """,
      author_email="jerome.kieffer@esrf.fr",
      description='Python implementation of fast azimuthal integration',
      url="http://forge.epn-campus.eu/azimuthal",
      download_url="http://forge.epn-campus.eu/projects/azimuthal/files",
      ext_package="pyFAI",
      scripts=script_files,
      ext_modules=[Extension(**hist_dic) ,
                   Extension(**relabel_dic),
                   Extension(**split_dic),
                   Extension(**splitBBox_dic),
                   Extension(**bilinear_dic),
                   Extension(**ocl_azim_dict)
                   ],
      packages=["pyFAI"],
      package_dir={"pyFAI": "pyFAI-src" },
#      data_files=data_files,
      test_suite="test",
      cmdclass={'build_ext': build_ext},
      data_files=[(installDir, [os.path.join("openCL", i) for i in ("ocl_azim_kernel_2.cl", "ocl_azim_kernel2d_2.cl")])]
      )

################################################################################
# Chech for Fabio to be present of the system
################################################################################
try:
    import fabio
except ImportError:
    print("pyFAI needs fabIO for all image reading and writing. This python module can be found on: \nhttp://sourceforge.net/projects/fable/files/fabio/0.0.7/")


################################################################################
# check if OpenMP modules, freshly installed can import    
################################################################################
pyFAI = None
sys.path.insert(0, installDir)
for loc in ["", ".", os.getcwd()]:
    if loc in sys.path:
        sys.path.pop(sys.path.index(loc))
for mod in sys.modules.copy():
    if mod.startswith("pyFAI"):
        sys.modules.pop(mod)
try:
    import pyFAI
except ImportError as E:
    print("Unable to import pyFAI: %s" % E)
else:
    print("PyFAI is installed in %s" % pyFAI.__file__)
    try:
        import pyFAI.histogram
    except ImportError as E:
        print("PyFAI.histogram failed to import. It is likely there is an OpenMP error: %s" % E)
    else:
        print("OpenMP libraries were found and pyFAI.histogram was successfully imported")

