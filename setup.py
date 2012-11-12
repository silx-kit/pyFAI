# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
__date__ = "12/11/2012"
__status__ = "stable"


import os, sys, glob, shutil, ConfigParser
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.sysconfig import get_python_lib

# ###############################################################################
# Check for Cython
# ###############################################################################
try:
    from Cython.Distutils import build_ext
    CYTHON = True
except ImportError:
    CYTHON = False
if CYTHON:
    try:
        import Cython.Compiler.Version
    except ImportError:
        CYTHON = False
    else:
        if Cython.Compiler.Version.version < "0.17":
            CYTHON = False

if CYTHON:
    cython_c_ext = ".pyx"
else:
    cython_c_ext = ".c"
    from distutils.command.build_ext import build_ext

# ###############################################################################
# check for OpenCL
# ###############################################################################
#
## temporary until pyopencl is used
#if "--without-opencl" in sys.argv:
#    OPENCL = None
#    sys.argv.remove('--without-opencl')
#else:
#    print("WARNING Compiling also the OpenCL extensions, \
#        add the --without-opencl option to skip this compilation")
#    OPENCL = True
#if OPENCL:
#    OCLINC = []
#    OCLLIBDIR = []
#    configparser = ConfigParser.ConfigParser()
#    configparser.read([os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                    "setup.cfg")])
#    if "OpenCL" in configparser.sections():
#        for item in configparser.items("OpenCL"):
#            if item[0] == "include-dirs":
#                OCLINC += item[1].split(os.pathsep)
#            elif item[0] == "library-dirs":
#                OCLLIBDIR += item[1].split(os.pathsep)



# ###############################################################################
# pyFAI extensions
# ###############################################################################
cython_modules = ["histogram", "splitPixel", "splitBBox", "splitBBoxLUT",
                  "relabel", "bilinear", "_geometry", "reconstruct"]
src = {}
for ext in cython_modules:
    src[ext] = os.path.join("src", ext + cython_c_ext)

_geometry_dic = dict(name="_geometry",
                     include_dirs=get_numpy_include_dirs(),
                     sources=[src['_geometry']],
                     extra_compile_args=['openmp'],
#                    extra_compile_args=['-g'],
                     extra_link_args=['openmp'])

reconstruct_dic = dict(name="reconstruct",
                       include_dirs=get_numpy_include_dirs(),
                       sources=[src['reconstruct']],
                       extra_compile_args=['openmp'],
#                      extra_compile_args=['-g'],
                       extra_link_args=['openmp'])

histogram_dic = dict(name="histogram",
                include_dirs=get_numpy_include_dirs(),
                sources=[src['histogram']],
                extra_compile_args=['openmp'],
                extra_link_args=['openmp'],
                )

splitPixel_dic = dict(name="splitPixel",
                 include_dirs=get_numpy_include_dirs(),
                 sources=[src['splitPixel']],
#                extra_compile_args=['-fopenmp'],
#                extra_link_args=['-fopenmp'],
                 )

splitBBox_dic = dict(name="splitBBox",
                     include_dirs=get_numpy_include_dirs(),
                     sources=[src['splitBBox']],
#                    extra_compile_args=['-g'],
#                    extra_link_args=['-fopenmp'])
                     )
splitBBoxLUT_dic = dict(name="splitBBoxLUT",
                        include_dirs=get_numpy_include_dirs(),
                        sources=[src['splitBBoxLUT']],
#                       extra_compile_args=['-g'],
                        extra_compile_args=['openmp'],
                        extra_link_args=['openmp'],
                        )

relabel_dic = dict(name="relabel",
                   include_dirs=get_numpy_include_dirs(),
                   sources=[src['relabel']])

bilinear_dic = dict(name="bilinear",
                    include_dirs=get_numpy_include_dirs(),
                    sources=[src['bilinear']])

ext_modules = [histogram_dic, splitPixel_dic, splitBBox_dic, splitBBoxLUT_dic, relabel_dic,
               _geometry_dic, reconstruct_dic, bilinear_dic]


#if OPENCL:
#    ocl_src = [os.path.join(*(pp.split("/"))) for pp in ("ocl_base.cpp",
#        "ocl_tools/ocl_tools.cc", "ocl_tools/ocl_tools_extended.cc",
#        "ocl_tools/cLogger/cLogger.c", "ocl_xrpd1d_fullsplit.cpp")]
#    if CYTHON:
#        ocl_src.append("ocl_azim.pyx")
#    else:
#        ocl_src.append("ocl_azim.cpp")
#    ocl_azim = [os.path.join("openCL", i) for i in  ocl_src]
#    openCL = OCLINC
#    j = ""
#    for i in "openCL/ocl_tools/cLogger".split("/"):
#        j = os.path.join(j, i)
#        openCL.insert(0, j)
#    ocl_azim_dict = dict(name="ocl_azim",
#                     sources=ocl_azim,
#                     include_dirs=openCL + get_numpy_include_dirs(),
#                     library_dirs=OCLLIBDIR,
#                     language="c++",
#                     libraries=[ "stdc++", "OpenCL"]  # "stdc++"
#                     )
#    ext_modules.append(ocl_azim_dict)

# ###############################################################################
# scripts and data installation
# ###############################################################################

installDir = os.path.join(get_python_lib(), "pyFAI")

if sys.platform == "win32":
    # This is for mingw32/gomp?
    data_files = [(installDir, [os.path.join("dll", "pthreadGC2.dll")])]
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


data_files += [(installDir, [os.path.join('openCL', o) for o in [
      "ocl_azim_kernel_2.cl", "ocl_azim_kernel2d_2.cl", "ocl_azim_LUT.cl"]])]

version = [eval(l.split("=")[1]) for l in open(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "pyFAI-src", "__init__.py"))
    if l.strip().startswith("version")][0]


# We subclass the build_ext class in order to handle compiler flags
# for openmp and opencl etc in a cross platform way
translator = {
        # Compiler
            # name, compileflag, linkflag
        'msvc' : {
            'openmp' : ('/openmp', ' '),
            'debug'  : ('/Zi', ' '),
            'OpenCL' : 'OpenCL',
            },
        'mingw32':{
            'openmp' : ('-fopenmp', '-fopenmp'),
            'debug'  : ('-g', '-g'),
            'stdc++' : 'stdc++',
            'OpenCL' : 'OpenCL'
            },
        'default':{
            'openmp' : ('-fopenmp', '-fopenmp'),
            'debug'  : ('-g', '-g'),
            'stdc++' : 'stdc++',
            'OpenCL' : 'OpenCL'
            }
        }

class build_ext_pyFAI(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type in translator:
            trans = translator[self.compiler.compiler_type]
        else:
            trans = translator['default']

        for e in self.extensions:
            e.extra_compile_args = [ trans[a][0] if a in trans else a
                                    for a in e.extra_compile_args]
            e.extra_link_args = [ trans[a][1] if a in trans else a
                                 for a in e.extra_link_args]
            e.libraries = filter(None, [ trans[a] if a in trans else None
                                        for a in e.libraries])

            # If you are confused look here:
            # print e, e.libraries
            # print e.extra_compile_args
            # print e.extra_link_args
        build_ext.build_extensions(self)



setup(name='pyFAI',
      version=version,
      author="Jérôme Kieffer (python), Peter Boesecke (geometry), Manuel Sanchez del Rio (algorithm), Vicente Armando Sole (algorithm), Dimitris Karkoulis (GPU) and Jon Wright (adaptations) ",
      author_email="jerome.kieffer@esrf.fr",
      description='Python implementation of fast azimuthal integration',
      url="http://forge.epn-campus.eu/azimuthal",
      download_url="http://forge.epn-campus.eu/projects/azimuthal/files",
      ext_package="pyFAI",
      scripts=script_files,
      ext_modules=[Extension(**dico) for dico in ext_modules],
      packages=["pyFAI"],
      package_dir={"pyFAI": "pyFAI-src" },
      test_suite="test",
      cmdclass={'build_ext': build_ext_pyFAI},
      data_files=data_files
      )

# ###############################################################################
# Check for Fabio to be present of the system
# ###############################################################################
try:
    import fabio
except ImportError:
    print("""pyFAI needs fabIO for all image reading and writing.
This python module can be found on:
http://sourceforge.net/projects/fable/files/fabio""")

try:
    import pyopencl
except ImportError:
    print("""pyFAI can use pyopencl to run on parallel accelerators like GPU; this is an optional dependency.
This python module can be found on:
http://pypi.python.org/pypi/pyopencl
""")


# ###############################################################################
# check if OpenMP modules, freshly installed can import
# ###############################################################################
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
    print pyFAI.__file__
except ImportError as E:
    print("Unable to import pyFAI: %s" % E)
else:
    print("PyFAI is installed in %s" % pyFAI.__file__)
    try:
        import pyFAI.histogram
        print  pyFAI.histogram.__file__
    except ImportError as E:
        print("PyFAI.histogram failed to import. It is likely there is an OpenMP error: %s" % E)
    else:
        print("OpenMP libraries were found and pyFAI.histogram was successfully imported")

