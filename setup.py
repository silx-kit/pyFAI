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
from __future__ import with_statement, print_function
"""
Setup script for python Fast Azimuthal Integration
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/11/2013"
__status__ = "stable"


import os
import sys
import glob
import shutil
import platform
from os.path import join
from distutils.core import setup, Extension, Command
from numpy.distutils.misc_util import get_numpy_include_dirs
from distutils.sysconfig import get_python_lib
from distutils.command.install_data import install_data

################################################################################
# Check for Cython
################################################################################
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
if ("--no-cython" in sys.argv):
    CYTHON = False
    sys.argv.remove("--no-cython")
        

if CYTHON:
    cython_c_ext = ".pyx"
else:
    cython_c_ext = ".c"
    from distutils.command.build_ext import build_ext


def rewriteManifest(with_testimages=False):
    """
    Rewrite the "Manifest" file ... if needed

    @param with_testimages: include
    """
    base = os.path.dirname(os.path.abspath(__file__))
    manifest_file = join(base, "MANIFEST.in")
    if not os.path.isfile(manifest_file):
        print("MANIFEST file is missing !!!")
        return
    manifest = [i.strip() for i in open(manifest_file)]
    changed = False

    if with_testimages:
        testimages = ["test/testimages/" + i for i in os.listdir(join(base, "test", "testimages"))]
        for image in testimages:
            if image not in manifest:
                manifest.append("include " + image)
                changed = True
    else:
        for line in manifest[:]:
            if line.startswith("include test/testimages"):
                changed = True
                manifest.remove(line)
    if changed:
        with open(manifest_file, "w") as f:
            f.write(os.linesep.join(manifest))
        # remove MANIFEST: will be re generated !
        os.unlink(manifest_file[:-3])

if ("sdist" in sys.argv):
    if ("--with-testimages" in sys.argv):
        sys.argv.remove("--with-testimages")
        rewriteManifest(with_testimages=True)
    else:
        rewriteManifest(with_testimages=False)

# ###############################################################################
# pyFAI extensions
# ###############################################################################
cython_modules = ["histogram", "splitPixel", "splitBBox", "splitBBoxLUT",
                  "relabel", "bilinear", "_geometry", "reconstruct", "fastcrc", "_distortion"]
src = {}
for ext in cython_modules:
    src[ext] = join("src", ext + cython_c_ext)

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

fastcrc_dic = dict(name="fastcrc",
                        include_dirs=get_numpy_include_dirs(),
                        sources=[src['fastcrc'] , join("src", "crc32.c")],
#                        extra_compile_args=['-msse4.2'],
                        )
_distortion_dic = dict(name="_distortion",
                        include_dirs=get_numpy_include_dirs(),
                        sources=[src['_distortion'] ],
#                        extra_compile_args=['-msse4.2'],
                        extra_compile_args=['openmp'],
                        extra_link_args=['openmp'],

                        )


ext_modules = [histogram_dic, splitPixel_dic, splitBBox_dic, splitBBoxLUT_dic, relabel_dic,
               _geometry_dic, reconstruct_dic, bilinear_dic, fastcrc_dic, _distortion_dic]


if (os.name != "posix") or ("x86" not in platform.machine()):
    ext_modules.remove(fastcrc_dic)


# ###############################################################################
# scripts and data installation
# ###############################################################################
global installDir
installDir = "pyFAI"

data_files = [(installDir, glob.glob("openCL/*.cl")),
              (join(installDir, "gui"), glob.glob("gui/*.ui")),
              (join(installDir, "calibration"), glob.glob("calibration/*.D"))]

if sys.platform == "win32":
    # This is for mingw32/gomp
    if tuple.__itemsize__ == 4:
        data_files[0][1].append(join("dll", "pthreadGC2.dll"))
    root = os.path.dirname(os.path.abspath(__file__))
    tocopy_files = []
    script_files = []
    for i in os.listdir(join(root, "scripts")):
        if os.path.isfile(join(root, "scripts", i)):
            if i.endswith(".py"):
                script_files.append(join("scripts", i))
            else:
                tocopy_files.append(join("scripts", i))
    for i in tocopy_files:
        filein = join(root, i)
        if (filein + ".py") not in script_files:
            shutil.copyfile(filein, filein + ".py")
            script_files.append(filein + ".py")

else:
    script_files = glob.glob("scripts/*")

version = [eval(l.split("=")[1]) for l in open(join(os.path.dirname(
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

cmdclass = {}

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
cmdclass['build_ext'] = build_ext_pyFAI

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys, subprocess
        os.chdir("test")
        errno = subprocess.call([sys.executable, 'test_all.py'])
        if errno != 0:
            raise SystemExit(errno)
        else:
            os.chdir("..")
cmdclass['test'] = PyTest
#######################
# build_doc commandes #
#######################

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None

if sphinx:
    class build_doc(BuildDoc):

        def run(self):

            # make sure the python path is pointing to the newly built
            # code so that the documentation is built on this and not a
            # previously installed version

            build = self.get_finalized_command('build')
            sys.path.insert(0, os.path.abspath(build.build_lib))

            # Build the Users Guide in HTML and TeX format
            for builder in ('html', 'latex'):
                self.builder = builder
                self.builder_target_dir = join(self.build_dir, builder)
                self.mkpath(self.builder_target_dir)
                builder_index = 'index_{0}.txt'.format(builder)
                BuildDoc.run(self)
            sys.path.pop(0)
    cmdclass['build_doc'] = build_doc

class smart_install_data(install_data):
    def run(self):
        install_cmd = self.get_finalized_command('install')
#        self.install_dir = join(getattr(install_cmd,'install_lib'), "data")
        self.install_dir = getattr(install_cmd, 'install_lib')
        print("DATA to be installed in %s" % self.install_dir)
        global installDir
        installDir = join(self.install_dir, installDir)
        return install_data.run(self)
cmdclass['install_data'] = smart_install_data


setup(name='pyFAI',
      version=version,
      author="Jérôme Kieffer (python), \
      Peter Boesecke (geometry), Manuel Sanchez del Rio (algorithm), Vicente Armando Sole (algorithm), \
      Dimitris Karkoulis (GPU), Jon Wright (adaptations) and Frederic-Emmanuel Picca",
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
      cmdclass=cmdclass,
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
sys.path.insert(0, os.path.dirname(installDir))
#print installDir
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

