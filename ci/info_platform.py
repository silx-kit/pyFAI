#!/usr/bin/env python
# coding: utf-8
# ##########################################################################
#
# Copyright (C) 2015-2018 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################

import sys
import numpy
print("Python %s bits" % (tuple.__itemsize__ * 8))
print("       maxsize: %s\t maxunicode: %s" % (sys.maxsize, sys.maxunicode))
print(sys.version)
try:
    from distutils.sysconfig import get_config_vars
except ImportError:
    from sysconfig import get_config_vars
config = get_config_vars("CONFIG_ARGS")
try:
    print("Config :" + " ".join(config))
except Exception:
    print("Config : None")
print("")
print("Numpy %s" % numpy.version.version)
print("      include %s" % numpy.get_include())
print("      options %s" % numpy.get_printoptions())
print("")
try:
    from silx.gui.qt import QT_VERSION_STR
except Exception as error:
    print("Unable to import Qt")
    print(error)
else:
    print("Qt version " + QT_VERSION_STR)
print("")
try:
    import pyopencl
except Exception as error:
    print("Unable to import pyopencl: %s" % error)
else:
    print("PyOpenCL platform:")
    for p in pyopencl.get_platforms():
        print("  %s" % p)
        for d in p.get_devices():
            print("    %s" % d)
