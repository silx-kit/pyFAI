#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "18/07/2017"

import sys
import logging
if "ps1" in dir(sys):
    logging.basicConfig()

import os
project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion, calc_hexversion
except ImportError:
    raise RuntimeError("Do NOT use %s from its sources: build it and use the built version" % project)

if sys.version_info < (2, 6):
    logger = logging.getLogger("pyFAI.__init__")
    logger.error("pyFAI required a python version >= 2.6")
    raise RuntimeError("pyFAI required a python version >= 2.6, now we are running: %s" % sys.version)

from .detectors import Detector
from .azimuthalIntegrator import AzimuthalIntegrator
from .decorators import depreclog
load = AzimuthalIntegrator.sload
detector_factory = Detector.factory


def tests(deprecation=False):
    """Runs the test suite of the installed version

    :param deprecation: enable/disables deprecation warning in the tests
    """
    if deprecation:
        depreclog.setLevel(logging.DEBUG)
    else:
        depreclog.setLevel(logging.ERROR)
    from . import test
    res = test.run_tests()
    depreclog.setLevel(logging.DEBUG)
    return res


def benchmarks(*arg, **kwarg):
    """Run the integrated benchmarks.

    See the documentation of pyFAI.benchmark.run_benchmark
    """
    from . import benchmark
    res = benchmark.run(*arg, **kwarg)
    return res
