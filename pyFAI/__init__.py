#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__date__ = "02/06/2016"

import sys
import logging
logging.basicConfig()

import os
project = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
try:
    from ._version import __date__ as date
    from ._version import version, version_info, hexversion, strictversion
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
    """
    Runs the test suite of the installed version

    @param deprecation: enable/disables deprecation warning in the tests
    """
    if deprecation:
        depreclog.setLevel(logging.DEBUG)
    else:
        depreclog.setLevel(logging.ERROR)
    from . import test
    res = test.run_tests()
    depreclog.setLevel(logging.DEBUG)
    return res


def benchmarks():
    """
    Run the benchmarks
    """
    from . import benchmark
    res = benchmark.run()
    return res
