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

version = "0.10.2"
date = "2014-11"
import sys, logging
logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("pyFAI.__init__")
    logger.error("pyFAI required a python version >= 2.6")
    raise RuntimeError("pyFAI required a python version >= 2.6, now we are running: %s" % sys.version)
from .azimuthalIntegrator import AzimuthalIntegrator
load = AzimuthalIntegrator.sload

def tests():
    from . import test
    test.run_tests()