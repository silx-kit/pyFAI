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
"test suite for OpenCL code"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/06/2012"


import unittest
import os
import numpy
import logging, time
import sys
import fabio
from utilstest import UtilsTest, Rwp, parseArgs
logger = parseArgs(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.geometryRefinement import AzimuthalIntegrator
from pyFAI.ocl_azim import Integrator1d
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab

datasets = [{"img":UtilsTest.getimage("1883/Pilatus1M.edf"), "poni":UtilsTest.getimage("1880/Pilatus1M.poni"), "spline": None},
            {"img":UtilsTest.getimage("1882/halfccd.edf"), "poni":UtilsTest.getimage("1880/halfccd.poni"), "spline": UtilsTest.getimage("1461/halfccd.spline")},
            {"img":UtilsTest.getimage("1881/Frelon2k.edf"), "poni":UtilsTest.getimage("1880/Frelon2k.poni"), "spline": UtilsTest.getimage("1461/frelon.spline")},
            {"img":UtilsTest.getimage("1884/Pilatus6M.cbf"), "poni":UtilsTest.getimage("1880/Pilatus6M.poni"), "spline": None},
            {"img":UtilsTest.getimage("1880/Fairchild.edf"), "poni":UtilsTest.getimage("1880/Fairchild.poni"), "spline": None},
      ]

class test_mask(unittest.TestCase):
    @classmethod
    def find_device(cls):
        inte = Integrator1d()
        inte.init("all", useFp64=True)
        ids = inte.get_contexed_Ids()
        if "cl_khr_int64_base_atomics" in inte.get_device_info()["extensions"]:
            return ids
        else:
            return None

#TODO
