#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: FasT Azimuthal integration
#             https://github.com/kif/pyFAI
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
"test suite for utilities library"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/04/2013"


import unittest
import numpy
import logging
import sys
import fabio
from utilstest import UtilsTest, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.utils

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


#TODO Test:
# gaussian_filter
# shift
# relabel
# averageDark
# averageImages
# boundingBox
# removeSaturatedPixel
#DONE:
## binning
## unbinning

class test_utils(unittest.TestCase):
    unbinned = numpy.random.random((64, 32))
    def setUp(self):
        """Download files"""
        pass
    def test_binning(self):
        """
        The masked image has a masked ring around 1.5deg with value -10
        without mask the pixels should be at -10 ; with mask they are at 0
        """
        binned = pyFAI.utils.binning(self.unbinned, (4, 2))
        self.assertEqual(binned.shape, (64 / 4, 32 / 2), "binned size is OK")
        unbinned = pyFAI.utils.unBinning(binned, (4, 2))
        self.assertEqual(unbinned.shape, self.unbinned.shape, "unbinned size is OK")
        self.assertAlmostEqual(unbinned.sum(), self.unbinned.sum(), 2, "content is the same")


def test_suite_all_Utils():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_utils("test_binning"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_Utils()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
