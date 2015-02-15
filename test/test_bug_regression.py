#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
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

"""test suite for non regression on some bugs.

Please refer to their respective bug number
https://github.com/kif/pyFAI/issues
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme Kieffer"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/02/2015"

import sys
import os
import unittest
import numpy

if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from utilstest import getLogger, UtilsTest#, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

class TestBug170(unittest.TestCase):
    """
    Test a mar345 image with 2300 pixels size
    """
    poni = """
Detector: Mar345
PixelSize1: 0.00015
PixelSize2: 0.00015
Distance: 0.446642915189
Poni1: 0.228413453499
Poni2: 0.272291324302
Rot1: 0.0233130647508
Rot2: 0.0011735285628
Rot3: -7.22446379865e-08
SplineFile: None
Wavelength: 7e-11
"""

    def setUp(self):
        self.ponifile = os.path.join(UtilsTest.tempdir, "bug170.poni")
        with open(self.ponifile, "w") as poni:
            poni.write(self.poni)
        self.data = numpy.random.random((2300,2300))

    def test_bug170(self):
        ai = pyFAI.load(self.ponifile)
        logger.debug(ai.mask.shape)
        logger.debug(ai.detector.pixel1)
        logger.debug(ai.detector.pixel2)
        ai.integrate1d(self.data, 2000)

    def tearDown(self):
        if os.path.exists(self.ponifile):
            os.unlink(self.ponifile)
        self.data = None

def test_suite_bug_regression():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBug170("test_bug170"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_bug_regression()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)


