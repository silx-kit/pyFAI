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
__date__ = "28/09/2015"

import sys
import os
import unittest
import numpy
import subprocess
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import getLogger, UtilsTest  # , Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import fabio


class TestBug170(unittest.TestCase):
    """
    Test a mar345 image with 2300 pixels size
    """

    def setUp(self):
        ponitxt = """
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
        self.ponifile = os.path.join(UtilsTest.tempdir, "bug170.poni")
        with open(self.ponifile, "w") as poni:
            poni.write(ponitxt)
        self.data = numpy.random.random((2300, 2300))

    def tearDown(self):
        if os.path.exists(self.ponifile):
            os.unlink(self.ponifile)
        self.data = None

    def test_bug170(self):
        ai = pyFAI.load(self.ponifile)
        logger.debug(ai.mask.shape)
        logger.debug(ai.detector.pixel1)
        logger.debug(ai.detector.pixel2)
        ai.integrate1d(self.data, 2000)


class TestBug211(unittest.TestCase):
    """
    Check the quantile filter in pyFAI-average
    """
    def setUp(self):
        shape = (100, 100)
        dtype = numpy.float32
        self.image_files = []
        self.outfile = os.path.join(UtilsTest.tempdir, "out.edf")
        res = numpy.zeros(shape, dtype=dtype)
        for i in range(5):
            fn = os.path.join(UtilsTest.tempdir, "img_%i.edf" % i)
            if i == 3:
                data = numpy.zeros(shape, dtype=dtype)
            elif i == 4:
                data = numpy.ones(shape, dtype=dtype)
            else:
                data = numpy.random.random(shape).astype(dtype)
                res += data
            e = fabio.edfimage.edfimage(data=data)
            e.write(fn)
            self.image_files.append(fn)
        self.res = res / 3.0
        self.exe, self.env = UtilsTest.script_path("pyFAI-average")

    def tearDown(self):
        for fn in self.image_files:
            os.unlink(fn)
        if os.path.exists(self.outfile):
            os.unlink(self.outfile)
        self.image_files = None
        self.res = None
        self.exe = self.env = None

    def test_quantile(self):
        p = subprocess.call([sys.executable, self.exe, "--quiet", "-q", "0.2-0.8", "-o", self.outfile] + self.image_files,
                            shell=False, env=self.env)
        self.assertEqual(p, 0, msg="pyFAI-average return code is 0")
        self.assert_(numpy.allclose(fabio.open(self.outfile).data, self.res),
                         "pyFAI-average with quantiles gives good results")


def test_suite_bug_regression():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestBug170("test_bug170"))
    testSuite.addTest(TestBug211("test_quantile"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_bug_regression()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)


