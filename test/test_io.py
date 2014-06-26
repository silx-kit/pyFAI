#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
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
"test suite for input/output stuff"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "24/06/2014"


import unittest
import os, shutil
import numpy
import logging, time
import sys
import fabio
import tempfile
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import io

class TestIsoTime(unittest.TestCase):
    def test_get(self):
        self.assert_(len(io.get_isotime()), 25)

    def test_from(self):
        t0 = time.time()
        isotime = io.get_isotime(t0)
        self.assert_(abs(t0 - io.from_isotime(isotime)) < 1, "timing are precise to the second")

class TestNexus(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tmpdir =tempfile.mkdtemp() 
    def test_new_detector(self):
        fname = os.path.join(self.tmpdir, "nxs.h5")
        nxs = io.Nexus(fname, "r+")
        nxs.new_detector()
        nxs.close()
#        os.system("h5ls -r -a %s" % fname)
    def tearDown(self):
        unittest.TestCase.tearDown(self)
        shutil.rmtree(self.tmpdir)
def test_suite_all_io():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestIsoTime("test_get"))
    testSuite.addTest(TestIsoTime("test_from"))
    testSuite.addTest(TestNexus("test_new_detector"))


    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_io()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
