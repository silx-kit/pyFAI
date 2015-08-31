#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
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

from __future__ import division, print_function, absolute_import
"test suite for inverse watershed space segmenting code."
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/04/2015"


import unittest
import numpy
import sys
import os
import fabio
import tempfile
if __name__ == '__main__':
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, getLogger, recursive_delete
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.watershed


class TestWatershed(unittest.TestCase):
    fname = "1883/Pilatus1M.edf"

    def setUp(self):
        self.data = fabio.open(UtilsTest.getimage(self.fname)).data

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.data = None

    def test_init(self):
        w = pyFAI.watershed.InverseWatershed(data=self.data)
        w.init()
        print(len(w.regions))
        from sys import getsizeof
        print(getsizeof(w))
        w.__dealloc__()
        print(getsizeof(w))


def test_suite_all_watershed():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestWatershed("test_init"))
    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_watershed()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    UtilsTest.clean_up()
