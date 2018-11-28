#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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


from __future__ import division, print_function, absolute_import

"""Test suite for utilities library"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/11/2018"

import unittest
import logging
import numpy
import tempfile
import os.path
import shutil

import fabio
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from .. import utils
from ..utils import ioutils
from .. import _version


class TestUtils(unittest.TestCase):

    def test_set(self):
        s = utils.FixedParameters()
        self.assertEqual(len(s), 0, "initial set is empty")
        s.add_or_discard("a", True)
        self.assertEqual(len(s), 1, "a is in set")
        s.add_or_discard("a", None)
        self.assertEqual(len(s), 1, "set is untouched")
        s.add_or_discard("a", False)
        self.assertEqual(len(s), 0, "set is empty again")
        s.add_or_discard("a", None)
        self.assertEqual(len(s), 0, "set is untouched")
        s.add_or_discard("a", False)
        self.assertEqual(len(s), 0, "set is still empty")

    def test_hexversion(self):
        # print(_version, type(_version))
        self.assertEqual(_version.calc_hexversion(1), 1 << 24, "Major is OK")
        self.assertEqual(_version.calc_hexversion(0, 1), 1 << 16, "Minor is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 1), 1 << 8, "Micro is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 0, 1), 1 << 4, "Release level is OK")
        self.assertEqual(_version.calc_hexversion(0, 0, 0, 0, 1), 1, "Serial is OK")


class TestIoUtils(unittest.TestCase):
    def setUp(self):
        shape = (10, 15)
        self.rnd1 = numpy.random.random(shape).astype(numpy.float32)
        self.rnd2 = numpy.random.random(shape).astype(numpy.float32)

        tmp_dir = os.path.join(UtilsTest.tempdir, self.id())
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        self.tmp_dir = tmp_dir

        fd, self.edf1 = tempfile.mkstemp(".edf", "testAI1", tmp_dir)
        os.close(fd)
        fd, self.edf2 = tempfile.mkstemp(".edf", "testAI2", tmp_dir)
        os.close(fd)
        fabio.edfimage.edfimage(data=self.rnd1).write(self.edf1)
        fabio.edfimage.edfimage(data=self.rnd2).write(self.edf2)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_single(self):
        files = (self.edf1,)
        data, source = ioutils.average_files(files, "mean")
        self.assertTrue(source == "%s" % (self.edf1,))
        self.assertTrue(abs(data - self.rnd1).max() == 0)

    def test_multi(self):
        files = (self.edf1, self.edf2)
        data, source = ioutils.average_files(files, "mean")
        self.assertTrue(source == "%s(%s,%s)" % ("mean", self.edf1, self.edf2))
        self.assertTrue(abs(data - 0.5 * (self.rnd1 + self.rnd2)).max() == 0)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestUtils))
    testsuite.addTest(loader(TestIoUtils))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
