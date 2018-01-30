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

"test suite for input/output stuff"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"


import unittest
import os
import shutil
import numpy
import time
import sys
import logging
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)
pyFAI = sys.modules["pyFAI"]
from pyFAI import io


class TestIsoTime(unittest.TestCase):
    def test_get(self):
        self.assertTrue(len(io.get_isotime()), 25)

    def test_from(self):
        t0 = time.time()
        isotime = io.get_isotime(t0)
        self.assertTrue(abs(t0 - io.from_isotime(isotime)) < 1, "timing are precise to the second")


class TestNexus(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tmpdir = os.path.join(UtilsTest.tempdir, "io_nexus")
        if not os.path.isdir(self.tmpdir):
            os.mkdir(self.tmpdir)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        shutil.rmtree(self.tmpdir)
        self.tmpdir = None

    def test_new_detector(self):
        if io.h5py is None:
            logger.warning("H5py not present, skipping test_io.TestNexus")
            return
        fname = os.path.join(self.tmpdir, "nxs.h5")
        nxs = io.Nexus(fname, "r+")
        nxs.new_detector()
        nxs.close()

        self.assertTrue(io.is_hdf5(fname), "nexus file is an HDF5")
        # os.system("h5ls -r -a %s" % fname)


class testHDF5Writer(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tmpdir = os.path.join(UtilsTest.tempdir, "io_HDF5Writer")
        if not os.path.isdir(self.tmpdir):
            os.mkdir(self.tmpdir)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        shutil.rmtree(self.tmpdir)
        self.tmpdir = None

    def test_writer(self):
        if io.h5py is None:
            logger.warning("H5py is absent on the system, skip HDF5 writing test")
            return
        h5file = os.path.join(self.tmpdir, "junk.h5")
        shape = 1024, 1024
        n = 100
        m = 10  # number of frames in memory
        data = numpy.random.random((m, shape[0], shape[1])).astype(numpy.float32)
        nmbytes = data.nbytes / 1e6 * n / m
        t0 = time.time()
        writer = io.HDF5Writer(filename=h5file, hpath="data")
        writer.init({"nbpt_azim": shape[0], "nbpt_rad": shape[1]})
        for i in range(n):
            writer.write(data[i % m], i)
        writer.close()
        t = time.time() - t0
        logger.info("Writing of HDF5 of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)", n, shape, nmbytes, t, nmbytes / t)
        statinfo = os.stat(h5file)
        self.assertTrue(statinfo.st_size / 1e6 > nmbytes, "file size (%s) is larger than dataset" % statinfo.st_size)


class testFabIOWriter(unittest.TestCase):
    """the tested class is not yet finished ... JK07/2017"""

    def setUp(self):
        unittest.TestCase.setUp(self)
        self.tmpdir = os.path.join(UtilsTest.tempdir, "io_FabIOwriter")
        if not os.path.isdir(self.tmpdir):
            os.mkdir(self.tmpdir)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        shutil.rmtree(self.tmpdir)
        self.tmpdir = None

    def test_writer(self):
        # self.skipTest("Untested")

        h5file = os.path.join(self.tmpdir)
        shape = 1024, 1024
        n = 100
        m = 10  # number of frames in memory
        data = numpy.random.random((m, shape[0], shape[1])).astype(numpy.float32)
        nmbytes = data.nbytes / 1e6 * n / m
        t0 = time.time()
        writer = io.FabioWriter(filename=h5file)
        writer.init({"nbpt_azim": shape[0], "nbpt_rad": shape[1], "prefix": "test"})
        for i in range(n):
            writer.write(data[i % m], i)
        writer.close()
        t = time.time() - t0
        logger.info("Writing of HDF5 of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)", n, shape, nmbytes, t, nmbytes / t)
        statinfo = os.stat(h5file)
        self.assertTrue(statinfo.st_size / 1e6 > nmbytes, "file size (%s) is larger than dataset" % statinfo.st_size)


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestIsoTime))
    testsuite.addTest(loader(TestNexus))
    testsuite.addTest(loader(testHDF5Writer))
    # testsuite.addTest(loader(testFabIOWriter))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
