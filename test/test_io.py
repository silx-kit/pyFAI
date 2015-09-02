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

"test suite for input/output stuff"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/09/2015"


import unittest
import os
import shutil
import numpy
import time
import sys
import fabio
import tempfile
is_main = (__name__ == '__main__')
if is_main:
    import pkgutil
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import UtilsTest, Rwp, getLogger

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

        self.assert_(io.is_hdf5(fname), "nexus file is an HDF5")
#        os.system("h5ls -r -a %s" % fname)



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
        logger.info("Writing of HDF5 of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)" % (n, shape, nmbytes, t, nmbytes / t))
        statinfo = os.stat(h5file)
        self.assert_(statinfo.st_size / 1e6 > nmbytes, "file size (%s) is larger than dataset" % statinfo.st_size)



class testFabIOWriter(unittest.TestCase):

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
        logger.info("Writing of HDF5 of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)" % (n, shape, nmbytes, t, nmbytes / t))
        statinfo = os.stat(h5file)
        self.assert_(statinfo.st_size / 1e6 > nmbytes, "file size (%s) is larger than dataset" % statinfo.st_size)


def test_suite_all_io():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestIsoTime("test_get"))
    testSuite.addTest(TestIsoTime("test_from"))
    testSuite.addTest(TestNexus("test_new_detector"))
    testSuite.addTest(testHDF5Writer("test_writer"))
#    testSuite.addTest(testFabIOWriter("test_writer"))
    return testSuite

if is_main:
    mysuite = test_suite_all_io()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
    UtilsTest.clean_up()
