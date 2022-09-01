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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/09/2022"

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
import pyFAI.io.spots
import fabio
import pyFAI.azimuthalIntegrator


class TestIsoTime(unittest.TestCase):

    def test_get(self):
        self.assertTrue(len(io.get_isotime()), 25)

    def test_from(self):
        t0 = time.perf_counter()
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
        nxs = io.Nexus(fname, "a")
        nxs.new_detector()
        nxs.close()

        self.assertTrue(io.is_hdf5(fname), "nexus file is an HDF5")
        # os.system("h5ls -r -a %s" % fname)

    def test_NXmonopd(self):
        img = fabio.open(UtilsTest.getimage("Pilatus1M.edf"))
        ai = pyFAI.load(UtilsTest.getimage("Pilatus1M.poni"))
        ref = ai.integrate1d(img.data, 1000, unit="2th_deg", error_model="poisson")
        fname = os.path.join(self.tmpdir, "NXmonopd.h5")
        io.nexus.save_NXmonpd(fname, ref, sample="AgBh", instrument="Dubble")
        res = io.nexus.load_nexus(fname)
        for i, j in zip(res, ref):
            self.assertTrue(numpy.allclose(i, j))
        for k in dir(ref):
            if k.startswith("__") or k in ["_sem", "std", "_std", "sem", "_sem"]:
                continue
            a = getattr(ref, k, None)
            b = getattr(res, k, None)
            if callable(a):
                continue
            elif isinstance(a, numpy.ndarray):
                self.assertTrue(numpy.allclose(a, b), msg=f"check {k}")
            elif isinstance(a, (int, float, str, tuple, type(None))):
                self.assertEqual(a, b, k)
            elif isinstance(a, io.ponifile.PoniFile):
                self.assertEqual(a.as_dict(), b.as_dict(), "Poni matches")
            elif isinstance(a, pyFAI.method_registry.IntegrationMethod):
                self.assertEqual(a.method, b.method, "method matches")
            elif isinstance(a, pyFAI.units.Unit):
                self.assertEqual(str(a), str(b), "unit matches")

            elif isinstance(a, dict):
                for l in a:
                    print(l, a[l])
                    self.assertEqual(a[l], b[l], f"{k}[{l}]")
            else:
                logger.warning("unchecked: %s vs %s", a, b)
        # clean up
        os.unlink(fname)


class TestHDF5Writer(unittest.TestCase):

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
        t0 = time.perf_counter()
        writer = io.HDF5Writer(filename=h5file, hpath="data")
        writer.init({"nbpt_azim": shape[0], "nbpt_rad": shape[1]})
        for i in range(n):
            writer.write(data[i % m], i)
        writer.close()
        t = time.perf_counter() - t0
        logger.info("Writing of HDF5 of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)", n, shape, nmbytes, t, nmbytes / t)
        statinfo = os.stat(h5file)
        self.assertTrue(statinfo.st_size / 1e6 > nmbytes, "file size (%s) is larger than dataset" % statinfo.st_size)


class TestFabIOWriter(unittest.TestCase):
    """TODO finish !"""

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

        shape = 100, 128
        n = 10
        m = 3  # number of frames in memory
        data = numpy.random.random((m, shape[0], shape[1])).astype(numpy.float32)
        nmbytes = data.nbytes / 1e6 * n / m
        t0 = time.perf_counter()
        writer = io.FabioWriter(extension=".edf")
        logger.info(writer.__repr__())
        writer.init({"nbpt_azim": shape[0],
                     "nbpt_rad": shape[1],
                     "prefix": "toto",
                     "index_format": "_%04d",
                     "start_index": 1,
                     "directory": self.tmpdir})
        logger.info(writer.__repr__())

        for i in range(n):
            writer.write(data[i % m])
        writer.close()
        t = time.perf_counter() - t0
        logger.info("Writing of Fabio of %ix%s (%.3fMB) took %.3f (%.3fMByte/s)", n, shape, nmbytes, t, nmbytes / t)
        self.assertEqual(len(os.listdir(self.tmpdir)), n)


class TestSpotWriter(unittest.TestCase):

    def setUp(self):
        unittest.TestCase.setUp(self)

        detector = pyFAI.detector_factory("pilatus300k")
        self.ai = pyFAI.azimuthalIntegrator.AzimuthalIntegrator(detector=detector)
        nframes = 100
        nspots = numpy.random.randint(1, nframes, size=nframes)
        self.spots = [numpy.empty(count, dtype=[("index", numpy.int32),
                                                ("intensity", numpy.float32),
                                                ("sigma", numpy.float32),
                                                ("pos0", numpy.float32),
                                                ("pos1", numpy.float32)])
                       for count in nspots]

    def test_nexus(self):
        tmpfile = os.path.join(UtilsTest.tempdir, "io_FabIOwriter_spots.nxs")
        io.spots.save_spots_nexus(tmpfile, self.spots, beamline="beamline", ai=self.ai)
        size = os.stat(tmpfile)
        self.assertGreater(size.st_size, sum(i.size for i in self.spots), "file is large enough")

    def test_cxi(self):
        tmpfile = os.path.join(UtilsTest.tempdir, "io_FabIOwriter_spots.nxs")
        io.spots.save_spots_cxi(tmpfile, self.spots, beamline="beamline", ai=self.ai)
        size = os.stat(tmpfile)
        self.assertGreater(size.st_size, sum(i.size for i in self.spots), "file is large enough")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestIsoTime))
    testsuite.addTest(loader(TestNexus))
    testsuite.addTest(loader(TestHDF5Writer))
    testsuite.addTest(loader(TestFabIOWriter))
    testsuite.addTest(loader(TestSpotWriter))
    return testsuite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
