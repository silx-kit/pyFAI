#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, division, print_function

__doc__ = "test suite for Azimuthal integrator class"
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/07/2017"


import unittest
import os
import numpy
import logging
import time
import copy
import fabio
import tempfile
from .utilstest import UtilsTest, Rwp, getLogger, recursive_delete
logger = getLogger(__file__)
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..detectors import Detector
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab
tmp_dir = UtilsTest.tempdir
try:
    from ..third_party import six
except (ImportError, Exception):
    import six


class TestAzimPilatus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = UtilsTest.getimage("Pilatus6M.cbf")

    def setUp(self):
        """Download files"""
        self.data = fabio.open(self.img).data
        self.ai = AzimuthalIntegrator(detector="pilatus6m")
        self.ai.setFit2D(300, 1326, 1303)

    def test_separate(self):
        bragg, amorphous = self.ai.separate(self.data)
        self.assertTrue(amorphous.max() < bragg.max(), "bragg is more intense than amorphous")
        self.assertTrue(amorphous.std() < bragg.std(), "bragg is more variatic than amorphous")


class TestAzimHalfFrelon(unittest.TestCase):
    """basic test"""

    def setUp(self):
        """Download files"""

        fit2dFile = 'fit2d.dat'
        halfFrelon = "LaB6_0020.edf"
        splineFile = "halfccd.spline"
        poniFile = "LaB6.poni"

        self.tmpfiles = {"cython": os.path.join(tmp_dir, "cython.dat"),
                         "cythonSP": os.path.join(tmp_dir, "cythonSP.dat"),
                         "numpy": os.path.join(tmp_dir, "numpy.dat")}

        self.fit2dFile = UtilsTest.getimage(fit2dFile)
        self.halfFrelon = UtilsTest.getimage(halfFrelon)
        self.splineFile = UtilsTest.getimage(splineFile)
        poniFile = UtilsTest.getimage(poniFile)

        with open(poniFile) as f:
            data = []
            for line in f:
                if line.startswith("SplineFile:"):
                    data.append("SplineFile: " + self.splineFile)
                else:
                    data.append(line.strip())
        self.poniFile = os.path.join(tmp_dir, os.path.basename(poniFile))
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

        with open(self.poniFile, "w") as f:
            f.write(os.linesep.join(data))
        self.fit2d = numpy.loadtxt(self.fit2dFile)
        self.ai = AzimuthalIntegrator()
        self.ai.load(self.poniFile)
        self.data = fabio.open(self.halfFrelon).data
        for tmpfile in self.tmpfiles.values():
            if os.path.isfile(tmpfile):
                os.unlink(tmpfile)

    def tearDown(self):
        """Remove temporary files"""
        for fn in self.tmpfiles.values():
            if os.path.exists(fn):
                os.unlink(fn)

    def test_numpy_vs_fit2d(self):
        """
        Compare numpy histogram with results of fit2d
        """
#        logger.info(self.ai.__repr__())
        tth, I = self.ai.xrpd_numpy(self.data,
                                    len(self.fit2d), self.tmpfiles["numpy"], correctSolidAngle=False)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp numpy/fit2d = %.3f", rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Numpy Histogram vs Fit2D: Rwp=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-b", label='fit2d')
            sp.plot(tth, I, "-r", label="numpy histogram")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            six.moves.input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp numpy/fit2d: %.3f" % rwp)

    def test_cython_vs_fit2d(self):
        """
        Compare cython histogram with results of fit2d
        """
#        logger.info(self.ai.__repr__())
        tth, I = self.ai.xrpd_cython(self.data,
                                     len(self.fit2d), self.tmpfiles["cython"], correctSolidAngle=False, pixelSize=None)
#        logger.info(tth)
#        logger.info(I)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp cython/fit2d = %.3f", rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Cython Histogram vs Fit2D: Rwp=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-b", label='fit2d')
            sp.plot(tth, I, "-r", label="cython")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            six.moves.input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp cython/fit2d: %.3f" % rwp)

    def test_cythonSP_vs_fit2d(self):
        """
        Compare cython splitPixel with results of fit2d
        """
        logger.info(self.ai.__repr__())
        pos = self.ai.cornerArray(self.data.shape)
        t0 = time.time()
        logger.info("in test_cythonSP_vs_fit2d Before SP")

        tth, I = self.ai.xrpd_splitPixel(self.data,
                                         len(self.fit2d),
                                         self.tmpfiles["cythonSP"],
                                         correctSolidAngle=False)
        logger.info("in test_cythonSP_vs_fit2d Before")
        t1 = time.time() - t0
#        logger.info(tth)
#        logger.info(I)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp cythonSP(t=%.3fs)/fit2d = %.3f", t1, rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('CythonSP Histogram vs Fit2D: Rwp=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-b", label='fit2d')
            sp.plot(tth, I, "-r", label="cython")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            six.moves.input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp cythonSP/fit2d: %.3f" % rwp)

    def test_cython_vs_numpy(self):
        """
        Compare cython histogram with numpy histogram
        """
#        logger.info(self.ai.__repr__())
        data = self.data
        tth_np, I_np = self.ai.xrpd_numpy(data,
                                          len(self.fit2d),
                                          correctSolidAngle=False)
        tth_cy, I_cy = self.ai.xrpd_cython(data,
                                           len(self.fit2d),
                                           correctSolidAngle=False)
        logger.info("before xrpd_splitPixel")
        tth_sp, I_sp = self.ai.xrpd_splitPixel(data,
                                               len(self.fit2d),
                                               correctSolidAngle=False)
        logger.info("After xrpd_splitPixel")
        rwp = Rwp((tth_cy, I_cy), (tth_np, I_np))
        logger.info("Rwp = %.3f", rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Numpy Histogram vs Cython: Rwp=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-y", label='fit2d')
            sp.plot(tth_np, I_np, "-b", label='numpy')
            sp.plot(tth_cy, I_cy, "-r", label="cython")
            sp.plot(tth_sp, I_sp, "-g", label="SplitPixel")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            six.moves.input("Press enter to quit")

        self.assertLess(rwp, 3, "Rwp cython/numpy: %.3f" % rwp)

    def test_separate(self):
        "test separate with a mask. issue #209 regression test"
        msk = self.data < 100
        bragg, amorphous = self.ai.separate(self.data, mask=msk)
        self.assertTrue(amorphous.max() < bragg.max(), "bragg is more intense than amorphous")
        self.assertTrue(amorphous.std() < bragg.std(), "bragg is more variatic than amorphous")

    def test_medfilt1d(self):
        ref = self.ai.medfilt1d(self.data, 1000, unit="2th_deg", method="csr")
        ocl = self.ai.medfilt1d(self.data, 1000, unit="2th_deg", method="ocl_csr")
        rwp = Rwp(ref, ocl)
        logger.info("test_medfilt1d median Rwp = %.3f", rwp)
        self.assertLess(rwp, 1, "Rwp medfilt1d Numpy/OpenCL: %.3f" % rwp)

        ref = self.ai.medfilt1d(self.data, 1000, unit="2th_deg", method="csr", percentile=(20, 80))
        ocl = self.ai.medfilt1d(self.data, 1000, unit="2th_deg", method="ocl_csr", percentile=(20, 80))
        rwp = Rwp(ref, ocl)
        logger.info("test_medfilt1d trimmed-mean Rwp = %.3f", rwp)
        self.assertLess(rwp, 3, "Rwp trimmed-mean Numpy/OpenCL: %.3f" % rwp)

    def test_radial(self):

        res = self.ai.integrate_radial(self.data, npt=360, npt_rad=10,
                                       radial_range=(3.6, 3.9), radial_unit="2th_deg")
        self.assertLess(res[0].min(), -179, "chi min at -180")
        self.assertGreater(res[0].max(), 179, "chi max at +180")
        self.assertGreater(res[1].min(), 120, "intensity min in ok")
        self.assertLess(res[1].max(), 10000, "intensity max in ok")

        res = self.ai.integrate_radial(self.data, npt=360, npt_rad=10,
                                       radial_range=(3.6, 3.9), radial_unit="2th_deg", unit="chi_rad")
        self.assertLess(res[0].min(), -3, "chi min at -3rad")
        self.assertGreater(res[0].max(), 0, "chi max at +3rad")


class TestFlatimage(unittest.TestCase):
    """test the caking of a flat image"""
    epsilon = 1e-4

    def test_splitPixel(self):
        shape = (200, 201)
        data = numpy.ones(shape, dtype="float64")
        det = Detector(1e-4, 1e-4, max_shape=shape)
        ai = AzimuthalIntegrator(0.1, 1e-2, 1e-2, detector=det)
        I = ai.xrpd2_splitPixel(data, 256, 2256, correctSolidAngle=False, dummy=-1.0)[0]
#        I = ai.xrpd2(data, 2048, 2048, correctSolidAngle=False, dummy= -1.0)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('cacking of a flat image: SplitPixel')
            sp = fig.add_subplot(111)
            sp.imshow(I, interpolation="nearest")
            fig.show()
            six.moves.input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon

    def test_splitBBox(self):
        shape = (200, 201)
        data = numpy.ones(shape, dtype="float64")
        det = Detector(1e-4, 1e-4, max_shape=shape)
        ai = AzimuthalIntegrator(0.1, 1e-2, 1e-2, detector=det)
        I = ai.xrpd2_splitBBox(data, 256, 256, correctSolidAngle=False, dummy=-1.0)[0]

        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('cacking of a flat image: SplitBBox')
            sp = fig.add_subplot(111)
            sp.imshow(I, interpolation="nearest")
            fig.show()
            six.moves.input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon


class TestSaxs(unittest.TestCase):
    saxsPilatus = "bsa_013_01.edf"
    maskFile = "Pcon_01Apr_msk.edf"
    maskRef = "bioSaxsMaskOnly.edf"

    def setUp(self):
        self.edfPilatus = UtilsTest.getimage(self.__class__.saxsPilatus)
        self.maskFile = UtilsTest.getimage(self.__class__.maskFile)
        self.maskRef = UtilsTest.getimage(self.__class__.maskRef)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.edfPilatus = self.maskFile = self.maskRef = None

    def test_mask(self):
        """test the generation of mask"""
        ai = AzimuthalIntegrator(detector="Pilatus1M")
        ai.wavelength = 1e-10

        data = fabio.open(self.edfPilatus).data
        mask = fabio.open(self.maskFile).data
        self.assertTrue(abs(ai.create_mask(data, mask=mask).astype(int) - fabio.open(self.maskRef).data).max() == 0, "test without dummy")
#         self.assertTrue(abs(self.ai.create_mask(data, mask=mask, dummy=-48912, delta_dummy=40000).astype(int) - fabio.open(self.maskDummy).data).max() == 0, "test_dummy")

    def test_normalization_factor(self):

        ai = AzimuthalIntegrator(detector="Pilatus100k")
        ai.wavelength = 1e-10
        methods = ["cython", "numpy", "lut", "csr", "ocl_lut", "ocl_csr", "splitpixel"]
        ref1d = {}
        ref2d = {}

        data = fabio.open(self.edfPilatus).data[:ai.detector.shape[0], :ai.detector.shape[1]]
        for method in methods:
            logger.debug("TestSaxs.test_normalization_factor method= " + method)
            ref1d[method + "_1"] = ai.integrate1d(copy.deepcopy(data), 100, method=method, error_model="poisson")
            ref1d[method + "_10"] = ai.integrate1d(copy.deepcopy(data), 100, method=method, normalization_factor=10, error_model="poisson")
            ratio_i = ref1d[method + "_1"].intensity.mean() / ref1d[method + "_10"].intensity.mean()
            ratio_s = ref1d[method + "_1"].sigma.mean() / ref1d[method + "_10"].sigma.mean()

            self.assertAlmostEqual(ratio_i, 10.0, places=3, msg="test_normalization_factor 1d intensity Method: %s ratio: %s expected 10" % (method, ratio_i))
            self.assertAlmostEqual(ratio_s, 10.0, places=3, msg="test_normalization_factor 1d sigma Method: %s ratio: %s expected 10" % (method, ratio_s))
            # ai.reset()
            ref2d[method + "_1"] = ai.integrate2d(copy.deepcopy(data), 100, 36, method=method, error_model="poisson")
            ref2d[method + "_10"] = ai.integrate2d(copy.deepcopy(data), 100, 36, method=method, normalization_factor=10, error_model="poisson")
            ratio_i = ref2d[method + "_1"].intensity.mean() / ref2d[method + "_10"].intensity.mean()
            # ratio_s = ref2d[method + "_1"].sigma.mean() / ref2d[method + "_10"].sigma.mean()
            self.assertAlmostEqual(ratio_i, 10.0, places=3, msg="test_normalization_factor 2d intensity Method: %s ratio: %s expected 10" % (method, ratio_i))
            # self.assertAlmostEqual(ratio_s, 10.0, places=3, msg="test_normalization_factor 2d sigma Method: %s ratio: %s expected 10" % (method, ratio_s))
            # ai.reset()

    def test_inpainting(self):
        logger.debug("TestSaxs.test_inpainting")
        img = fabio.open(self.edfPilatus).data
        ai = AzimuthalIntegrator(detector="Pilatus1M")
        ai.setFit2D(2000, 870, 102.123456789)  # rational numbers are hell !
        mask = img < 0
        inp = ai.inpainting(img, mask)
        neg = (inp < 0).sum()
        logger.debug("neg=%s" % neg)
        self.assertTrue(neg == 0, "all negative pixels got inpainted actually all but %s" % neg)
        self.assertTrue(mask.sum() > 0, "some pixel needed inpainting")


class TestSetter(unittest.TestCase):
    def setUp(self):
        self.ai = AzimuthalIntegrator()
        shape = (10, 15)
        self.rnd1 = numpy.random.random(shape).astype(numpy.float32)
        self.rnd2 = numpy.random.random(shape).astype(numpy.float32)
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)

        fd, self.edf1 = tempfile.mkstemp(".edf", "testAI1", tmp_dir)
        os.close(fd)
        fd, self.edf2 = tempfile.mkstemp(".edf", "testAI2", tmp_dir)
        os.close(fd)
        fabio.edfimage.edfimage(data=self.rnd1).write(self.edf1)
        fabio.edfimage.edfimage(data=self.rnd2).write(self.edf2)

    def tearDown(self):
        recursive_delete(tmp_dir)

    def test_flat(self):
        self.ai.set_flatfiles((self.edf1, self.edf2), method="mean")
        self.assertTrue(self.ai.flatfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "flatfiles string is OK")
        self.assertTrue(abs(self.ai.flatfield - 0.5 * (self.rnd1 + self.rnd2)).max() == 0, "Flat array is OK")

    def test_dark(self):
        self.ai.set_darkfiles((self.edf1, self.edf2), method="mean")
        self.assertTrue(self.ai.darkfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "darkfiles string is OK")
        self.assertTrue(abs(self.ai.darkcurrent - 0.5 * (self.rnd1 + self.rnd2)).max() == 0, "Dark array is OK")


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestAzimHalfFrelon))
    testsuite.addTest(loader(TestFlatimage))
    testsuite.addTest(loader(TestSetter))
    testsuite.addTest(loader(TestAzimPilatus))
    testsuite.addTest(loader(TestSaxs))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
