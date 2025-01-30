#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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


"""test suite for Azimuthal integrator class"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2025"

import unittest
import os
import numpy
import logging
import time
import copy
import fabio
import gc
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..integrator.azimuthal import AzimuthalIntegrator
from ..method_registry import IntegrationMethod
from ..containers import ErrorModel
from ..detectors import Detector, detector_factory
if logger.getEffectiveLevel() <= logging.DEBUG:
    import pylab
from pyFAI import units
from ..utils import mathutil
from ..utils.logging_utils import logging_disabled
from ..opencl import pyopencl


class TestAzimHalfFrelon(unittest.TestCase):
    """basic test"""

    @classmethod
    def setUpClass(cls):
        """Download files"""
        super(TestAzimHalfFrelon, cls).setUpClass()

        fit2dFile = 'fit2d.dat'
        halfFrelon = "LaB6_0020.edf"
        splineFile = "halfccd.spline"
        poniFile = "LaB6.poni"

        tmp_dir = os.path.join(UtilsTest.tempdir, "TestAzimHalfFrelon")
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        cls.tmp_dir = tmp_dir

        cls.tmpfiles = {"cython": os.path.join(tmp_dir, "cython.dat"),
                        "cythonSP": os.path.join(tmp_dir, "cythonSP.dat"),
                        "numpy": os.path.join(tmp_dir, "numpy.dat")}

        cls.fit2dFile = UtilsTest.getimage(fit2dFile)
        cls.halfFrelon = UtilsTest.getimage(halfFrelon)
        cls.splineFile = UtilsTest.getimage(splineFile)
        poniFile = UtilsTest.getimage(poniFile)

        with open(poniFile) as f:
            data = []
            for line in f:
                if line.startswith("SplineFile:"):
                    data.append("SplineFile: " + cls.splineFile)
                else:
                    data.append(line.strip())
        cls.poniFile = os.path.join(tmp_dir, os.path.basename(poniFile))

        with open(cls.poniFile, "w") as f:
            f.write(os.linesep.join(data))
        cls.fit2d = numpy.loadtxt(cls.fit2dFile)
        cls.ai = AzimuthalIntegrator()
        cls.ai.load(cls.poniFile)
        with fabio.open(cls.halfFrelon) as fimg:
            cls.data = fimg.data
        for tmpfile in cls.tmpfiles.values():
            if os.path.isfile(tmpfile):
                os.unlink(tmpfile)

    @classmethod
    def tearDownClass(cls):
        """Remove temporary files"""
        super(TestAzimHalfFrelon, cls).tearDownClass()
        for fn in cls.tmpfiles.values():
            if os.path.exists(fn):
                os.unlink(fn)
        cls.fit2d = None
        cls.ai = None
        cls.data = None
        cls.tmpfiles = None

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        try:
            self.__class__.ai.reset()
        except Exception as e:
            logger.error(e)
        gc.collect()

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def test_numpy_vs_fit2d(self):
        """
        Compare numpy histogram with results of fit2d
        """
        tth, I = self.ai.integrate1d_ng(self.data,
                                        len(self.fit2d),
                                        filename=self.tmpfiles["numpy"],
                                        correctSolidAngle=False,
                                        method=("no", "histogram", "cython"),
                                        unit="2th_deg")
        rwp = mathutil.rwp((tth, I), self.fit2d.T)
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
            input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp numpy/fit2d: %.3f" % rwp)

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def test_cython_vs_fit2d(self):
        """
        Compare cython histogram with results of fit2d
        """
        tth, I = self.ai.integrate1d_ng(self.data,
                                     len(self.fit2d),
                                     filename=self.tmpfiles["cython"],
                                     correctSolidAngle=False,
                                     unit='2th_deg',
                                     method=("no", "histogram", "cython"))
        rwp = mathutil.rwp((tth, I), self.fit2d.T)
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
            input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp cython/fit2d: %.3f" % rwp)

    @unittest.skipIf(UtilsTest.low_mem, "test using >200M")
    def test_cythonSP_vs_fit2d(self):
        """
        Compare cython splitPixel with results of fit2d
        """
        logger.info(self.ai.__repr__())
        self.ai.corner_array(self.data.shape, unit=units.TTH_RAD, scale=False)
        # this was just to enforce the initalization of the array
        t0 = time.perf_counter()
        logger.info("in test_cythonSP_vs_fit2d Before SP")

        tth, I = self.ai.integrate1d_ng(self.data,
                                        len(self.fit2d),
                                        filename=self.tmpfiles["cythonSP"],
                                        method=("full", "histogram", "cython"),
                                        correctSolidAngle=False,
                                        unit="2th_deg")
        logger.info("in test_cythonSP_vs_fit2d Before")
        t1 = time.perf_counter() - t0
        rwp = mathutil.rwp((tth, I), self.fit2d.T)
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
            input("Press enter to quit")
        self.assertLess(rwp, 11, "Rwp cythonSP/fit2d: %.3f" % rwp)

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def test_cython_vs_numpy(self):
        """
        Compare cython histogram with numpy histogram
        """
        tth_np, I_np = self.ai.integrate1d_ng(self.__class__.data,
                                              len(self.fit2d),
                                              correctSolidAngle=False,
                                              unit="2th_deg",
                                              method=("no", "histogram", "python"))
        tth_cy, I_cy = self.ai.integrate1d_ng(self.__class__.data,
                                              len(self.fit2d),
                                              correctSolidAngle=False,
                                              unit="2th_deg",
                                              method=("no", "histogram", "cython"))
        tth_sp, I_sp = self.ai.integrate1d_ng(self.__class__.data,
                                              len(self.fit2d),
                                              correctSolidAngle=False,
                                              unit="2th_deg",
                                              method=("full", "histogram", "cython"))

        rwp = mathutil.rwp((tth_cy, I_cy), (tth_np, I_np))
        logger.info("Histogram Cython/Numpy Rwp = %.3f", rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig, sp = pylab.subplots()
            fig.suptitle('Numpy Histogram vs Cython: Rwp=%.3f' % rwp)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-y", label='fit2d')
            sp.plot(tth_np, I_np, "-b", label='numpy')
            sp.plot(tth_cy, I_cy, "-r", label="cython")
            sp.plot(tth_sp, I_sp, "-g", label="SplitPixel")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            input("Press enter to quit")

        self.assertLess(rwp, 3, "Rwp cython/numpy: %.3f" % rwp)

    def test_separate(self):
        "test separate with a mask. issue #209 regression test"
        msk = self.data < 100
        res = self.ai.separate(self.data, mask=msk)
        bragg, amorphous = res

        self.assertLess(amorphous.max(), bragg.max(), "bragg is more intense than amorphous")
        self.assertLess(amorphous.std(), bragg.std(), "bragg is more variatic than amorphous")
        self.assertGreater(numpy.diff(res.radial).min(), 0, "radial position is stricly monotonic")
        self.assertEqual(res.radial.shape, res.intensity.shape, "1D intensities are of proper shape")

    @unittest.skipIf(UtilsTest.low_mem, "test using >100Mb")
    def test_medfilt1d(self):
        N = 1000
        param = {"unit": "2th_deg"}
        # legacy version"
        if UtilsTest.opencl and pyopencl:
            with logging_disabled(logging.WARNING):
                ref = self.ai.medfilt1d_legacy(self.data, N, method="bbox_csr", **param)
                ocl = self.ai.medfilt1d_legacy(self.data, N, method="bbox_ocl_csr", **param)
            rwp = mathutil.rwp(ref, ocl)
            logger.info("test_medfilt1d legacy median Rwp = %.3f", rwp)
            self.assertLess(rwp, 1, "Rwp medfilt1d Cython/OpenCL: %.3f" % rwp)

            with logging_disabled(logging.WARNING):
                ref = self.ai.medfilt1d_legacy(self.data, N, method="bbox_csr", percentile=(20, 80), **param)
                ocl = self.ai.medfilt1d_legacy(self.data, N, method="bbox_ocl_csr", percentile=(20, 80), **param)
            rwp = mathutil.rwp(ref, ocl)
            logger.info("test_medfilt1d legacy trimmed-mean Rwp = %.3f", rwp)
            self.assertLess(rwp, 3, "Rwp trimmed-mean Cython/OpenCL: %.3f" % rwp)

        # new version"
        ref = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", "cython"), **param)
        pyt = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", "python"), **param)
        rwp_pyt = mathutil.rwp(ref, pyt)
        logger.info("test_medfilt1d ng median Rwp_python = %.3f", rwp_pyt)
        self.assertLess(rwp_pyt, 0.1, "Rwp medfilt1d_ng Cython/Python: %.3f" % rwp_pyt)

        if UtilsTest.opencl and pyopencl:
            ocl = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", "opencl"), **param)
            rwp_ocl = mathutil.rwp(ref, ocl)
            logger.info("test_medfilt1d ng median Rwp_opencl = %.3f", rwp_ocl)
            self.assertLess(rwp_ocl, 0.1, "Rwp medfilt1d_ng Cython/OpenCL: %.3f" % rwp_ocl)

        ref = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", "cython"), percentile=(20, 80), **param)
        ref = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", "python"), percentile=(20, 80), **param)
        rwp_pyt = mathutil.rwp(ref, pyt)
        logger.info("test_medfilt1d ng trimmed-mean Rwp_python = %.3f", rwp_pyt)
        self.assertLess(rwp_pyt, 2, "Rwp trimmed-mean Cython/Python: %.3f" % rwp_pyt)
        if UtilsTest.opencl and pyopencl:
            ocl = self.ai.medfilt1d_ng(self.data, N, method=("no", "csr", 'opencl'), percentile=(20, 80), **param)
            rwp_ocl = mathutil.rwp(ref, ocl)
            logger.info("test_medfilt1d ng trimmed-mean Rwp_opencl = %.3f", rwp_ocl)
            self.assertLess(rwp, 0.1, "Rwp trimmed-mean Cython/OpenCL: %.3f" % rwp_ocl)
        ref = ocl = pyt = rwp = rwp_ocl = rwp_pyt = None

    def test_radial(self):
        "Non regression for #1602"
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
        self.assertGreater(res[1].min(), 120, "intensity min in ok")
        self.assertLess(res[1].max(), 10000, "intensity max in ok")

        res = self.ai.integrate_radial(self.data, npt=360, npt_rad=10,
                                       radial_range=(3.6, 3.9), radial_unit="2th_deg",
                                       method=("full", "CSR", "opencl"))
        self.assertLess(res[0].min(), -179, "chi min at -180")
        self.assertGreater(res[0].max(), 179, "chi max at +180")
        self.assertGreater(res[1].min(), 120, "intensity min in ok")
        self.assertLess(res[1].max(), 10000, "intensity max in ok")


class TestFlatimage(unittest.TestCase):
    """test the caking of a flat image"""

    @classmethod
    def setUpClass(cls):
        cls.epsilon = 1e-4
        cls.shape = (200, 201)
        cls.data = numpy.ones(cls.shape, dtype="float64")
        det = Detector(1e-4, 1e-4, max_shape=cls.shape)
        cls.ai = AzimuthalIntegrator(0.1, 1e-2, 1e-2, detector=det)

    @classmethod
    def tearDownClass(cls):
        cls.epsilon = cls.shape = cls.data = cls.ai = None

    def test_splitPixel(self):
        res = self.ai.integrate2d(self.data, 256, 256, correctSolidAngle=False, dummy=-1.0,
                           method='splitpixel', unit='2th_deg')
        I = res[0]
        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig, ax = pylab.subplots()
            fig.suptitle('cacking of a flat image: SplitPixel')
            ax.imshow(I, interpolation="nearest")
            fig.show()
            input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon

    def test_splitBBox(self):
        I = self.ai.integrate2d(self.data, 256, 256, correctSolidAngle=False, dummy=-1.0,
                           unit="2th_deg", method='splitbbox')[0]

        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig, ax = pylab.subplots()
            fig.suptitle('caking of a flat image: SplitBBox')
            ax.imshow(I, interpolation="nearest")
            fig.show()
            input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon

    def test_guess_bins(self):
        "This test can be rather noisy on 32bits platforms !!!"
        res = self.ai.guess_max_bins(unit="2th_deg")
        self.assertEqual(res, 240, "the number of bins found is correct (240)")

    def test_guess_rad(self):
        res = self.ai.guess_npt_rad()
        self.assertEqual(res, 142, "the number of bins found is correct (142)")

    def test_guess_polarization(self):
        with fabio.open(UtilsTest.getimage("Eiger4M.edf")) as fimg:
            img = fimg.data
        ai = AzimuthalIntegrator.sload(UtilsTest.getimage("Eiger4M.poni"))
        self.assertLess(abs(ai.guess_polarization(img) - 0.5), 0.1)


class TestSaxs(unittest.TestCase):
    saxsPilatus = "bsa_013_01.edf"
    maskFile = "Pcon_01Apr_msk.edf"
    maskRef = "bioSaxsMaskOnly.edf"

    def setUp(self):
        self.edfPilatus = UtilsTest.getimage(self.__class__.saxsPilatus)
        self.maskFile = UtilsTest.getimage(self.__class__.maskFile)
        self.maskRef = UtilsTest.getimage(self.__class__.maskRef)

    def tearDown(self):
        unittest.TestCase.tearDown(self)
        self.edfPilatus = self.maskFile = self.maskRef = None

    def test_mask(self):
        """test the generation of mask"""
        ai = AzimuthalIntegrator(detector="Pilatus1M")
        ai.wavelength = 1e-10

        with fabio.open(self.edfPilatus) as fimg:
            data = fimg.data
        with fabio.open(self.maskFile) as fimg:
            mask = fimg.data
        with fabio.open(self.maskRef) as fimg:
            maskRef = fimg.data
        self.assertTrue(abs(ai.create_mask(data, mask=mask).astype(int) - maskRef).max() == 0, "test without dummy")
        # self.assertTrue(abs(self.ai.create_mask(data, mask=mask, dummy=-48912, delta_dummy=40000).astype(int) - fabio.open(self.maskDummy).data).max() == 0, "test_dummy")

    def test_positive_mask(self):
        ai = AzimuthalIntegrator()
        data = numpy.array([[0, 1, 2, 3, 4]])
        mask = numpy.array([[0, 0, 0, 1, 2]])
        result = ai.create_mask(data, mask)
        self.assertEqual(list(result[0]), [False, False, False, True, True])

    def test_negative_mask(self):
        ai = AzimuthalIntegrator()
        data = numpy.array([[0, 1, 2, 3, 4]])
        mask = numpy.array([[0, 0, 0, -2, -1]])
        result = ai.create_mask(data, mask)
        self.assertEqual(list(result[0]), [False, False, False, True, True])

    def test_bool_mask(self):
        ai = AzimuthalIntegrator()
        ai.USE_LEGACY_MASK_NORMALIZATION = True
        data = numpy.array([[0, 1, 2, 3, 4]])
        mask = numpy.array([[False, False, False, True, True]])
        result = ai.create_mask(data, mask)
        self.assertEqual(list(result[0]), [False, False, False, True, True])

    def test_legacy_mask(self):
        ai = AzimuthalIntegrator()
        ai.USE_LEGACY_MASK_NORMALIZATION = True
        data = numpy.array([[0, 1, 2, 3]])
        mask = numpy.array([[0, 1, 1, 1]])
        with logging_disabled(logging.WARNING):
            result = ai.create_mask(data, mask, mode="numpy")
        self.assertEqual(list(result[0]), [False, True, True, True])

        data = numpy.array([[0, 1, 2, 3]])
        mask = numpy.array([[1, 0, 0, 0]])
        with logging_disabled(logging.WARNING):
            result = ai.create_mask(data, mask, mode="numpy")
        self.assertEqual(list(result[0]), [False, True, True, True])

    def test_no_legacy_mask(self):
        ai = AzimuthalIntegrator()
        ai.USE_LEGACY_MASK_NORMALIZATION = False
        data = numpy.array([[0, 1, 2, 3]])
        mask = numpy.array([[0, 1, 1, 1]])
        result = ai.create_mask(data, mask, mode="numpy")
        self.assertEqual(list(result[0]), [True, False, False, False])
        data = numpy.array([[0, 1, 2, 3]])
        mask = numpy.array([[1, 0, 0, 0]])
        result = ai.create_mask(data, mask, mode="numpy")
        self.assertEqual(list(result[0]), [False, True, True, True])

    def test_normalization_factor(self):

        ai = AzimuthalIntegrator(detector="Imxpad S10")
        ai.wavelength = 1e-10
        methods = ["cython", "numpy", "lut", "csr", "splitpixel"]
        if UtilsTest.opencl and os.name != 'nt':
            methods.extend(["ocl_lut", "ocl_csr"])

        ref1d = {}
        ref2d = {}
        with fabio.open(self.edfPilatus) as fimg:
            data = fimg.data[:ai.detector.shape[0],:ai.detector.shape[1]]
        for method in methods:
            logger.debug("TestSaxs.test_normalization_factor method= " + method)
            ref1d[method + "_1"] = ai.integrate1d_ng(copy.deepcopy(data), 100, method=method, error_model="poisson")
            ref1d[method + "_10"] = ai.integrate1d_ng(copy.deepcopy(data), 100, method=method, normalization_factor=10, error_model="poisson")
            ratio_i = ref1d[method + "_1"].intensity.mean() / ref1d[method + "_10"].intensity.mean()
            ratio_s = ref1d[method + "_1"].sigma.mean() / ref1d[method + "_10"].sigma.mean()
            self.assertAlmostEqual(ratio_i, 10.0, places=3, msg=f"test_normalization_factor 1d intensity Method: {ref1d[method + '_1'].method} ratio: {ratio_i} expected 10")
            self.assertAlmostEqual(ratio_s, 10.0, places=3, msg=f"test_normalization_factor 1d sigma Method: {ref1d[method + '_1'].method} ratio: {ratio_s} expected 10")
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
        with fabio.open(self.edfPilatus) as fimg:
            img = fimg.data
        ai = AzimuthalIntegrator(detector="Pilatus1M")
        ai.setFit2D(2000, 870, 102.123456789)  # rational numbers are hell !
        mask = img < 0
        inp = ai.inpainting(img, mask)
        neg = (inp < 0).sum()
        logger.debug("neg=%s" % neg)
        self.assertTrue(neg == 0, "all negative pixels got inpainted actually all but %s" % neg)
        self.assertTrue(mask.sum() > 0, "some pixel needed inpainting")

    def test_variance(self):
        "tests the different variance model available"
        with fabio.open(self.edfPilatus) as fimg:
            img = fimg.data
        ai = AzimuthalIntegrator(pixel1=172e-6, pixel2=172e-6)
        ai.setFit2D(2000, 870, 102.123456789)  # rational numbers are hell !
        ai.wavelength = 1e-10
        mask = img < 0
        res_poisson = ai.integrate1d_ng(img, 1000, mask=mask, error_model="poisson")
        self.assertGreater(res_poisson.sigma.min(), 0, "Poisson error are positive")
        res_azimuthal = ai.integrate1d_ng(img, 1000, mask=mask, error_model="azimuthal")
        self.assertGreater(res_azimuthal.sigma.min(), 0, "Azimuthal error are positive")

    def test_empty(self):
        """Non regression about #1760"""
        ai = AzimuthalIntegrator(detector="Imxpad S10", wavelength=1e-10)
        img = numpy.empty(ai.detector.shape)
        ref = ai.empty
        target = -42
        self.assertNotEqual(ref, target, "buggy test !")
        for m in ("LUT", "CSR", "CSC"):
            ai.integrate1d(img, 100, method=("no", m, "cython"))
        for k, v in ai.engines.items():
            self.assertEqual(v.engine.empty, ref, k)
        ai.empty = target
        for k, v in ai.engines.items():
            self.assertEqual(v.engine.empty, target, k)
        ai.empty = ref
        for k, v in ai.engines.items():
            self.assertEqual(v.engine.empty, ref, k)

    def test_empty_csr(self):
        ai = AzimuthalIntegrator(detector="Imxpad S10", wavelength=1e-10)
        with self.assertLogs('pyFAI.ext.sparse_builder', level='WARNING') as cm:
            ai.setup_sparse_integrator(shape=ai.detector.shape, npt=100,
                                       pos0_range=(90, 100),
                                       unit="2th_deg",
                                       split='no',
                                       algo='CSR',
                                       empty=None,
                                       scale=True)
            self.assertTrue(cm.output[0].startswith('WARNING:pyFAI.ext.sparse_builder:Sparse matrix is empty. Expect errors or non-sense results!'),
                            "Actually emits the expected warning")


class TestSetter(unittest.TestCase):

    def setUp(self):
        self.ai = AzimuthalIntegrator()
        shape = (10, 15)
        rng = UtilsTest.get_rng()
        self.rnd1 = rng.random(shape).astype(numpy.float32)
        self.rnd2 = rng.random(shape).astype(numpy.float32)

        fd, self.edf1 = UtilsTest.tempfile(".edf", "testAI1", dir=__class__.__name__)
        os.close(fd)
        fd, self.edf2 = UtilsTest.tempfile(".edf", "testAI2", dir=__class__.__name__)
        os.close(fd)
        fabio.edfimage.edfimage(data=self.rnd1).write(self.edf1)
        fabio.edfimage.edfimage(data=self.rnd2).write(self.edf2)

    def test_flat(self):
        with logging_disabled(logging.WARNING):
            self.ai.set_flatfiles((self.edf1, self.edf2), method="mean")
            self.assertTrue(self.ai.flatfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "flatfiles string is OK")
        self.assertTrue(abs(self.ai.flatfield - 0.5 * (self.rnd1 + self.rnd2)).max() == 0, "Flat array is OK")

    def test_dark(self):
        with logging_disabled(logging.WARNING):
            self.ai.set_darkfiles((self.edf1, self.edf2), method="mean")
            self.assertTrue(self.ai.darkfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "darkfiles string is OK")
        self.assertTrue(abs(self.ai.darkcurrent - 0.5 * (self.rnd1 + self.rnd2)).max() == 0, "Dark array is OK")


class TestIntergrationNextGeneration(unittest.TestCase):

    def test_histo(self):
        det = detector_factory("Imxpad S10")
        data = UtilsTest.get_rng().random(det.shape)
        ai = AzimuthalIntegrator(detector=det, wavelength=1e-10)

        method = ("no", "histogram", "python")
        python = ai._integrate1d_ng(data, 100, method=method, error_model="poisson")
        self.assertEqual(python.compute_engine, "pyFAI.engines.histogram_engine.histogram1d_engine")
        self.assertEqual(str(python.unit), "q_nm^-1")

        method = ("no", "histogram", "cython")
        cython = ai._integrate1d_ng(data, 100, method=method, error_model="poisson")

        self.assertEqual(cython.compute_engine, "pyFAI.ext.histogram.histogram1d_engine")
        self.assertEqual(str(cython.unit), "q_nm^-1")
        self.assertTrue(numpy.allclose(cython.radial, python.radial), "cython position are the same")
        self.assertTrue(numpy.allclose(cython.intensity, python.intensity), "cython intensities are the same")
        self.assertTrue(numpy.allclose(cython.sigma, python.sigma), f"cython errors are the same, aerr={(abs(python.sigma - cython.sigma)).max()}")
        self.assertTrue(numpy.allclose(cython.sum_signal, python.sum_signal), "cython sum_signal are the same")
        self.assertTrue(numpy.allclose(cython.sum_variance, python.sum_variance), "cython sum_variance are the same")
        self.assertTrue(numpy.allclose(cython.sum_normalization, python.sum_normalization), "cython sum_normalization are the same")
        self.assertTrue(numpy.allclose(cython.count, python.count), "cython count are the same")

        method = ("no", "histogram", "opencl")
        actual_method = ai._normalize_method(method=method, dim=1, default=ai.DEFAULT_METHOD_1D).method[1:4]
        if actual_method != method:
            reason = "Skipping TestIntergrationNextGeneration.test_histo as OpenCL method not available"
            self.skipTest(reason)
        opencl = ai._integrate1d_ng(data, 100, method=method, error_model="poisson")
        self.assertEqual(opencl.compute_engine, "pyFAI.opencl.azim_hist.OCL_Histogram1d")
        self.assertEqual(str(opencl.unit), "q_nm^-1")
        self.assertTrue(numpy.allclose(opencl.radial, python.radial), "opencl position are the same")
        self.assertTrue(numpy.allclose(opencl.intensity, python.intensity), "opencl intensities are the same")
        self.assertTrue(numpy.allclose(opencl.sigma, python.sigma), "opencl errors are the same")
        self.assertTrue(numpy.allclose(opencl.sum_signal.sum(axis=-1), python.sum_signal), "opencl sum_signal are the same")
        self.assertTrue(numpy.allclose(opencl.sum_variance.sum(axis=-1), python.sum_variance),
                        f"opencl sum_variance are the same {abs(opencl.sum_variance.sum(axis=-1) - python.sum_variance).max()}")
        self.assertTrue(numpy.allclose(opencl.sum_normalization.sum(axis=-1), python.sum_normalization), "opencl sum_normalization are the same")
        self.assertTrue(numpy.allclose(opencl.count, python.count), "opencl count are the same")


class TestRange(unittest.TestCase):
    """Test for reduced range in radia/azimuthal direction"""

    @classmethod
    def setUpClass(cls):
        detector = detector_factory("Pilatus 200k")
        shape = detector.shape
        with fabio.open(UtilsTest.getimage("Pilatus1M.edf")) as fimg:
            cls.img = fimg.data[:shape[0], :shape[1]]
        cls.ai = AzimuthalIntegrator.sload(UtilsTest.getimage("Pilatus1M.poni"))
        cls.ai.detector = detector
        cls.unit = "r_mm"
        cls.azim_range = (-90, 90)
        cls.rad_range = (10, 100)
        cls.npt = 500
        # cls.ref_medfilt = cls.ai.medfilt1d(cls.img, cls.npt, unit=cls.unit)
        centerx = cls.ai.getFit2D()["centerX"]
        cls.img[:, int(centerx - 1)] = 0

    @classmethod
    def tearDownClass(cls):
        cls.unit = cls.azim_range = cls.rad_range = cls.ai = cls.img = None

    def tearDown(self) -> None:
        self.ai.reset()

    def test_medfilt(self):
        # legacy
        with logging_disabled(logging.WARNING):
            res = self.ai.medfilt1d_legacy(self.img, self.npt, unit=self.unit, azimuth_range=self.azim_range, radial_range=self.rad_range)
        self.assertGreaterEqual(res.radial.min(), min(self.rad_range))
        self.assertLessEqual(res.radial.max(), max(self.rad_range))
        # new generation
        res = self.ai.medfilt1d_ng(self.img, self.npt, unit=self.unit, azimuth_range=self.azim_range, radial_range=self.rad_range)
        self.assertGreaterEqual(res.radial.min(), min(self.rad_range))
        self.assertLessEqual(res.radial.max(), max(self.rad_range))


    def test_sigma_clip(self):
        # legacy
        with logging_disabled(logging.WARNING):
            res = self.ai._sigma_clip_legacy(self.img, self.npt, unit=self.unit, azimuth_range=self.azim_range, radial_range=self.rad_range)
        self.assertGreaterEqual(res.radial.min(), min(self.rad_range))
        self.assertLessEqual(res.radial.max(), max(self.rad_range))

        # new generation
        res = self.ai.sigma_clip_ng(self.img, self.npt, unit=self.unit, azimuth_range=self.azim_range, radial_range=self.rad_range)
        self.assertGreaterEqual(res.radial.min(), min(self.rad_range))
        self.assertLessEqual(res.radial.max(), max(self.rad_range))

    def test_sigma_clip_ng(self):
        for case in ({"error_model":"poisson", "max_iter":3, "thres":6},
                     {"error_model":"azimuthal", "max_iter":3, "thres":0},
                     ):
            results = {}
            for impl in ('python',  # Python is already fixed, please fix the 2 others
                         'cython',
                         # 'opencl' #TODO
                         ):
                try:
                    res = self.ai.sigma_clip_ng(self.img, self.npt, unit=self.unit,
                                                azimuth_range=self.azim_range, radial_range=self.rad_range,
                                                method=("no", "csr", impl),
                                                **case)
                except RuntimeError as err:
                    logger.warning("got RuntimeError with impl %s: %s case: %s", impl, err, case)
                    continue
                else:
                    results[impl] = res
                self.assertGreaterEqual(res.radial.min(), min(self.rad_range), msg=f"impl: {impl}, case {case}")
                self.assertLessEqual(res.radial.max(), max(self.rad_range), msg=f"impl: {impl}, case {case}")
            ref = results['python']
            for what, tol in (("radial", 1e-8),
                              ("count", 1),
                              # ("intensity", 1e-6),
                              # ("sigma", 1e-6),
                              ("sum_normalization", 1e-1),
                              ("count", 1e-1)):
                for impl in results:
                    obt = results[impl]
                    # print(impl, what, obt.__getattribute__(what).max(),
                    # abs(ref.__getattribute__(what) - obt.__getattribute__(what)).max(),
                    # abs((ref.__getattribute__(what) - obt.__getattribute__(what)) / ref.__getattribute__(what)).max())
                    self.assertTrue(numpy.allclose(obt.__getattribute__(what), ref.__getattribute__(what), atol=10, rtol=tol),
                                    msg=f"Sigma clipping matches for impl {impl} on paramter {what} with error_model {case['error_model']}")

    def test_variance_2d(self, error_model="poisson"):
        """This test checks that the variance is actually calculated and positive
        for all integration methods available"""

        # def print_mem():
        #     try:
        #         import psutil
        #     except:
        #         logger.error("psutil missing")
        #     else:
        #         logger.warning("Memory consumption: %s",psutil.virtual_memory())
        ai = AzimuthalIntegrator.sload(self.ai)  # make an empty copy and work on just one module of the detector (much faster)
        ai.detector = detector_factory("Imxpad S10")
        img = self.img[:ai.detector.shape[0],:ai.detector.shape[1]]

        methods = { k.method[1:4]:k for k in  IntegrationMethod.select_method(dim=2)}
        logger.info("methods investigated" + "\n".join([str(i) for i in methods.values()]))

        error_model = ErrorModel.parse(error_model)
        if error_model == ErrorModel.VARIANCE:
            variance = numpy.maximum(1, self.img)
        else:
            variance = None
        failed = []
        for m in methods.values():
            res = ai.integrate2d_ng(img, 11, 13, variance=variance, error_model=error_model, method=m)
            # ai.reset()
            v = res.sum_variance
            if v.min() < 0:
                failed.append(f"min variance is positive or null with {res.method}, error model {error_model.as_str()}")
                # print_mem()
            if v.max() <= 0:
                failed.append(f"max variance is strictly positive with {res.method}, error model {error_model.as_str()}")
                # print_mem()
            s = res.sigma
            if s.min() < 0:
                failed.append(f"min sigma is positive or null with {res.method}, error model {error_model.as_str()}")
                # print_mem()
            if s.max() <= 0:
                failed.append(f"max sigma is strictly positive with {res.method}, error model {error_model.as_str()}")
                # print_mem()
        for err_msg in failed:
            logger.error(err_msg)
        self.assertEqual(len(failed), 0, f"Number of failed tests in test_variance_2d: {len(failed)}")


class TestFlexible2D(unittest.TestCase):
    """Test integration in non-azimuthal 2D unit"""

    @classmethod
    def setUpClass(cls):
        with fabio.open(UtilsTest.getimage("moke.tif")) as fimg:
            cls.img = fimg.data
        det = detector_factory("Detector", {"pixel1":1e-4, "pixel2":1e-4})
        ai = AzimuthalIntegrator(detector=det, wavelength=1e-10)
        ai.setFit2D(100, 300, 300)
        cls.ai = ai

    @classmethod
    def tearDownClass(cls) -> None:
        super(TestFlexible2D, cls).tearDownClass()
        cls.ai = cls.img = None

    def test_flexible(self):
        for m in IntegrationMethod.select_method(dim=2, impl="cython"):
            res = self.ai.integrate2d(self.img, 50, 50, method=m, unit=("qx_nm^-1", "qy_nm^-1"))
            img, rad, azim = res
            self.assertTrue(numpy.nanmax(img) > 0, f"image is non empty for {m}")
            radmax = rad.max()
            radmin = rad.min()
            self.assertTrue(15 < radmax < 20, f"Upper bound radial is  15<{radmax}<20 for {m}")
            self.assertTrue(-20 < radmin < -15, f"Lower bound radial is  -20<{radmin}<-15 for {m}")
            azimax = azim.max()
            azimin = azim.min()
            self.assertTrue(10 < azimax < 20, f"Upper bound azimuthal is  10<{azimax}<20 for {m} ")
            self.assertTrue(-20 < azimin < -15, f"Lower bound azimuthal is  -20<{azimin}<-15 for {m}")


class TestUnweighted(unittest.TestCase):
    """Test for validating weighted/unweighted average provide correct results"""

    @classmethod
    def setUpClass(cls):
        rng = UtilsTest.get_rng()
        det = detector_factory("imxpad_s10") #very small detector, 10kpix
        # det = detector_factory("mythen") #very small detector, 1kpix
        # det = detector_factory("pilatus100k") #very small detector, 100kpix
        cls.img = rng.uniform(0.5, 1.5, det.shape)
        cls.ai = AzimuthalIntegrator(detector=det)
        cls.kwargs = {"flat": cls.img,
                      "unit": "r_mm",
                      "correctSolidAngle": False}

    @classmethod
    def tearDownClass(cls) -> None:
        cls.ai = cls.img = cls.kwargs = None

    def test_weighted(self):

        done = set()
        for method in IntegrationMethod._registry.values():
            self.ai.reset()
            if method.method[:3] in done:
                continue
            method = method.weighted
            try:
                if method.dim == 1:
                    res = self.ai.integrate1d(self.img, 10, method=method, **self.kwargs)
                elif method.dim == 2:
                    res = self.ai.integrate2d(self.img, 10, method=method, **self.kwargs)
            except Exception as err:
                print("Unable to integrate using method", method)
                raise err
            sum_signal = res.sum_signal
            sum_normalization = res.sum_normalization
            count = res.count
            if method.impl == "OpenCL":
                done.add(method.method[:3])
                if sum_signal.shape != count.shape:
                    sum_normalization = sum_normalization[..., 0]
                    sum_signal = sum_signal[..., 0]
            try:
                self.assertTrue(numpy.allclose(sum_signal, sum_normalization), f"Weighted: signal == norm for {method}")
                self.assertFalse(numpy.allclose(sum_normalization, count), f"Weighted: norm != count for {method}")
            except Exception as err:
                self.fail(f"Weighted failed for {method} with exception {err}")

    def test_unweighted(self):
        done = set()
        for method in IntegrationMethod._registry.values():
            self.ai.reset()
            if method.method[:3] in done:
                continue
            method = method.unweighted
            if method.dim == 1:
                res = self.ai.integrate1d(self.img, 100, method=method, **self.kwargs)
            elif method.dim == 2:
                res = self.ai.integrate2d(self.img, 100, method=method, **self.kwargs)
            sum_signal = res.sum_signal
            sum_normalization = res.sum_normalization
            count = res.count
            if method.impl == "OpenCL":
                done.add(method.method[:3])
                if sum_signal.shape != count.shape:
                    sum_normalization = sum_normalization[..., 0]
                    sum_signal = sum_signal[..., 0]
            try:
                self.assertTrue(numpy.allclose(sum_signal, sum_normalization), f"Unweighted: signal == norm for {method} because signal==flat")
                self.assertTrue(numpy.allclose(sum_normalization, count), f"Unweighted: norm == count for {method}")
            except AssertionError as err:
                raise err
            except Exception as err:
                self.fail(f"Unweighted failed for {method} with exception {err}")

# Non-regression tests added for pyFAI version 2025.01
class TestRadialAzimuthalScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist = 0.1
        poni1 = 0.02
        poni2 = 0.02
        detector = detector_factory("Pilatus100k")
        wavelength = 1e-10
        cls.data = UtilsTest.get_rng().random(detector.shape)
        cls.ai = AzimuthalIntegrator(dist=dist,
                                 poni1=poni1,
                                 poni2=poni2,
                                 wavelength=wavelength,
                                 detector=detector,
                             )

    def test_limits_normal_units(self):
        qnm = units.to_unit("q_nm^-1")
        qA = units.to_unit("q_A^-1")
        chideg = units.to_unit("chi_deg")
        chirad = units.to_unit("chi_rad")
        nm_range = [10,20]
        A_range = [1,2]
        deg_range = [-30,30]
        rad_range = [-1,1]
        CONFIGS = [{"unit" : (qnm, chideg), "radial_range" : nm_range, "azimuth_range" : deg_range},
                   {"unit" : (chideg, qnm), "radial_range" : deg_range, "azimuth_range" : nm_range},
                   {"unit" : (qA, chideg), "radial_range" : A_range, "azimuth_range" : deg_range},
                   {"unit" : (chideg, qA), "radial_range" : deg_range, "azimuth_range" : A_range},
                   {"unit" : (qA, chirad), "radial_range" : A_range, "azimuth_range" : rad_range},
                   {"unit" : (chirad, qA), "radial_range" : rad_range, "azimuth_range" : A_range},
                   {"unit" : (qA, chideg), "radial_range" : A_range, "azimuth_range" : deg_range},
                   {"unit" : (chideg, qA), "radial_range" : deg_range, "azimuth_range" : A_range},
        ]
        atol = 1e-1
        self.ai.chiDiscAtPi = True
        for config in CONFIGS:
            res = self.ai.integrate2d(data=self.data, npt_azim=360, npt_rad=500, **config)
            self.assertAlmostEqual(res.radial.min(), config["radial_range"][0], delta=atol)
            self.assertAlmostEqual(res.radial.max(), config["radial_range"][1], delta=atol)
            self.assertAlmostEqual(res.azimuthal.min(), config["azimuth_range"][0], delta=atol)
            self.assertAlmostEqual(res.azimuthal.max(), config["azimuth_range"][1], delta=atol)

    def test_limits_fiber_units(self):
        ## TODO next fiber units PR
        ...


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestAzimHalfFrelon))
    testsuite.addTest(loader(TestFlatimage))
    testsuite.addTest(loader(TestSetter))
    testsuite.addTest(loader(TestSaxs))
    testsuite.addTest(loader(TestIntergrationNextGeneration))
    testsuite.addTest(loader(TestRange))
    testsuite.addTest(loader(TestFlexible2D))
    testsuite.addTest(loader(TestUnweighted))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
