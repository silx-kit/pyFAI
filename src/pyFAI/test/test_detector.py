#!/usr/bin/env python
# coding: utf-8
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2023 European Synchrotron Radiation Facility, Grenoble, France
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
#

"test suite for masked arrays"

__author__ = "Picca Frédéric-Emmanuel, Jérôme Kieffer",
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "MIT+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/06/2024"

import os
import shutil
import unittest
import numpy
import time
import logging
logger = logging.getLogger(__name__)
from .. import detectors
from ..detectors import detector_factory, ALL_DETECTORS
from .. import io
from .. import utils
from .utilstest import UtilsTest


class TestDetector(unittest.TestCase):

    def test_detector_instanciate(self):
        """
        this method try to instantiate all the detectors
        """
        for name, klass in ALL_DETECTORS.items():
            det = klass()
            config = det.get_config()
            first = detector_factory(name, config)
            res = first == det
            logger.debug("Detector name: %s config %s, same as factory %s",
                         name, config, res)

            self.assertEqual(res, True, name)
            second = detector_factory(name)
            second.set_config(config)
            res = first == det
            logger.debug("Detector name: %s config %s, same as factory %s",
                         name, config, res)
            self.assertEqual(res, True, name)

    def test_reading_non_default_args(self):
        config = {"pixel1": 1, "pixel2": 2, "orientation":3}
        detector = detector_factory("adsc_q315", config)
        self.assertEqual(detector.get_config(), config)
        self.assertEqual(detector.pixel1, config["pixel1"])
        self.assertEqual(detector.pixel2, config["pixel2"])

    def test_detector_imxpad_s140(self):
        """
        The masked image has a masked ring around 1.5deg with value
        -10 without mask the pixels should be at -10 ; with mask they
        are at 0
        """
        imxpad = detector_factory("imxpad_s140")

        # check that the cartesian coordinates is cached
        self.assertEqual(hasattr(imxpad, '_pixel_edges'), True)
        self.assertEqual(imxpad._pixel_edges, None)
        y, x, z = imxpad.calc_cartesian_positions()
        self.assertEqual(imxpad._pixel_edges is None, False)

        # now check that the cached values are identical for each
        # method call
        y1, x1, z1 = imxpad.calc_cartesian_positions()
        self.assertEqual(numpy.all(numpy.equal(y1, y)), True)
        self.assertEqual(numpy.all(numpy.equal(x1, x)), True)
        self.assertEqual(z, None)
        self.assertEqual(z1, None)
        # check that a few pixel positions are ok.
        self.assertAlmostEqual(y[0, 0], 1 * 130e-6 / 2.)
        self.assertAlmostEqual(y[3, 0], y[2, 0] + 130e-6)
        self.assertAlmostEqual(y[119, 0], y[118, 0] + 130e-6 * 3.5 / 2.)

        self.assertAlmostEqual(x[0, 0], 1 * 130e-6 / 2.)
        self.assertAlmostEqual(x[0, 3], x[0, 2] + 130e-6)
        self.assertAlmostEqual(x[0, 79], x[0, 78] + 130e-6 * 3.5 / 2.)

    def test_detector_rayonix_sx165(self):
        """
        rayonix detectors have different pixel size depending on the binning.
        Check that the set_binning method works for the sx_165

        #personal communication of M. Blum:

        self.desired_pixelsizes[4096]        = 39.500
        self.desired_pixelsizes[2048]        = 79.000
        self.desired_pixelsizes[1364]        = 118.616
        self.desired_pixelsizes[1024]        = 158.000
        self.desired_pixelsizes[512]        = 316.000

        """
        sx165 = detector_factory("rayonixsx165")

        # check the default pixels size and the default binning
        self.assertAlmostEqual(sx165.pixel1, 395e-7)
        self.assertAlmostEqual(sx165.pixel2, 395e-7)
        self.assertEqual(sx165.binning, (1, 1))

        # check binning 1
        sx165.binning = 1
        self.assertAlmostEqual(sx165.pixel1, 395e-7)
        self.assertAlmostEqual(sx165.pixel2, 395e-7)
        self.assertEqual(sx165.binning, (1, 1))

        # check binning 2
        sx165.binning = 2
        self.assertAlmostEqual(sx165.pixel1, 79e-6)
        self.assertAlmostEqual(sx165.pixel2, 79e-6)
        self.assertEqual(sx165.binning, (2, 2))

        # check binning 4
        sx165.binning = 4
        self.assertAlmostEqual(sx165.pixel1, 158e-6)
        self.assertAlmostEqual(sx165.pixel2, 158e-6)
        self.assertEqual(sx165.binning, (4, 4))

        # check binning 8
        sx165.binning = 8
        self.assertAlmostEqual(sx165.pixel1, 316e-6)
        self.assertAlmostEqual(sx165.pixel2, 316e-6)
        self.assertEqual(sx165.binning, (8, 8))

        # check a non standard binning
        sx165.binning = 10
        self.assertAlmostEqual(sx165.pixel1, sx165.pixel2)

    def test_detector_jungfrau(self):
        j = detector_factory("Jungfrau")
        last = j.get_pixel_corners()[-1, -1]
        self.assertAlmostEqual(last[:, 1].max(), 514 * 75e-6, places=7, msg="height match")
        self.assertAlmostEqual(last[:, 2].max(), 1030 * 75e-6, places=7, msg="width match")

    def test_nexus_detector(self):
        if io.h5py is None:
            logger.warning("H5py not present, skipping test_detector.TestDetector.test_nexus_detector")
            raise unittest.SkipTest("H5py not present, skipping test_detector.TestDetector.test_nexus_detector")
        tmpdir = os.path.join(UtilsTest.tempdir, "test_nexus_detector")
        os.makedirs(tmpdir)
        known_fail = []
        for det_name in ALL_DETECTORS:
            fname = os.path.join(tmpdir, det_name + ".h5")
            if os.path.exists(fname):  # already tested with another alias
                continue
            det = detector_factory(det_name)
            logger.debug("%s --> nxs", det_name)
            if (det.pixel1 is None) or (det.shape is None):
                continue
            if (det.shape[0] > 1900) or (det.shape[1] > 1900):
                continue

            det.save(fname)
            new_det = detector_factory(fname)
            for what in ("pixel1", "pixel2", "name", "max_shape", "shape", "binning"):
                if "__len__" in dir(det.__getattribute__(what)):
                    self.assertEqual(det.__getattribute__(what), new_det.__getattribute__(what), "%s is the same for %s" % (what, fname))
                else:
                    self.assertAlmostEqual(det.__getattribute__(what), new_det.__getattribute__(what), 4, "%s is the same for %s" % (what, fname))
            if (det.mask is not None) or (new_det.mask is not None):
                self.assertTrue(numpy.allclose(det.mask, new_det.mask), "%s mask is not the same" % det_name)

            if det.shape[0] > 2000:
                continue
            try:
                r = det.calc_cartesian_positions()
                o = new_det.calc_cartesian_positions()
            except MemoryError:
                logger.warning("Test nexus_detector failed due to short memory on detector %s", det_name)
                continue
            self.assertEqual(len(o), len(r), "data have same dimension")

            err1 = abs(r[0] - o[0]).max()
            err2 = abs(r[1] - o[1]).max()
            if det.name in known_fail:
                continue
            if err1 > 1e-6:
                logger.error("%s precision on pixel position 1 is better than 1µm, got %e", det_name, err1)
            if err2 > 1e-6:
                logger.error("%s precision on pixel position 1 is better than 1µm, got %e", det_name, err2)

            self.assertLess(err1, 1e-6, f"{det_name} precision on pixel position 1 is better than 1µm, got {err1:e}")
            self.assertLess(err2, 1e-6, f"{det_name} precision on pixel position 2 is better than 1µm, got {err1:e}")
            if not det.IS_FLAT:
                err = abs(r[2] - o[2]).max()
                self.assertTrue(err < 1e-6, "%s precision on pixel position 3 is better than 1µm, got %e" % (det_name, err))
            self.assertEqual(det.CORNERS, new_det.CORNERS, "Number of pixel corner is consistent")
        # check Pilatus with displacement maps
        # check spline
        # check SPD displacement

        shutil.rmtree(tmpdir)

    def test_guess_binning(self):

        # Mar 345 2300 pixels with 150 micron size
        mar = detector_factory("mar345")
        shape = 2300, 2300
        mar.guess_binning(shape)
        self.assertEqual(shape, mar.mask.shape, "Mar345 detector has right mask shape")
        self.assertEqual(mar.pixel1, 150e-6, "Mar345 detector has pixel size 150µ")

        mar = detector_factory("mar345")
        shape = 3450, 3450
        mar.guess_binning(shape)
        self.assertEqual(shape, mar.mask.shape, "Mar345 detector has right mask shape")
        self.assertEqual(mar.pixel1, 100e-6, "Mar345 detector has pixel size 100µ")

        mar = detector_factory("mar165")
        shape = 1364, 1364
        mar.guess_binning(shape)
        self.assertEqual(shape, mar.mask.shape, "Mar165 detector has right mask shape")
        self.assertEqual(mar.pixel1, 118.616e-6, "Mar166 detector has pixel size 118.616µ")
        self.assertEqual(mar.binning, (3, 3), "Mar165 has 3x3 binning")

        mar = detector_factory("RayonixLx170")
        shape = 192, 384
        mar.guess_binning(shape)
        self.assertEqual(mar.binning, (10, 10), "RayonixLx170 has 10x10 binning")

        p = detector_factory("Perkin")
        self.assertEqual(p.pixel1, 200e-6, "raw detector has good pixel size")
        self.assertEqual(p.binning, (2, 2), "raw detector has good pixel binning")
        p.guess_binning((4096, 4096))
        self.assertEqual(p.pixel1, 100e-6, "unbinned detector has good pixel size")
        self.assertEqual(p.binning, (1, 1), "unbinned detector has good pixel binning")

    def test_Xpad_flat(self):
        d = detector_factory("Xpad S540 flat")
        cy = d.calc_cartesian_positions(use_cython=True)
        np = d.calc_cartesian_positions(use_cython=False)
        self.assertTrue(numpy.allclose(cy[0], np[0]), "max_delta1=" % abs(cy[0] - np[0]).max())
        self.assertTrue(numpy.allclose(cy[1], np[1]), "max_delta2=" % abs(cy[1] - np[1]).max())

    def test_non_flat(self):
        """
        tests specific to non flat detectors to ensure consistency
        """
        a = detector_factory("Aarhus")
        # to limit the memory footprint, devide size by 100
        a.binning = (10, 10)
        t0 = time.perf_counter()
        n = a.get_pixel_corners(use_cython=False)
        t1 = time.perf_counter()
        a._pixel_corners = None
        c = a.get_pixel_corners(use_cython=True)
        t2 = time.perf_counter()
        logger.info("Aarhus.get_pixel_corners timing Numpy: %.3fs Cython: %.3fs", t1 - t0, t2 - t1)
        self.assertTrue(abs(n - c).max() < 1e-6, "get_pixel_corners cython == numpy")
        # test pixel center coordinates
        t0 = time.perf_counter()
        n1, n2, n3 = a.calc_cartesian_positions(use_cython=False)
        t1 = time.perf_counter()
        c1, c2, c3 = a.calc_cartesian_positions(use_cython=True)
        t2 = time.perf_counter()
        logger.info("Aarhus.calc_cartesian_positions timing Numpy: %.3fs Cython: %.3fs", t1 - t0, t2 - t1)
        self.assertTrue(abs(n1 - c1).max() < 1e-6, "cartesian coord1 cython == numpy")
        self.assertTrue(abs(n2 - c2).max() < 1e-6, "cartesian coord2 cython == numpy")
        self.assertTrue(abs(n3 - c3).max() < 1e-6, "cartesian coord3 cython == numpy")

    def test_nexus_copy(self):
        filename = os.path.join(UtilsTest.tempdir, self.id() + ".h5")
        detector = detectors.ImXPadS10()
        detector.save(filename)
        detector = detectors.NexusDetector(filename)
        import copy
        cloned = copy.copy(detector)
        numpy.testing.assert_array_almost_equal(detector.get_pixel_corners(), cloned.get_pixel_corners())

    def test_bug_1378(self):
        from ..detectors import Detector
        from ..calibrant import CalibrantFactory
        from pyFAI.geometryRefinement import GeometryRefinement
        calibrant_factory = CalibrantFactory()
        ceo2 = calibrant_factory("CeO2")
        img_shape = (280, 290)
        detector = Detector(100e-6, 110e-6)
        detector.max_shape = detector.shape = img_shape

        dx = dy = numpy.ones(shape=img_shape)

        detector.set_dx(dx)
        detector.set_dy(dy)

        pattern_geometry = GeometryRefinement([[1, 1, 0], [2, 1, 1]],
                                              dist=1,
                                              wavelength=0.3344e-10,
                                              detector=detector,
                                              calibrant=ceo2)

    def test_displacements(self):
        from ..detectors import Detector
        import copy
        detector = Detector(pixel1=90e-6, pixel2=110e-6, splineFile=None, max_shape=(110, 90))
        ref = detector.get_pixel_corners()
        detector.reset_pixel_corners()

        delta_y = 0.3  # pixel
        delta_x = 0.1  # pixel

        detector_a = copy.copy(detector)
        dx = numpy.ones(detector_a.shape) * delta_x
        dy = numpy.ones(detector_a.shape) * delta_y

        detector_a.set_dx(dx)
        detector_a.set_dy(dy)
        obt_a = detector_a.get_pixel_corners()
        self.assertTrue(numpy.allclose(obt_a[..., 1] - detector.pixel1 * delta_y, ref[..., 1]), msg="dy on center")
        self.assertTrue(numpy.allclose(obt_a[..., 2] - detector.pixel2 * delta_x, ref[..., 2]), msg="dx on center")

        detector_b = copy.copy(detector)
        big_shape = tuple(i + 1 for i in detector.shape)
        dx = numpy.ones(big_shape) * delta_x
        dy = numpy.ones(big_shape) * delta_y
        detector_b.set_dx(dx)
        detector_b.set_dy(dy)
        obt_b = detector_b.get_pixel_corners()
        self.assertTrue(numpy.allclose(obt_b[..., 1] - detector.pixel1 * delta_y, ref[..., 1]), msg="dy on edge")
        self.assertTrue(numpy.allclose(obt_b[..., 2] - detector.pixel2 * delta_x, ref[..., 2]), msg="dx on edge")

        self.assertTrue(numpy.allclose(obt_b, obt_a))

    def test_hexagonal_detector(self):
        pix = detector_factory("Pixirad1")
        self.assertEqual(pix.CORNERS, 6, "detector has 6 corners")

        wl = 1e-10
        from ..calibrant import ALL_CALIBRANTS
        from ..integrator.azimuthal import AzimuthalIntegrator
        AgBh = ALL_CALIBRANTS("AgBh")
        AgBh.wavelength = 1e-10
        ai = AzimuthalIntegrator(detector=pix, wavelength=wl)
        img = AgBh.fake_calibration_image(ai, Imax=10000, W=0.00001)

        ai.integrate1d(img, 500, method=("no", "histogram", "cython"))
        ai.integrate2d(img, 500, method=("no", "histogram", "cython"))
        ai.integrate1d(img, 500, method=("bbox", "histogram", "cython"))
        ai.integrate2d(img, 500, method=("bbox", "histogram", "cython"))
        try:
            ai.integrate1d(img, 500, method=("full", "histogram", "cython"))
            ai.integrate2d(img, 500, method=("full", "histogram", "cython"))
        except Exception as err:
            self.skipTest(f"SplitPixel does not work (yet) with hexagonal pixels: {err}")

    def test_abstract(self):
        shape = 10, 12
        det = detector_factory("detector")
        self.assertEqual(det.shape, None)
        z = numpy.zeros(shape)
        res = det.dynamic_mask(z)
        self.assertEqual(det.shape, shape)
        self.assertEqual(abs(z-res).max(), 0)

    def test_factory_warning(self):
        with self.assertLogs('pyFAI.detectors._common', level='ERROR') as cm:
            d = detector_factory("pilatus1M", {"toto": "pippo", "pixel1": 1})
            self.assertEqual(d.pixel1, 1, "taken into account")
            self.assertNotEqual(d.pixel2, 1, "default value")
            self.assertTrue("Factory: Left-over config parameters in" in  cm.output[0], "emits an error")

    def test_regression_2140(self):
        """the mask has the max_shape and not shape.
        Binning not taken into account.
        """
        d = detector_factory("Eiger2_CdTe_9M")
        binned = tuple(i//2 for i in d.shape)
        d.guess_binning(binned)
        self.assertEqual(binned, d.mask.shape, "mask has been binned as well ")


class TestOrientation(unittest.TestCase):
    @classmethod
    def setUpClass(cls)->None:
        super(TestOrientation, cls).setUpClass()
        cls.orient1 = detector_factory("Pilatus100k", config={"orientation":1})
        cls.orient2 = detector_factory("Pilatus100k", config={"orientation":2})
        cls.orient3 = detector_factory("Pilatus100k", config={"orientation":3})
        cls.orient4 = detector_factory("Pilatus100k", config={"orientation":4})

    @classmethod
    def tearDownClass(cls)->None:
        super(TestOrientation, cls).tearDownClass()
        cls.orient1 = None
        cls.orient2 = None
        cls.orient3 = None
        cls.orient4 = None

    def test_centers(self):
        p1, p2, _ = self.orient1.calc_cartesian_positions()
        #orient2 -> flip rl
        r1, r2, _ = self.orient2.calc_cartesian_positions()
        self.assertTrue(numpy.allclose(p1, numpy.fliplr(r1)), "orient 2vs1 dim1,y center")
        self.assertTrue(numpy.allclose(p2, numpy.fliplr(r2)), "orient 2vs1 dim2,x center")
        #orient3 -< rotate180
        r1, r2, _ = self.orient3.calc_cartesian_positions()
        self.assertTrue(numpy.allclose(p1, numpy.flipud(numpy.fliplr(r1))), "orient 3vs1 dim1,y center")
        self.assertTrue(numpy.allclose(p2, numpy.flipud(numpy.fliplr(r2))), "orient 3vs1 dim2,x center")
        #orient4 -> flip u-d
        r1, r2, _ = self.orient4.calc_cartesian_positions()
        self.assertTrue(numpy.allclose(p1, numpy.flipud(r1)), "orient 4vs1 dim1,y center")
        self.assertTrue(numpy.allclose(p2, numpy.flipud(r2)), "orient 4vs1 dim2,x center")

    def test_corners(self):
        p1, p2, _ = self.orient1.calc_cartesian_positions(center=False)
        #orient2 -> flip rl
        r1, r2, _ = self.orient2.calc_cartesian_positions(center=False)
        self.assertTrue(numpy.allclose(p1, numpy.fliplr(r1)), "orient 2vs1 dim1,y corner")
        self.assertTrue(numpy.allclose(p2, numpy.fliplr(r2)), "orient 2vs1 dim2,x corner")
        #orient3 -< rotate180
        r1, r2, _ = self.orient3.calc_cartesian_positions(center=False)
        self.assertTrue(numpy.allclose(p1, numpy.flipud(numpy.fliplr(r1))), "orient 3vs1 dim1,y corner")
        self.assertTrue(numpy.allclose(p2, numpy.flipud(numpy.fliplr(r2))), "orient 3vs1 dim2,x corner")
        #orient4 -> flip u-d
        r1, r2, _ = self.orient4.calc_cartesian_positions(center=False)
        self.assertTrue(numpy.allclose(p1, numpy.flipud(r1)), "orient 4vs1 dim1,y corner")
        self.assertTrue(numpy.allclose(p2, numpy.flipud(r2)), "orient 4vs1 dim2,x corner")

    def test_corners2(self):
        """similar to what is made in geometry ...."""

        shape = self.orient1.shape
        d1 = utils.expand2d(numpy.arange(shape[0] + 1.0), shape[1] + 1.0, False)
        d2 = utils.expand2d(numpy.arange(shape[1] + 1.0), shape[0] + 1.0, True)
        for orient in (self.orient1, self.orient2, self.orient3, self.orient4):
            for use_cython in (True, False):
                p1, p2, p3 = orient.calc_cartesian_positions(d1, d2, center=False, use_cython=use_cython)
                p1/=orient.pixel1
                p2/=orient.pixel2
                self.assertEqual(p3, None, f"P3 is None for {orient} with use_cython={use_cython}")
                self.assertEqual(p1.min(), 0, f"P1_min is 0 for {orient} with use_cython={use_cython}")
                self.assertEqual(p1.max(), shape[0], f"P1_max is shape for {orient} with use_cython={use_cython}")
                self.assertEqual(p2.min(), 0, f"P2_min is 0 for {orient} with use_cython={use_cython}")
                self.assertEqual(p2.max(), shape[1], f"P2_max is shape for {orient} with use_cython={use_cython}")

    def test_corners3(self):
        """similar to what is made in geometry ...."""
        for orient in (self.orient1, self.orient2, self.orient3, self.orient4):
            yc,xc, _ = orient.calc_cartesian_positions(center=1)
            tmp = orient.get_pixel_corners().mean(axis=-2)
            zm = tmp[..., 0]
            ym = tmp[..., 1]
            xm = tmp[..., 2]
            self.assertTrue(numpy.all(zm==0), f"Z is OK (detector {orient} is flat)")
            self.assertTrue(numpy.allclose(ym, yc), f"Y is OK (detector {orient})")
            self.assertTrue(numpy.allclose(xm, xc), f"X is OK (detector {orient})")


    def test_points(self):
        npt = 1000
        rng = UtilsTest.get_rng()
        Y = rng.integers(0, self.orient1.shape[0]-1, size=npt)
        X = rng.integers(0, self.orient1.shape[1]-1, size=npt)
        #orient1
        r1, r2, _ = self.orient1.calc_cartesian_positions(Y, X)
        ref1, ref2, _ = self.orient1.calc_cartesian_positions()
        p1 = ref1[Y, X]
        p2 = ref2[Y, X]
        self.assertTrue(numpy.allclose(r1, p1), "orient 1 dim1,y points")
        self.assertTrue(numpy.allclose(r2, p2), "orient 1 dim2,x points")

        #orient2
        r1, r2, _ = self.orient2.calc_cartesian_positions(Y, X)
        ref1, ref2, _ = self.orient2.calc_cartesian_positions()
        p1 = ref1[Y, X]
        p2 = ref2[Y, X]
        self.assertTrue(numpy.allclose(r1, p1), "orient 2 dim1,y points")
        self.assertTrue(numpy.allclose(r2, p2), "orient 2 dim2,x points")

        #orient3
        r1, r2, _ = self.orient3.calc_cartesian_positions(Y, X)
        ref1, ref2, _ = self.orient3.calc_cartesian_positions()
        p1 = ref1[Y, X]
        p2 = ref2[Y, X]
        self.assertTrue(numpy.allclose(r1, p1), "orient 3 dim1,y points")
        self.assertTrue(numpy.allclose(r2, p2), "orient 3 dim2,x points")

        #orient4
        r1, r2, _ = self.orient4.calc_cartesian_positions(Y, X)
        ref1, ref2, _ = self.orient4.calc_cartesian_positions()
        p1 = ref1[Y, X]
        p2 = ref2[Y, X]
        self.assertTrue(numpy.allclose(r1, p1), "orient 4 dim1,y points")
        self.assertTrue(numpy.allclose(r2, p2), "orient 4 dim2,x points")

    def test_origin(self):
        self.assertEqual(detector_factory("Pilatus100k", {"orientation":0}).origin, (0, 0))
        self.assertEqual(detector_factory("Pilatus100k", {"orientation":1}).origin,(195, 487))
        self.assertEqual(detector_factory("Pilatus100k", {"orientation":2}).origin,(195, 0))
        self.assertEqual(detector_factory("Pilatus100k", {"orientation":3}).origin,(0, 0))
        self.assertEqual(detector_factory("Pilatus100k", {"orientation":4}).origin,(0, 487))


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestDetector))
    testsuite.addTest(loader(TestOrientation))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
