#!/usr/bin/env python
# coding: utf-8
#
#    Project: Fast Azimuthal Integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "16/05/2019"

import os
import tempfile
import shutil
import unittest
import numpy
import time
import logging
logger = logging.getLogger(__name__)
from .. import detectors
from ..detectors import detector_factory, ALL_DETECTORS
from .. import io
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
        config = {"pixel1": 1, "pixel2": 2}
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

    def test_nexus_detector(self):
        tmpdir = tempfile.mkdtemp()
        known_fail = []
        if io.h5py is None:
            logger.warning("H5py not present, skipping test_detector.TestDetector.test_nexus_detector")
            return
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

            self.assertTrue(err1 < 1e-6, "%s precision on pixel position 1 is better than 1µm, got %e" % (det_name, err1))
            self.assertTrue(err2 < 1e-6, "%s precision on pixel position 2 is better than 1µm, got %e" % (det_name, err2))
            if not det.IS_FLAT:
                err = abs(r[2] - o[2]).max()
                self.assertTrue(err < 1e-6, "%s precision on pixel position 3 is better than 1µm, got %e" % (det_name, err))

        # check Pilatus with displacement maps
        # check spline
        # check SPD sisplacement

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
        t0 = time.time()
        n = a.get_pixel_corners(use_cython=False)
        t1 = time.time()
        a._pixel_corners = None
        c = a.get_pixel_corners(use_cython=True)
        t2 = time.time()
        logger.info("Aarhus.get_pixel_corners timing Numpy: %.3fs Cython: %.3fs", t1 - t0, t2 - t1)
        self.assertTrue(abs(n - c).max() < 1e-6, "get_pixel_corners cython == numpy")
        # test pixel center coordinates
        t0 = time.time()
        n1, n2, n3 = a.calc_cartesian_positions(use_cython=False)
        t1 = time.time()
        c1, c2, c3 = a.calc_cartesian_positions(use_cython=True)
        t2 = time.time()
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


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestDetector))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
