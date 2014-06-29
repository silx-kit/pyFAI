#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    author: Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
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
"test suite for masked arrays"

__author__ = "Picca Frédéric-Emmanuel, Jérôme Kieffer",
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/06/2014"

import sys, os, tempfile, shutil
import unittest
import numpy
from utilstest import getLogger  # UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.detectors import detector_factory, ALL_DETECTORS


class TestDetector(unittest.TestCase):

    def test_detector_instanciate(self):
        """
        this method try to instanciate all the detectors
        """
        for k, v in ALL_DETECTORS.iteritems():
            v()

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
        y, x = imxpad.calc_cartesian_positions()
        self.assertEqual(imxpad._pixel_edges is None, False)

        # now check that the cached values are identical for each
        # method call
        y1, x1 = imxpad.calc_cartesian_positions()
        self.assertEqual(numpy.all(numpy.equal(y1, y)), True)
        self.assertEqual(numpy.all(numpy.equal(x1, x)), True)

        # check that a few pixel positions are ok.
        self.assertAlmostEqual(y[0, 0], 2.5 * 130e-6 / 2.)
        self.assertAlmostEqual(y[3, 0], y[2, 0] + 130e-6)
        self.assertAlmostEqual(y[119, 0], y[118, 0] + 130e-6 * 3.5 / 2.)

        self.assertAlmostEqual(x[0, 0], 2.5 * 130e-6 / 2.)
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
        known_fail = [] #TODO: fix broken detectors
        print(tmpdir)
        for det_name in ALL_DETECTORS:
            fname = os.path.join(tmpdir, det_name + ".h5")
            if os.path.exists(fname): #already tested with another alias
                continue
            print fname
            det = detector_factory(det_name)
            if (det.pixel1 is None) or (det.shape is None):
                continue
            print det

            det.save(fname)
            new_det = detector_factory(fname)
            print new_det
            for what in ("pixel1", "pixel2", "name", "max_shape", "shape", "binning"):
                if "__len__" in dir(det.__getattribute__(what)):
                    self.assertEqual(det.__getattribute__(what), new_det.__getattribute__(what), "%s is the same for %s" % (what, fname))
                else:
                    self.assertAlmostEqual(det.__getattribute__(what), new_det.__getattribute__(what), 4, "%s is the same for %s" % (what, fname))
            print det, det.shape
            r1, r2 = det.calc_cartesian_positions()
            o1, o2 = new_det.calc_cartesian_positions()
            err1 = abs(r1 - o1).max()
            err2 = abs(r2 - o2).max()
            if det.name not in known_fail:
                self.assert_(err1 < 1e-6, "precision on pixel position 1 is better than 1µm, got %e" % err2)
                self.assert_(err2 < 1e-6, "precision on pixel position 2 is better than 1µm, got %e" % err2)
            #todo: check masks
        shutil.rmtree(tmpdir)


def test_suite_all_detectors():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestDetector("test_detector_instanciate"))
    testSuite.addTest(TestDetector("test_detector_imxpad_s140"))
    testSuite.addTest(TestDetector("test_detector_rayonix_sx165"))
    testSuite.addTest(TestDetector("test_nexus_detector"))

    return testSuite

if __name__ == '__main__':
    mysuite = test_suite_all_detectors()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
