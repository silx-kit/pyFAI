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

"""Test suites for calibrants"""

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/08/2018"

import unittest
import logging
import sys
import copy
import numpy
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..third_party import six
from ..calibrant import Calibrant, get_calibrant, Cell, CALIBRANT_FACTORY
from ..detectors import ALL_DETECTORS
from ..azimuthalIntegrator import AzimuthalIntegrator


class TestCalibrant(unittest.TestCase):
    """
    Test calibrant installation and loading
    """
    def test_factory(self):
        # by default we provide 11 calibrants
        count = len(CALIBRANT_FACTORY)
        self.assertTrue(count > 10, "at least 11 calibrants are available, got %s" % count)

        self.assertTrue("LaB6" in CALIBRANT_FACTORY, "LaB6 is a calibrant")

        # ensure each calibrant instance is unique
        cal1 = get_calibrant("LaB6")
        cal1.wavelength = 1e-10
        cal2 = get_calibrant("LaB6")
        self.assertTrue(cal2.wavelength is None, "calibrant is delivered without wavelength")

        # check that it is possible to instantiate all calibrant
        for _k, v in CALIBRANT_FACTORY.items():
            self.assertTrue(isinstance(v, Calibrant))

    def test_2th(self):
        lab6 = get_calibrant("LaB6")
        lab6.wavelength = 1.54e-10
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 25, "We expect 25 rings for LaB6")

        lab6.setWavelength_change2th(1e-10)
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 59, "We expect 59 rings for LaB6")

        lab6.setWavelength_change2th(2e-10)
        tth = lab6.get_2th()
        self.assertEqual(len(tth), 15, "We expect 15 rings for LaB6")

        self.assertEqual(lab6.get_2th_index(1.0, 0.04), 3, "right index picked")

    def test_fake(self):
        """test for fake image generation"""
        with_plot = (logger.getEffectiveLevel() <= logging.DEBUG)
        if with_plot:
            from matplotlib import pyplot
            fig = pyplot.figure()
            ax = fig.add_subplot(1, 1, 1)

        detectors = set(ALL_DETECTORS.values())
        for _idx, detector in enumerate(detectors):
            det = detector()
            # Skip generic detectors
            if "MAX_SHAPE" not in dir(det):
                continue
            # skip the big detectors for now
            if max(det.MAX_SHAPE) > 2000:
                continue
            ai = AzimuthalIntegrator(dist=0.01, poni1=0, poni2=0, detector=det)
            calibrant = get_calibrant("LaB6")
            calibrant.set_wavelength(1e-10)
            img = calibrant.fake_calibration_image(ai)

            if with_plot:
                ax.cla()
                ax.set_title(det.name)

                ax.imshow(img, interpolation='nearest')
                fig.show()
                six.moves.input("enter> ")
            logger.info("%s min: %s max: %s ", det.name, img.min(), img.max())
            self.assertTrue(img.shape == det.shape, "Image (%s) has the right size" % (det.name,))
            self.assertTrue(img.sum() > 0, "Image (%s) contains some data" % (det.name,))
            sys.stderr.write(".")

    def test_get_peaks(self):
        calibrant = get_calibrant("LaB6")
        calibrant.wavelength = 1e-10
        ref = calibrant.get_2th()

        delta = abs(calibrant.get_peaks() - numpy.rad2deg(ref))
        self.assertLess(delta.max(), 1e-10, "results are the same")

        self.assertEqual(len(calibrant.get_peaks("q_A^-1")), len(ref), "length is OK")

    def test_factory_create_calibrant(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        self.assertIsNot(c1, c2)
        self.assertEquals(c1, c2)

    def test_same(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        self.assertEquals(c1, c2)

    def test_same2(self):
        c1 = get_calibrant("LaB6")
        c2 = get_calibrant("LaB6")
        c1.set_wavelength(1e-10)
        c2.set_wavelength(1e-10)
        self.assertEquals(c1, c2)

    def test_not_same_dspace(self):
        # this 2 calibrant must only be used there to test the lazy-loading
        c1 = get_calibrant("LaB6_SRM660a")
        c2 = get_calibrant("LaB6_SRM660b")
        self.assertNotEquals(c1, c2)

    def test_not_same_wavelength(self):
        c1 = get_calibrant("LaB6")
        c1.set_wavelength(1e-10)
        c2 = get_calibrant("LaB6")
        self.assertNotEquals(c1, c2)

    def test_copy(self):
        c1 = get_calibrant("AgBh")
        c2 = copy.copy(c1)
        self.assertIsNot(c1, c2)
        self.assertEquals(c1, c2)
        c2.set_wavelength(1e-10)
        self.assertNotEquals(c1, c2)

    def test_hash(self):
        c1 = get_calibrant("AgBh")
        c2 = get_calibrant("AgBh")
        c3 = get_calibrant("AgBh")
        c3.set_wavelength(1e-10)
        c4 = get_calibrant("LaB6")
        store = {}
        store[c1] = True
        self.assertTrue(c1 in store)
        self.assertTrue(c2 in store)
        self.assertTrue(c3 not in store)
        self.assertTrue(c4 not in store)


class TestCell(unittest.TestCase):
    """
    Test generation of a calibrant from a cell
    """
    def test_class(self):
        c = Cell()
        self.assertAlmostEqual(c.volume, 1.0, msg="Volume of triclinic 1,1,1,90,90,90 == 1.0, got %s" % c.volume)
        c = Cell(1, 2, 3)
        self.assertAlmostEqual(c.volume, 6.0, msg="Volume of triclinic 1,2,3,90,90,90 == 6.0, got %s" % c.volume)
        c = Cell(1, 2, 3, 90, 30, 90)
        self.assertAlmostEqual(c.volume, 3.0, msg="Volume of triclinic 1,2,3,90,30,90 == 3.0, got %s" % c.volume)

    def test_classmethods(self):
        c = Cell.cubic(1)
        self.assertAlmostEqual(c.volume, 1.0, msg="Volume of cubic 1 == 1.0, got %s" % c.volume)
        c = Cell.tetragonal(2, 3)
        self.assertAlmostEqual(c.volume, 12.0, msg="Volume of tetragonal 2,3 == 12.0, got %s" % c.volume)
        c = Cell.orthorhombic(1, 2, 3)
        self.assertAlmostEqual(c.volume, 6.0, msg="Volume of orthorhombic 1,2,3 == 6.0, got %s" % c.volume)

    def test_dspacing(self):
        c = Cell.cubic(1)
        cd = c.d_spacing(0.1)
        cds = list(cd.keys())
        cds.sort()

        t = Cell()
        td = t.d_spacing(0.1)
        tds = list(td.keys())
        tds.sort()

        self.assertEquals(cds, tds, msg="d-spacings are the same")
        for k in cds:
            self.assertEquals(cd[k], td[k], msg="plans are the same for d=%s" % k)

    def test_helium(self):
        # self.skipTest("Not working")
        a = 4.242
        href = "A.F. Schuch and R.L. Mills, Phys. Rev. Lett., 1961, 6, 596."
        he = Cell.cubic(a)
        self.assertTrue(len(he.d_spacing(1)) == 15, msg="got 15 lines for He")
        he.save("He", "Helium", href, 1.0, UtilsTest.tempdir)

    def test_hydrogen(self):
        # self.skipTest("Not working")
        href = "DOI: 10.1126/science.239.4844.1131"
        h = Cell.hexagonal(2.6590, 4.3340)
        self.assertAlmostEqual(h.volume, 26.537, places=3, msg="Volume for H cell is correct")
        self.assertTrue(len(h.d_spacing(1)) == 14, msg="got 14 lines for H")
        h.save("H", "Hydrogen", href, 1.0, UtilsTest.tempdir)


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestCalibrant))
    testsuite.addTest(loader(TestCell))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
