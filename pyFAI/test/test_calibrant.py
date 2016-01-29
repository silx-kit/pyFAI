#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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

__doc__ = """Test suites for calibrants"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jérôme.Kieffer@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "29/01/2016"

import unittest
import logging
import sys
import os
from .utilstest import getLogger, UtilsTest
logger = getLogger(__file__)
try:
    import six
except:
    from pyFAI.third_party import six

from ..calibrant import Calibrant, ALL_CALIBRANTS, Cell
from ..detectors import ALL_DETECTORS
from .. import AzimuthalIntegrator


class TestCalibrant(unittest.TestCase):
    """
    Test calibrant installation and loading
    """
    def test_factory(self):
        # by default we provide 11 calibrants
        l = len(ALL_CALIBRANTS)
        self.assert_(l > 10, "at least 11 calibrants are available, got %s" % l)

        self.assert_("LaB6" in ALL_CALIBRANTS, "LaB6 is a calibrant")

        # ensure each calibrant instance is unique
        cal1 = ALL_CALIBRANTS["LaB6"]
        cal1.wavelength = 1e-10
        cal2 = ALL_CALIBRANTS["LaB6"]
        self.assert_(cal2.wavelength is None, "calibrant is delivered without wavelength")

        # check that it is possible to instanciate all calibrant
        for k, v in ALL_CALIBRANTS.items():
            self.assertTrue(isinstance(v, Calibrant))

    def test_2th(self):
        lab6 = ALL_CALIBRANTS["LaB6"]
        lab6.wavelength = 1.54e-10
        tth = lab6.get_2th()
        self.assert_(len(tth) == 25, "We expect 25 rings for LaB6")
        lab6.setWavelength_change2th(1e-10)
        tth = lab6.get_2th()
        self.assert_(len(tth) == 25, "We still expect 25 rings for LaB6 (some are missing lost)")
        lab6.setWavelength_change2th(2e-10)
        tth = lab6.get_2th()
        self.assert_(len(tth) == 15, "Only 15 remaining out of 25 rings for LaB6 (some additional got lost)")

    def test_fake(self):
        """test for fake image generation"""
        with_plot = (logger.getEffectiveLevel() <= logging.DEBUG)
        if with_plot:
            import matplotlib
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

        detectors = set(ALL_DETECTORS.values())
        for idx, detector in enumerate(detectors):
            det = detector()
            # Skip generic detectors
            if "MAX_SHAPE" not in dir(det):
                continue
            # skip the big detectors for now
            if max(det.MAX_SHAPE) > 2000:
                continue
            ai = AzimuthalIntegrator(dist=0.01, poni1=0, poni2=0,
                                           detector=det)
            calibrant = ALL_CALIBRANTS["LaB6"]
            calibrant.set_wavelength(1e-10)
            img = calibrant.fake_calibration_image(ai)

            if with_plot:
                ax.cla()
                ax.set_title(det.name)

                ax.imshow(img, interpolation='nearest')
                fig.show()
                six.moves.input("enter> ")
            logger.info("%s min: %s max: %s " % (det.name, img.min(), img.max()))
            self.assert_(img.shape == det.shape, "Image (%s) has the right size" % (det.name,))
            self.assert_(img.sum() > 0, "Image (%s) contains some data" % (det.name,))
            sys.stderr.write(".")


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
        a = 4.242
        href = "A.F. Schuch and R.L. Mills, Phys. Rev. Lett., 1961, 6, 596."
        he = Cell.cubic(a)
        self.assert_(len(he.d_spacing(1)) == 15, msg="got 15 lines for He")
        he.save("He", "Helium", href, 1.0, UtilsTest.tempdir)

    def test_hydrogen(self):
        href = "DOI: 10.1126/science.239.4844.1131"
        h = Cell.hexagonal(2.6590, 4.3340)
        self.assertAlmostEqual(h.volume, 26.537, msg="Volume for H cell is correct")
        self.assert_(len(h.d_spacing(1)) == 14, msg="got 14 lines for H")
        h.save("H", "Hydrogen", href, 1.0, UtilsTest.tempdir)


def suite():
    testsuite = unittest.TestSuite()
    testsuite.addTest(TestCalibrant("test_factory"))
    testsuite.addTest(TestCalibrant("test_2th"))
    testsuite.addTest(TestCalibrant("test_fake"))
    testsuite.addTest(TestCell("test_class"))
    testsuite.addTest(TestCell("test_classmethods"))
    testsuite.addTest(TestCell("test_dspacing"))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
