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
from __future__ import absolute_import, division, print_function

"""
Test suites for calibrants 
"""


import unittest
import sys

if __name__ == '__main__':
    import pkgutil, os
    __path__ = pkgutil.extend_path([os.path.dirname(__file__)], "pyFAI.test")
from .utilstest import getLogger UtilsTest

logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]

from pyFAI.calibrant import Calibrant, ALL_CALIBRANTS, Cell
from pyFAI.detectors import ALL_DETECTORS


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
        with_plot = False
        if with_plot:
            import matplotlib
            matplotlib.use('Agg')

            import matplotlib.pyplot as plt

            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib import rcParams

            pp = PdfPages('fake.pdf')
            rcParams['font.size'] = 6
            plt.clf()

        detectors = set(ALL_DETECTORS.values())
        for idx, detector in enumerate(detectors):
            det = detector()
            # Skip generic detectors
            if "MAX_SHAPE" not in dir(det):
                continue
            # skip the big detectors for now
            if max(det.MAX_SHAPE) > 2000:
                continue
            ai = pyFAI.AzimuthalIntegrator(dist=0.01, poni1=0, poni2=0,
                                           detector=det)
            calibrant = ALL_CALIBRANTS["LaB6"]
            calibrant.set_wavelength(1e-10)
            img = calibrant.fake_calibration_image(ai)

            if with_plot:
                plt.clf
                plt.subplot(3, 4, idx % 12)
                plt.title(det.name)
                plt.imshow(img, interpolation='nearest')

                if idx != 0 and idx % 12 == 0:
                    pp.savefig()
                    plt.clf()
                print(det.name, img.min(), img.max())
            self.assert_(img.shape == det.shape, "Image (%s) has the right size" % (det.name,))
            self.assert_(img.sum() > 0, "Image (%s) contains some data" % (det.name,))
            sys.stderr.write(".")

        if with_plot:
            pp.savefig()
            pp.close()


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


def test_suite_all_calibrant():
    testSuite = unittest.TestSuite()
    testSuite.addTest(TestCalibrant("test_factory"))
    testSuite.addTest(TestCalibrant("test_2th"))
    testSuite.addTest(TestCalibrant("test_fake"))
    testSuite.addTest(TestCell("test_class"))
    testSuite.addTest(TestCell("test_classmethods"))
    testSuite.addTest(TestCell("test_dspacing"))
    return testSuite


if __name__ == '__main__':
    mysuite = test_suite_all_calibrant()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
