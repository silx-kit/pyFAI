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

from __future__ import absolute_import, division, print_function

"""Test suite for non-gui peak picking class"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/03/2018"


import unittest
import os
import numpy
import sys
import logging
import shutil
from .utilstest import UtilsTest
logger = logging.getLogger(__name__)
from ..gui.peak_picker import PeakPicker
from ..calibrant import Calibrant
from ..geometryRefinement import GeometryRefinement


class TestPeakPicking(unittest.TestCase):
    """basic test"""

    def setUp(self):
        """Download files"""

        self.calibFile = "mock.tif"
        self.ctrlPt = {0: (300, 230),
                       1: (300, 212),
                       2: (300, 195),
                       3: (300, 177),
                       4: (300, 159),
                       5: (300, 140),
                       6: (300, 123),
                       7: (300, 105),
                       8: (300, 87)}
        self.tth = numpy.radians(numpy.arange(4, 13))
        self.wavelength = 1e-10
        self.ds = self.wavelength * 5e9 / numpy.sin(self.tth / 2)
        self.calibrant = Calibrant(dSpacing=self.ds)
        self.maxiter = 100
        self.tmp_dir = os.path.join(UtilsTest.tempdir, "peak_picking")
        self.logfile = os.path.join(self.tmp_dir, "testpeakPicking.log")
        self.nptfile = os.path.join(self.tmp_dir, "testpeakPicking.npt")

        # download files

        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.img = UtilsTest.getimage(self.calibFile)
        self.pp = PeakPicker(self.img, calibrant=self.calibrant, wavelength=self.wavelength)
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        if os.path.isfile(self.logfile):
            os.unlink(self.logfile)
        if os.path.isfile(self.nptfile):
            os.unlink(self.nptfile)

    def tearDown(self):
        """Remove temporary files"""
        shutil.rmtree(self.tmp_dir)
        self.calibFile = self.ctrlPt = self.tth = self.wavelength = self.ds = None
        self.calibrant = self.maxiter = self.tmp_dir = self.logfile = self.nptfile = None

    def test_peakPicking(self):
        """first test peak-picking then checks the geometry found is OK"""
        for i in self.ctrlPt:
            with open(self.logfile, "a") as log:
                pts = self.pp.massif.find_peaks(self.ctrlPt[i], stdout=log)
            logger.info("point %s at ring #%i (tth=%.1f deg) generated %i points", self.ctrlPt[i], i, self.tth[i], len(pts))
            if len(pts) > 0:
                self.pp.points.append(pts, ring=i)
            else:
                logger.error("point %s caused error (%s) ", i, self.ctrlPt[i])

        self.pp.points.save(self.nptfile)
        lstPeak = self.pp.points.getListRing()
        logger.info("After peak-picking, we have %s points generated from %s groups ", len(lstPeak), len(self.ctrlPt))
        gr = GeometryRefinement(lstPeak, dist=0.01, pixel1=1e-4, pixel2=1e-4, wavelength=self.wavelength, calibrant=self.calibrant)
        gr.guess_poni()
        logger.info(gr.__repr__())
        last = sys.maxint if sys.version_info[0] < 3 else sys.maxsize
        for i in range(self.maxiter):
            delta2 = gr.refine2()
            logger.info(gr.__repr__())
            if delta2 == last:
                logger.info("refinement finished after %s iteration", i)
                break
            last = delta2
        self.assertEquals(last < 1e-4, True, "residual error is less than 1e-4, got %s" % last)
        self.assertAlmostEqual(gr.dist, 0.1, 2, "distance is OK, got %s, expected 0.1" % gr.dist)
        self.assertAlmostEqual(gr.poni1, 3e-2, 2, "PONI1 is OK, got %s, expected 3e-2" % gr.poni1)
        self.assertAlmostEqual(gr.poni2, 3e-2, 2, "PONI2 is OK, got %s, expected 3e-2" % gr.poni2)
        self.assertAlmostEqual(gr.rot1, 0, 2, "rot1 is OK, got %s, expected 0" % gr.rot1)
        self.assertAlmostEqual(gr.rot2, 0, 2, "rot2 is OK, got %s, expected 0" % gr.rot2)
        self.assertAlmostEqual(gr.rot3, 0, 2, "rot3 is OK, got %s, expected 0" % gr.rot3)


class TestMassif(unittest.TestCase):
    """test for ring extraction algorithm with image which needs binning (non regression test)"""
    calibFile = "mock.tif"
    # TODO !!!


def suite():
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite = unittest.TestSuite()
    testsuite.addTest(loader(TestPeakPicking))
    testsuite.addTest(loader(TestMassif))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
