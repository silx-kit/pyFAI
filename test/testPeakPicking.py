#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
"test suite for peak picking class"

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20140106"


import unittest
import os
import numpy
import logging, time
import sys
import fabio
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
import pyFAI.peakPicker
import pyFAI.geometryRefinement
from pyFAI.peakPicker import PeakPicker
from pyFAI.geometryRefinement import GeometryRefinement
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


class test_peak_picking(unittest.TestCase):
    """basic test"""
    calibFile = "1788/moke.tif"
#    gr = GeometryRefinement()


    ctrlPt = {0:(300, 230),
              1:(300, 212),
              2:(300, 195),
              3:(300, 177),
              4:(300, 159),
              5:(300, 140),
              6:(300, 123),
              7:(300, 105),
              8:(300, 87)}
    tth = numpy.radians(numpy.arange(4, 13))
    wavelength = 1e-10
    ds = wavelength * 5e9 / numpy.sin(tth / 2)
    maxiter = 100
    tmp_dir = os.environ.get("PYFAI_TEMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))
    logfile = os.path.join(tmp_dir, "testpeakPicking.log")
    nptfile = os.path.join(tmp_dir, "testpeakPicking.npt")
    def setUp(self):
        """Download files"""
        self.img = UtilsTest.getimage(self.__class__.calibFile)
        self.pp = PeakPicker(self.img, dSpacing=self.ds, wavelength=self.wavelength)
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        if os.path.isfile(self.logfile):
            os.unlink(self.logfile)
        if os.path.isfile(self.nptfile):
            os.unlink(self.nptfile)
    def tearDown(self):
        """Remove temporary files"""
        unittest.TestCase.tearDown(self)
        if os.path.isfile(self.logfile):
            os.unlink(self.logfile)
        if os.path.isfile(self.nptfile):
            os.unlink(self.nptfile)

    def test_peakPicking(self):
        """first test peak-picking then checks the geometry found is OK"""
        for i in self.ctrlPt:
            pts = self.pp.massif.find_peaks(self.ctrlPt[i], stdout=open(self.logfile, "a"))
            logger.info("point %s at ring #%i (tth=%.1f deg) generated %i points", self.ctrlPt[i], i, self.tth[i], len(pts))
            if len(pts) > 0:
                self.pp.points.append(pts, angle=self.tth[i], ring=i)
            else:
                logger.error("point %s caused error (%s) ", i, self.ctrlPt[i])

        self.pp.points.save(self.nptfile)
        lstPeak = self.pp.points.getListRing()
#        print self.pp.points
#        print lstPeak
        logger.info("After peak-picking, we have %s points generated from %s groups ", len(lstPeak), len(self.ctrlPt))
        gr = GeometryRefinement(lstPeak, dist=0.01, pixel1=1e-4, pixel2=1e-4, wavelength=self.wavelength, dSpacing=self.ds)
        gr.guess_poni()
        logger.info(gr.__repr__())
        last = sys.maxint
        for i in range(self.maxiter):
            delta2 = gr.refine2()
            logger.info(gr.__repr__())
            if delta2 == last:
                logger.info("refinement finished after %s iteration" % i)
                break
            last = delta2
        self.assertEquals(last < 1e-4, True, "residual error is less than 1e-4, got %s" % last)
        self.assertAlmostEquals(gr.dist, 0.1, 2, "distance is OK, got %s, expected 0.1" % gr.dist)
        self.assertAlmostEquals(gr.poni1, 3e-2, 2, "PONI1 is OK, got %s, expected 3e-2" % gr.poni1)
        self.assertAlmostEquals(gr.poni2, 3e-2, 2, "PONI2 is OK, got %s, expected 3e-2" % gr.poni2)
        self.assertAlmostEquals(gr.rot1, 0, 2, "rot1 is OK, got %s, expected 0" % gr.rot1)
        self.assertAlmostEquals(gr.rot2, 0, 2, "rot2 is OK, got %s, expected 0" % gr.rot2)
        self.assertAlmostEquals(gr.rot3, 0, 2, "rot3 is OK, got %s, expected 0" % gr.rot3)

#        print self.pp.points
class test_Massif(unittest.TestCase):
    """test for ring extraction algorithm with image which needs binning (non regression test)"""
    calibFile = "1788/moke.tif"

def test_suite_all_PeakPicking():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_peak_picking("test_peakPicking"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_PeakPicking()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
