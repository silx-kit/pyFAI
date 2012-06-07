#!/usr/bin/env python
# -*- coding: utf8 -*-
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
__date__ = "04/06/2012"


import unittest
import os
import numpy
import logging, time
import sys
import fabio
force_build = False
for opts in sys.argv[1:]:
    if opts in ["-d", "--debug"]:
        logging.basicConfig(level=logging.DEBUG)
        sys.argv.pop(sys.argv.index(opts))
    elif opts in ["-i", "--info"]:
        logging.basicConfig(level=logging.INFO)
        sys.argv.pop(sys.argv.index(opts))
    elif opts in ["-f", "--force"]:
        force_build = True
        sys.argv.pop(sys.argv.index(opts))
logger = logging.getLogger("testPeakPicking")

try:
    logger.debug("tests loaded from file: %s" % __file__)
except:
    __file__ = os.getcwd()
    logger.debug("tests loaded from file: %s" % __file__)

from utilstest import UtilsTest, Rwp
if force_build:
    UtilsTest.forceBuild()
pyFAI = sys.modules["pyFAI"]
from pyFAI.peakPicker import PeakPicker
from pyFAI.geometryRefinement import GeometryRefinement

if logger.getEffectiveLevel() <= logging.INFO:
    import pylab


class test_peak_picking(unittest.TestCase):
    """basic test"""
    calibFile = "1788/moke.tif"
#    gr = GeometryRefinement()
    ctrlPt = {4:(300, 230),
              5:(300, 212),
              6:(300, 195),
              7:(300, 177),
              8:(300, 159),
              9:(300, 140),
              10:(300, 123),
              11:(300, 105),
              12:(300, 87)}
    maxiter = 100
    def setUp(self):
        """Download files"""
        self.img = UtilsTest.getimage(self.__class__.calibFile)
        self.pp = PeakPicker(self.img)
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.tmpdir = os.path.join(dirname, "tmp")
        if not os.path.isdir(self.tmpdir):
            os.mkdir(self.tmpdir)

    def test_peakPicking(self):
        """first test peak-picking then checks the geometry found is OK"""
        logfile = os.path.join(self.tmpdir, "testpeakPicking.log")

        for i in self.ctrlPt:
            pts = self.pp.massif.find_peaks(self.ctrlPt[i], stdout=open(logfile, "a"))
            logger.info("point %s at %i deg generated %i points", self.ctrlPt[i], i, len(pts))
            if len(pts) > 0:
                self.pp.points.append_2theta_deg(pts, i)
            else:
                logger.error("point %s caused error (%s) ", i, self.ctrlPt[i])

        self.pp.points.save(os.path.join(self.tmpdir, "testpeakPicking.npt"))
        lstPeak = self.pp.points.getList()
        logger.info("After peak-picking, we have %s points generated from %s points ", len(lstPeak), len(self.ctrlPt))
        gr = GeometryRefinement(lstPeak, dist=0.05, pixel1=1e-4, pixel2=1e-4)
        logger.info(gr.__repr__())
        last = sys.maxint
        for i in range(self.maxiter):
            delta2 = gr.refine2()
            logger.info(gr.__repr__())
            if delta2 == last:
                logger.info("refinement finished after %s iteration" % i)
                break
            last = delta2
        self.assertEquals(last < 1e-4, True, "residual error is less than 1e-4")
        self.assertAlmostEquals(gr.dist, 0.1, 2, "distance is OK")
        self.assertAlmostEquals(gr.poni1, 3e-2, 2, "PONI1 is OK")
        self.assertAlmostEquals(gr.poni2, 3e-2, 2, "PONI2 is OK")
        self.assertAlmostEquals(gr.rot1, 0, 2, "rot1 is OK")
        self.assertAlmostEquals(gr.rot2, 0, 2, "rot2 is OK")
        self.assertAlmostEquals(gr.rot3, 0, 2, "rot3 is OK")

#        print self.pp.points

def test_suite_all_PeakPicking():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_peak_picking("test_peakPicking"))
    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_PeakPicking()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
