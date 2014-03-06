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
"test suite for Azimuthal integrator class"

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
import tempfile
from utilstest import UtilsTest, Rwp, getLogger
logger = getLogger(__file__)
pyFAI = sys.modules["pyFAI"]
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
if logger.getEffectiveLevel() <= logging.INFO:
    import pylab

class test_azim_halfFrelon(unittest.TestCase):
    """basic test"""
    fit2dFile = '1460/fit2d.dat'
    halfFrelon = "1464/LaB6_0020.edf"
    splineFile = "1461/halfccd.spline"
    poniFile = "1463/LaB6.poni"
    ai = None
    fit2d = None
    tmp_dir = os.environ.get("PYFAI_TEMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))
    tmpfiles = {"cython":os.path.join(tmp_dir, "cython.dat"),
               "cythonSP":os.path.join(tmp_dir, "cythonSP.dat"),
               "numpy": os.path.join(tmp_dir, "numpy.dat")}

    def setUp(self):
        """Download files"""
        self.fit2dFile = UtilsTest.getimage(self.__class__.fit2dFile)
        self.halfFrelon = UtilsTest.getimage(self.__class__.halfFrelon)
        self.splineFile = UtilsTest.getimage(self.__class__.splineFile)
        poniFile = UtilsTest.getimage(self.__class__.poniFile)

        with open(poniFile) as f:
            data = []
            for line in f:
                if line.startswith("SplineFile:"):
                    data.append("SplineFile: " + self.splineFile)
                else:
                    data.append(line.strip())
        self.poniFile = os.path.join(self.tmp_dir, os.path.basename(poniFile))
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        with open(self.poniFile, "w") as f:
            f.write(os.linesep.join(data))
        self.fit2d = numpy.loadtxt(self.fit2dFile)
        self.ai = AzimuthalIntegrator()
        self.ai.load(self.poniFile)
        self.data = fabio.open(self.halfFrelon).data
        for tmpfile in self.tmpfiles.values():
            if os.path.isfile(tmpfile):
                os.unlink(tmpfile)

    def tearDown(self):
        """Remove temporary files"""
        unittest.TestCase.tearDown(self)
        for tmpfile in self.tmpfiles.values():
            if os.path.isfile(tmpfile):
                os.unlink(tmpfile)
        if os.path.isfile(self.poniFile):
            os.unlink(self.poniFile)

    def test_numpy_vs_fit2d(self):
        """
        Compare numpy histogram with results of fit2d
        """
#        logger.info(self.ai.__repr__())
        tth, I = self.ai.xrpd_numpy(self.data,
                                     len(self.fit2d), self.tmpfiles["numpy"], correctSolidAngle=False)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp numpy/fit2d = %.3f" % rwp)
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
            raw_input("Press enter to quit")
        assert rwp < 11

    def test_cython_vs_fit2d(self):
        """
        Compare cython histogram with results of fit2d
        """
#        logger.info(self.ai.__repr__())
        tth, I = self.ai.xrpd_cython(self.data,
                                     len(self.fit2d), "tmp/cython.dat", correctSolidAngle=False, pixelSize=None)
#        logger.info(tth)
#        logger.info(I)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp cython/fit2d = %.3f" % rwp)
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
            raw_input("Press enter to quit")
        assert rwp < 11

    def test_cythonSP_vs_fit2d(self):
        """
        Compare cython splitPixel with results of fit2d
        """
        logger.info(self.ai.__repr__())
        pos = self.ai.cornerArray(self.data.shape)
        t0 = time.time()
        logger.info("in test_cythonSP_vs_fit2d Before SP")

        tth, I = self.ai.xrpd_splitPixel(self.data,
                                     len(self.fit2d), self.tmpfiles["cythonSP"], correctSolidAngle=False)
        logger.info("in test_cythonSP_vs_fit2d Before")
        t1 = time.time() - t0
#        logger.info(tth)
#        logger.info(I)
        rwp = Rwp((tth, I), self.fit2d.T)
        logger.info("Rwp cythonSP(t=%.3fs)/fit2d = %.3f" % (t1, rwp))
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
            raw_input("Press enter to quit")
        assert rwp < 11


    def test_cython_vs_numpy(self):
        """
        Compare cython histogram with numpy histogram
        """
#        logger.info(self.ai.__repr__())
        data = self.data
        tth_np, I_np = self.ai.xrpd_numpy(data,
                                     len(self.fit2d), correctSolidAngle=False)
        tth_cy, I_cy = self.ai.xrpd_cython(data,
                                     len(self.fit2d), correctSolidAngle=False)
        logger.info("before xrpd_splitPixel")
        tth_sp, I_sp = self.ai.xrpd_splitPixel(data,
                                     len(self.fit2d), correctSolidAngle=False)
        logger.info("After xrpd_splitPixel")
        rwp = Rwp((tth_cy, I_cy), (tth_np, I_np))
        logger.info("Rwp = %.3f" % rwp)
        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('Numpy Histogram vs Cython: Rwp=%.3f' % rwp)
            sp = fig.add_subplot(111)
            sp.plot(self.fit2d.T[0], self.fit2d.T[1], "-y", label='fit2d')
            sp.plot(tth_np, I_np, "-b", label='numpy')
            sp.plot(tth_cy, I_cy , "-r", label="cython")
            sp.plot(tth_sp, I_sp , "-g", label="SplitPixel")
            handles, labels = sp.get_legend_handles_labels()
            fig.legend(handles, labels)
            fig.show()
            raw_input("Press enter to quit")

        assert rwp < 3

class test_flatimage(unittest.TestCase):
    """test the caking of a flat image"""
    epsilon = 1e-4
    def test_splitPixel(self):
        data = numpy.ones((2000, 2000), dtype="float64")
        ai = AzimuthalIntegrator(0.1, 1e-2, 1e-2, pixel1=1e-5, pixel2=1e-5)
        I = ai.xrpd2_splitPixel(data, 2048, 2048, correctSolidAngle=False, dummy= -1.0)[0]
#        I = ai.xrpd2(data, 2048, 2048, correctSolidAngle=False, dummy= -1.0)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('cacking of a flat image: SplitPixel')
            sp = fig.add_subplot(111)
            sp.imshow(I, interpolation="nearest")
            fig.show()
            raw_input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon

    def test_splitBBox(self):
        data = numpy.ones((2000, 2000), dtype="float64")
        ai = AzimuthalIntegrator(0.1, 1e-2, 1e-2, pixel1=1e-5, pixel2=1e-5)
        I = ai.xrpd2_splitBBox(data, 2048, 2048, correctSolidAngle=False, dummy= -1.0)[0]
#        I = ai.xrpd2(data, 2048, 2048, correctSolidAngle=False, dummy= -1.0)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logging.info("Plotting results")
            fig = pylab.figure()
            fig.suptitle('cacking of a flat image: SplitBBox')
            sp = fig.add_subplot(111)
            sp.imshow(I, interpolation="nearest")
            fig.show()
            raw_input("Press enter to quit")
        I[I == -1.0] = 1.0
        assert abs(I.min() - 1.0) < self.epsilon
        assert abs(I.max() - 1.0) < self.epsilon

class test_saxs(unittest.TestCase):
    saxsPilatus = "1492/bsa_013_01.edf"
    maskFile = "1491/Pcon_01Apr_msk.edf"
    maskRef = "1490/bioSaxsMaskOnly.edf"
    maskDummy = "1488/bioSaxsMaskDummy.edf"
    poniFile = "1489/bioSaxs.poni"
    ai = None
    tmp_dir = os.environ.get("PYFAI_TEMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))

    def setUp(self):
        self.edfPilatus = UtilsTest.getimage(self.__class__.saxsPilatus)
        self.maskFile = UtilsTest.getimage(self.__class__.maskFile)
        self.poniFile = UtilsTest.getimage(self.__class__.poniFile)
        self.maskRef = UtilsTest.getimage(self.__class__.maskRef)
        self.maskDummy = UtilsTest.getimage(self.__class__.maskDummy)
        self.ai = AzimuthalIntegrator()
        self.ai.load(self.poniFile)
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def test_mask(self):
        """test the generation of mask"""
        print self.edfPilatus
        data = fabio.open(self.edfPilatus).data
        mask = fabio.open(self.maskFile).data
        assert abs(self.ai.makeMask(data, mask=mask).astype(int) - fabio.open(self.maskRef).data).max() == 0
        assert abs(self.ai.makeMask(data, mask=mask, dummy= -2, delta_dummy=1.1).astype(int) - fabio.open(self.maskDummy).data).max() == 0

class test_setter(unittest.TestCase):
    tmp_dir = os.environ.get("PYFAI_TEMPDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))

    def setUp(self):
        self.ai = AzimuthalIntegrator()
        shape = (10, 15)
        self.rnd1 = numpy.random.random(shape).astype(numpy.float32)
        self.rnd2 = numpy.random.random(shape).astype(numpy.float32)
        if not os.path.isdir(self.tmp_dir):
            os.mkdir(self.tmp_dir)

        fd, self.edf1 = tempfile.mkstemp(".edf", "testAI1", self.tmp_dir)
        os.close(fd)
        fd, self.edf2 = tempfile.mkstemp(".edf", "testAI2", self.tmp_dir)
        os.close(fd)
        fabio.edfimage.edfimage(data=self.rnd1).write(self.edf1)
        fabio.edfimage.edfimage(data=self.rnd2).write(self.edf2)
    def tearDown(self):
        if os.path.exists(self.edf1):
            os.unlink(self.edf1)
        if os.path.exists(self.edf2):
            os.unlink(self.edf2)
    def test_flat(self):
        self.ai.set_flatfiles((self.edf1,self.edf2), method="mean")
        self.assert_(self.ai.flatfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "flatfiles string is OK")
        self.assert_(abs(self.ai.flatfield-0.5*(self.rnd1+self.rnd2)).max() == 0, "Flat array is OK")
    def test_dark(self):
        self.ai.set_darkfiles((self.edf1, self.edf2), method="mean")
        self.assert_(self.ai.darkfiles == "%s(%s,%s)" % ("mean", self.edf1, self.edf2), "darkfiles string is OK")
        self.assert_(abs(self.ai.darkcurrent-0.5*(self.rnd1+self.rnd2)).max() == 0, "Dark array is OK")

def test_suite_all_AzimuthalIntegration():
    testSuite = unittest.TestSuite()
    testSuite.addTest(test_azim_halfFrelon("test_cython_vs_fit2d"))
    testSuite.addTest(test_azim_halfFrelon("test_numpy_vs_fit2d"))
    testSuite.addTest(test_azim_halfFrelon("test_cythonSP_vs_fit2d"))
    testSuite.addTest(test_azim_halfFrelon("test_cython_vs_numpy"))
    testSuite.addTest(test_flatimage("test_splitPixel"))
    testSuite.addTest(test_flatimage("test_splitBBox"))
    testSuite.addTest(test_setter("test_flat"))
    testSuite.addTest(test_setter("test_dark"))
# This test is known to be broken ...
#    testSuite.addTest(test_saxs("test_mask"))

    return testSuite

if __name__ == '__main__':

    mysuite = test_suite_all_AzimuthalIntegration()
    runner = unittest.TextTestRunner()
    runner.run(mysuite)
