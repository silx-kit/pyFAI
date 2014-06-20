# !/usr/bin/env python
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "21/02/2013"
__status__ = "development"

import unittest
from utilstest import UtilsTest, getLogger
import logging, threading
import types, os, sys
import numpy
logger = logging.getLogger("pyFAI.distortion")
logging.basicConfig(level=logging.INFO)
from math import ceil, floor
pyFAI = sys.modules["pyFAI"]
from pyFAI import detectors, ocl_azim_lut, _distortion, _distortionCSR, distortion
from pyFAI.utils import timeit
import fabio

#import pyFAI._distortion
#import pyFAI._distortionCSR


def test():

#    workin on 256x256
#    x, y = numpy.ogrid[:256, :256]
#    grid = numpy.logical_or(x % 10 == 0, y % 10 == 0) + numpy.ones((256, 256), numpy.float32)
#    det = detectors.FReLoN("frelon_8_8.spline")

#    # working with halfccd spline
    x, y = numpy.ogrid[:1024, :2048]
    grid = numpy.logical_or(x % 100 == 0, y % 100 == 0) + numpy.ones((1024, 2048), numpy.float32)
    
    splineFilePath = "1461/halfccd.spline"
    splineFile = UtilsTest.getimage(splineFilePath)  
    det = detectors.FReLoN(splineFile)
    # working with halfccd spline
#    x, y = numpy.ogrid[:2048, :2048]
#    grid = numpy.logical_or(x % 100 == 0, y % 100 == 0).astype(numpy.float32) + numpy.ones((2048, 2048), numpy.float32)
#    det = detectors.FReLoN("frelon.spline")


    print det, det.max_shape
    disLUT = _distortion.Distortion(det)
    print disLUT
    lut = disLUT.calc_LUT_size()
    print disLUT.lut_size
    print lut.mean()

    disLUT.calc_LUT()
    outLUT = disLUT.correct(grid)
    fabio.edfimage.edfimage(data=outLUT.astype("float32")).write("test_correct_LUT.edf")

    print("*"*50)


    print det, det.max_shape
    disCSR = _distortionCSR.Distortion(det,foo=64)
    print disCSR
    lut = disCSR.calc_LUT_size()
    print disCSR.lut_size
    print lut.mean()

    disCSR.calc_LUT()
    outCSR = disCSR.correct(grid)
    fabio.edfimage.edfimage(data=outCSR.astype("float32")).write("test_correct_CSR.edf")

    print("*"*50)

    disCSR.setDevice()
    outCSRocl = disCSR.correct(grid)
    fabio.edfimage.edfimage(data=outCSRocl.astype("float32")).write("test_correct_CSR.edf")

    print("*"*50)
    
    print det, det.max_shape
    disLUTpy = distortion.Distortion(det)
    print disLUTpy
    lut = disLUTpy.calc_LUT_size()
    print disLUTpy.lut_size
    print lut.mean()

    disLUTpy.calc_LUT()
    outLUTpy = disLUTpy.correct(grid)
    fabio.edfimage.edfimage(data=outLUTpy.astype("float32")).write("test_correct_LUT.edf")

    print("*"*50)

#    x, y = numpy.ogrid[:2048, :2048]
#    grid = numpy.logical_or(x % 100 == 0, y % 100 == 0)
#    det = detectors.FReLoN("frelon.spline")
#    print det, det.max_shape
#    dis = Distortion(det)
#    print dis
#    lut = dis.calc_LUT_size()
#    print dis.lut_size
#    print "LUT mean & max", lut.mean(), lut.max()
#    dis.calc_LUT()
#    out = dis.correct(grid)
#    fabio.edfimage.edfimage(data=out.astype("float32")).write("test2048.edf")
    import matplotlib;matplotlib.use('GTK');import pylab
    #pylab.imshow(outLUT)
    #pylab.show()
    #pylab.imshow(outCSR)  # , interpolation="nearest")
# , interpolation="nearest")
#    pylab.show()
    pylab.imshow(outCSRocl)
    pylab.show()
    #pylab.imshow(outLUTpy)
    #pylab.show()
    
    assert numpy.allclose(outLUT,outCSRocl)

if __name__ == "__main__":
    det = dis = lut = None
    test()
