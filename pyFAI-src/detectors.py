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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/04/2012"
__status__ = "beta"

import os, logging, threading
logger = logging.getLogger("pyFAI.detectors")
import numpy
from spline import Spline

class Detector(object):
    "Generic class representing a 2D detector"
    def __init__(self, pixel1=None, pixel2=None, splineFile=None):
        self.name = self.__class__.__name__
        self.pixel1 = pixel1
        self.pixel2 = pixel2
        self.max_shape = (None, None)
        self.binning = (1, 1)
        self.mask = None
        self._splineFile = None
        self.spline = None
        self._splineCache = {} #key=(dx,xpoints,ypoints) value: ndarray
        self._sem = threading.Semaphore()
        if splineFile:
            self.set_splineFile(splineFile)

    def __repr__(self):
        return "Detector %s\t Spline= %s\t PixelSize= %.3e, %.3e m" % (self.name, self.splineFile, self.pixel1, self.pixel2)

    def get_splineFile(self):
        return self._splineFile
    def set_splineFile(self, splineFile):
        if splineFile is not None:
            self._splineFile = os.path.abspath(splineFile)
            self.spline = Spline(self._splineFile)
            #NOTA : X is axis 1 and Y is Axis 0 
            self.pixel2, self.pixel1 = self.spline.getPixelSize()
            self._splineCache = {}
        else:
            self._splineFile = None
            self.spline = None
    splineFile = property(get_splineFile, set_splineFile)

    def getPyFAI(self):
        return {"pixel1":self.pixel1,
                "pixel2":self.pixel2,
                "splineFile":self._splineFile}

    def getFit2D(self):
        return {"pixelX":self.pixel2 * 1e6,
                "pixelY":self.pixel1 * 1e6,
                "splineFile":self._splineFile}

    def setPyFAI(self, **kwarg):
        for kw in kwarg:
            if kw in ["pixel1", "pixel2"]:
                setattr(self, kw, kwarg[kw])
            elif kw == "splineFile":
                self.set_splineFile(kwarg[kw])

    def setFit2D(self):
        for kw, val in kwarg.items():
            if kw == "pixelX":
                self.pixel2 = val * 1e-6
            elif kw == "pixelY":
                self.pixel1 = val * 1e-6
            elif kw == "splineFile":
                self.set_splineFile(kwarg[kw])

    def calc_catesian_positions(self, d1=None, d2=None):
        """
        Calculate the position of each pixel center in cartesian coordinate 
        and in meter of a couple of coordinates. 
        The half pixel offset is taken into account here !!!
        
        @param d1: ndarray of dimension 1 or 2 containing the Y pixel positions
        @param d2: ndarray of dimension 1or 2 containing the X pixel positions

        @return: 2-arrays of same shape as d1 & d2 with the position in meter

        d1 and d2 must have the same shape, returned array will have the same shape.
        """
        if (d1 is None):
            d1 = numpy.arange(self.max_shape[0])
        if (d2 is None):
            d2 = numpy.arange(self.max_shape[1])

        if self.spline is None:
            dX = 0.
            dY = 0.
        else:
            if d2.ndim == 1:
                keyX = ("dX", tuple(d1), tuple(d2))
                keyY = ("dY", tuple(d1), tuple(d2))
                if keyX not in self._splineCache:
                    self._splineCache[keyX] = numpy.array([self.spline.splineFuncX(i2, i1) for i1, i2 in zip(d1 + 0.5, d2 + 0.5)], dtype="float64")
                if keyY not in self._splineCache:
                    self._splineCache[keyY] = numpy.array([self.spline.splineFuncY(i2, i1) for i1, i2 in zip(d1 + 0.5, d2 + 0.5)], dtype="float64")
                dX = self._splineCache[keyX]
                dY = self._splineCache[keyY]
            else:
                dX = self.spline.splineFuncX(d2 + 0.5, d1 + 0.5)
                dY = self.spline.splineFuncY(d2 + 0.5, d1 + 0.5)
        p1 = (self.pixel1 * (dY + 0.5 + d1))
        p2 = (self.pixel2 * (dX + 0.5 + d2))
        return p1, p2

    def get_mask(self):
        """
        Should return a generic mask for the detector
        """
        if self.mask is None:
            raise NotImplementedError("detector.getmask is not implemented for detector %s" % self.__class__.__name)
        else:
            return self.mask

class Pilatus(Detector):
    "Pilatus detector: generic description"
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Detector.__init__(self, pixel1, pixel2)
    def get_mask(self):
        """
        Returns a generic mask for Pilatus detecors...
        """
        if self.mask is None:
            with self._sem:
                if self.mask is None:
                    if self.max_shape[0] is None or\
                        self.max_shape[1] is None:
                        raise NotImplementedError("Generic Pilatus detector does not know the max size ...")
                    self.mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
                    #workinng in dim0 = Y
                    for i in range(self.MODULE_SIZE[0], self.max_shape[0], self.MODULE_SIZE[0] + self.MODULE_GAP[0]):
                        self.mask[i: i + self.MODULE_GAP[0], :] = 1
                    #workinng in dim1 = X
                    for i in range(self.MODULE_SIZE[1], self.max_shape[1], self.MODULE_SIZE[1] + self.MODULE_GAP[1]):
                        self.mask[:, i: i + self.MODULE_GAP[1]] = 1
        return self.mask

class Pilatus1M(Pilatus):
    "Pilatus 1M detector"
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (1043, 981)
class Pilatus2M(Pilatus):
    "Pilatus 2M detector"
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (1475, 1679)
class Pilatus6M(Pilatus):
    "Pilatus 6M detector"
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (2527, 2463)
class Fairchild(Detector):
    "Fairchild Condor 486:90 detector"
    def __init__(self, pixel1=15e-6, pixel2=15e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.name = "Fairchild Condor 486:90"
        self.max_shape = (4096, 4096)
class FReLoN(Detector):
    "FReLoN detector (spline mandatory to correct for geometric distortion)"
    def __init__(self, splineFile):
        Detector.__init__(self, splineFile)
        self.max_shape = (self.spline.ymax - self.spline.ymin, self.spline.xmax - self.splinex.xmin)
