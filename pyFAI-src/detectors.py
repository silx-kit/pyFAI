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
__date__ = "29/11/2012"
__status__ = "stable"

import os, logging, threading, sys
logger = logging.getLogger("pyFAI.detectors")
import numpy
from spline import Spline
try:
    from fastcrc import crc32
except:
    from zlib import crc32



class Detector(object):
    "Generic class representing a 2D detector"
    def __init__(self, pixel1=None, pixel2=None, splineFile=None):
        self.name = self.__class__.__name__
        self.pixel1 = pixel1
        self.pixel2 = pixel2
        self.max_shape = (None, None)
        self._binning = (1, 1)
        self._mask = False
        self._mask_crc = None
        self._maskfile = None
        self._splineFile = None
        self.spline = None
        self._splineCache = {}  # key=(dx,xpoints,ypoints) value: ndarray
        self._sem = threading.Semaphore()
        if splineFile:
            self.set_splineFile(splineFile)

    def __repr__(self):
        if (self.pixel1 is None) and (self.pixel2 is None):
            return "Undefined detector"
        return "Detector %s\t Spline= %s\t PixelSize= %.3e, %.3e m" % (self.name, self.splineFile, self.pixel1, self.pixel2)

    def get_splineFile(self):
        return self._splineFile
    def set_splineFile(self, splineFile):
        if splineFile is not None:
            self._splineFile = os.path.abspath(splineFile)
            self.spline = Spline(self._splineFile)
            # NOTA : X is axis 1 and Y is Axis 0
            self.pixel2, self.pixel1 = self.spline.getPixelSize()
            self._splineCache = {}
        else:
            self._splineFile = None
            self.spline = None
    splineFile = property(get_splineFile, set_splineFile)
    def get_binning(self):
        return self._binning
    def set_binning(self, bin_size=(1, 1)):
        if "__len__" in bin_size and len(bin_size) >= 2:
            bin_size = (float(bin_size[0]), float(bin_size[1]))
        else:
            b = float(bin_size)
            bin_size = (b, b)
        if bin_size != self._binning:
            ratioX = bin_size[1] / self._binning[1]
            ratioY = bin_size[0] / self._binning[0]
            if self.spline is not None:
                self.spline.bin((ratioX, ratioY))
            else:
                self.pixel1 *= ratioY
                self.pixel2 *= ratioX
            self._binning = bin_size
    binning = property(get_binning, set_binning)


    def getPyFAI(self):
        return {"detector":self.name,
                "pixel1":self.pixel1,
                "pixel2":self.pixel2,
                "splineFile":self._splineFile}

    def getFit2D(self):
        return {"pixelX":self.pixel2 * 1e6,
                "pixelY":self.pixel1 * 1e6,
                "splineFile":self._splineFile}

    def setPyFAI(self, **kwarg):
        if "detector" in kwarg:
            self = detector_factory(kwarg["detector"])
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

    def calc_cartesian_positions(self, d1=None, d2=None):
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

    def calc_mask(self):
        """
        Detectors with gaps should overwrite this method with something actually calculating the mask!
        """
#        logger.debug("Detector.calc_mask is not implemented for generic detectors")
        return None

    def get_mask(self):
        if self._mask is False:
            with self._sem:
                if self._mask is False:
                    self._mask = self.calc_mask() #gets None in worse cases
                    if self._mask is not None:
                        self._mask_crc = crc32(self._mask)
        return self._mask
    def set_mask(self, mask):
        with self._sem:
            self._mask = mask
            self._mask_crc = crc32(mask)
    mask = property(get_mask, set_mask)
    def set_maskfile(self, maskfile):
        try:
            import fabio
        except:
            ImportError("Please install fabio to load images")
        with self._sem:
            self._mask = numpy.ascontiguousarray(fabio.open(maskfile).data, dtype=numpy.int8)
            self._mask_crc = crc32(self._mask)
            self._maskfile = maskfile
    def get_maskfile(self):
        return self._maskfile
    maskfile = property(get_maskfile, set_maskfile)


class Pilatus(Detector):
    "Pilatus detector: generic description"
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Detector.__init__(self, pixel1, pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % (self.name, self.pixel1, self.pixel2)

    def calc_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        if self.max_shape[0] is None or\
            self.max_shape[1] is None:
            raise NotImplementedError("Generic Pilatus detector does not know the max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.MODULE_SIZE[0], self.max_shape[0], self.MODULE_SIZE[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.MODULE_SIZE[1], self.max_shape[1], self.MODULE_SIZE[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

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

class Xpad_flat(Detector):
    "Xpad detector: generic description for image with "
    MODULE_SIZE = (120, 80)
    MODULE_GAP = (3 + 3.57 * 1000 / 130, 3)#in pixels
    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.max_shape = (960, 560)
    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % (self.name, self.pixel1, self.pixel2)

    def calc_mask(self):
        """
        Returns a generic mask for Xpad detectors...
        discards the first line and raw form all modules: those are 2.5x bigger and often mis - behaving
        """
        with self._sem:
            if self.max_shape[0] is None or\
                self.max_shape[1] is None:
                raise NotImplementedError("Generic Xpad detector does not know the max size ...")
            mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
            # workinng in dim0 = Y                   
            for i in range(0, self.max_shape[0], self.MODULE_SIZE[0]):
                mask[i, :] = 1
                mask[i + self.MODULE_SIZE[0] - 1, :] = 1
            # workinng in dim1 = X
            for i in range(0, self.max_shape[1], self.MODULE_SIZE[1]):
                mask[:, i ] = 1
                mask[:, i + self.MODULE_SIZE[1] - 1] = 1
        return mask

    def calc_cartesian_positions(self, d1=None, d2=None):
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
            c1 = numpy.arange(self.max_shape[0])
            for i in range(self.max_shape[0] // self.MODULE_SIZE[0]):
                c1[i * self.MODULE_SIZE[0], (i + 1) * self.MODULE_SIZE[0]] += i * self.MODULE_GAP[0]
        else:
            c1 = d1 + (d1.astype(numpy.int64) // self.MODULE_SIZE[0]) * self.MODULE_GAP[0]
        if (d2 is None):
            c2 = numpy.arange(self.max_shape[1])
            for i in range(self.max_shape[1] // self.MODULE_SIZE[1]):
                c2[i * self.MODULE_SIZE[1], (i + 1) * self.MODULE_SIZE[1]] += i * self.MODULE_GAP[1]
        else:
            c2 = d2 + (d2.astype(numpy.int64) // self.MODULE_SIZE[1]) * self.MODULE_GAP[1]

        p1 = self.pixel1 * (0.5 + c1)
        p2 = self.pixel2 * (0.5 + c2)
        return p1, p2

ALL_DETECTORS = {# I prefer not allowing the instantiation of generic detectors
                 #"detector": Detector,
                 #"pilatus":Pilatus,
                 "pilatus1m":Pilatus1M,
                 "pilatus2m":Pilatus2M,
                 "pilatus6m":Pilatus6M,
                 "condor":Fairchild,
                 "fairchild":Fairchild,
                 "frelon":FReLoN,
                 "xpad":Xpad_flat,
                 "xpad_flat":Xpad_flat,
                 }


def detector_factory(name):
    """
    A kind of factory...
    @param name: name of a detector
    @return: an instance of the right detector
    """
    name = name.lower()
    if name in ALL_DETECTORS:
        return ALL_DETECTORS[name]()
    else:
        raise RuntimeError("Detector %s is unknown !" % (name))
