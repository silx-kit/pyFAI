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
__date__ = "29/08/2012"
__status__ = "stable"

import os
import logging
import threading
import numpy

logger = logging.getLogger("pyFAI.detectors")

from pyFAI import spline
from pyFAI.utils import lazy_property
try:
    from pyFAI.fastcrc import crc32
except ImportError:
    from zlib import crc32
try:
    import fabio
except ImportError:
    fabio = None


class Detector(object):
    """
    Generic class representing a 2D detector
    """
    force_pixel = False
    isDetector = True #used to recognize detector classes
    def __init__(self, pixel1=None, pixel2=None, splineFile=None):
        """
        @param pixel1: size of the pixel in meter along the slow dimension (often Y)
        @type pixel1: float
        @param pixel2: size of the pixel in meter along the fast dimension (often X)
        @type pixel2: float
        @param splineFile: path to file containing the geometric correction.
        @type splineFile: str
        """
        try:  
            self.name = self.__class__.name
        except:
            self.name = self.__class__.__name__
        self._pixel1 = None
        self._pixel2 = None
        if pixel1:
            self.pixel1 = pixel1
        if pixel2:
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
        if (self._pixel1 is None) or (self._pixel2 is None):
            return "Undefined detector"
        return "Detector %s\t Spline= %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self.splineFile, self._pixel1, self._pixel2)
    def set_config(self, config):
        """
        Sets the configuration of the detector. This implies:
        - Orientation: integers 
        - Binning
        - ROI
        
        The configuration is either a python dictionnary or a JSON string or a file containing this JSON configuration
        
        keys in that dictionnary are :
        "orientation": integers from 0 to 7
        "binning": integer or 2-tuple of integers. If only one integer is provided, 
        "offset": coordinate (in pixels) of the start of the detector 
        """

        raise NotImplementedError

    def set_config(self, config):
        """
        Sets the configuration of the detector. This implies:
        - Orientation: integers 
        - Binning
        - ROI
        
        The configuration is either a python dictionnary or a JSON string or a file containing this JSON configuration
        
        keys in that dictionnary are :
        "orientation": integers from 0 to 7
        "binning": integer or 2-tuple of integers. If only one integer is provided, 
        "offset": coordinate (in pixels) of the start of the detector 
        """

        raise NotImplementedError

    def get_splineFile(self):
        return self._splineFile

    def set_splineFile(self, splineFile):
        if splineFile is not None:
            self._splineFile = os.path.abspath(splineFile)
            self.spline = spline.Spline(self._splineFile)
            # NOTA : X is axis 1 and Y is Axis 0
            self._pixel2, self._pixel1 = self.spline.getPixelSize()
            self._splineCache = {}
        else:
            self._splineFile = None
            self.spline = None
    splineFile = property(get_splineFile, set_splineFile)

    def get_binning(self):
        return self._binning

    def set_binning(self, bin_size=(1, 1)):
        """
        Set the "binning" of the detector,

        @param bin_size: binning as integer or tuple of integers.
        @type bin_size: (int, int)
        """
        if "__len__" in dir(bin_size) and len(bin_size) >= 2:
            bin_size = (float(bin_size[0]), float(bin_size[1]))
        else:
            b = float(bin_size)
            bin_size = (b, b)
        if bin_size != self._binning:
            ratioX = bin_size[1] / self._binning[1]
            ratioY = bin_size[0] / self._binning[0]
            if self.spline is not None:
                self.spline.bin((ratioX, ratioY))
                self._pixel2, self._pixel1 = self.spline.getPixelSize()
                self._splineCache = {}
            else:
                self._pixel1 *= ratioY
                self._pixel2 *= ratioX
            self._binning = bin_size

    binning = property(get_binning, set_binning)


    def getPyFAI(self):
        """
        Helper method to serialize the description of a detector using the pyFAI way
        with everything in S.I units.

        @return: representation of the detector easy to serialize
        @rtype: dict
        """
        return {"detector": self.name,
                "pixel1": self._pixel1,
                "pixel2": self._pixel2,
                "splineFile": self._splineFile}

    def getFit2D(self):
        """
        Helper method to serialize the description of a detector using the Fit2d units

        @return: representation of the detector easy to serialize
        @rtype: dict
        """
        return {"pixelX": self._pixel2 * 1e6,
                "pixelY": self._pixel1 * 1e6,
                "splineFile": self._splineFile}

    def setPyFAI(self, **kwarg):
        """
        Twin method of getPyFAI: setup a detector instance according to a description

        @param kwarg: dictionary containing detector, pixel1, pixel2 and splineFile

        """
        if "detector" in kwarg:
            self = detector_factory(kwarg["detector"])
        for kw in kwarg:
            if kw in ["pixel1", "pixel2"]:
                setattr(self, kw, kwarg[kw])
            elif kw == "splineFile":
                self.set_splineFile(kwarg[kw])

    def setFit2D(self, **kwarg):
        """
        Twin method of getFit2D: setup a detector instance according to a description

        @param kwarg: dictionary containing pixel1, pixel2 and splineFile

        """
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

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)

        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if (d1 is None):
            d1 = numpy.outer(numpy.arange(self.max_shape[0]), numpy.ones(self.max_shape[1]))

        if (d2 is None):
            d2 = numpy.outer(numpy.ones(self.max_shape[0]), numpy.arange(self.max_shape[1]))

        if self.spline is None:
            dX = 0.
            dY = 0.
        else:
            if d2.ndim == 1:
                keyX = ("dX", tuple(d1), tuple(d2))
                keyY = ("dY", tuple(d1), tuple(d2))
                if keyX not in self._splineCache:
                    self._splineCache[keyX] = \
                        numpy.array([self.spline.splineFuncX(i2, i1)
                                     for i1, i2 in zip(d1 + 0.5, d2 + 0.5)],
                                    dtype="float64")
                if keyY not in self._splineCache:
                    self._splineCache[keyY] = \
                        numpy.array([self.spline.splineFuncY(i2, i1)
                                     for i1, i2 in zip(d1 + 0.5, d2 + 0.5)],
                                    dtype="float64")
                dX = self._splineCache[keyX]
                dY = self._splineCache[keyY]
            else:
                dX = self.spline.splineFuncX(d2 + 0.5, d1 + 0.5)
                dY = self.spline.splineFuncY(d2 + 0.5, d1 + 0.5)
        p1 = (self._pixel1 * (dY + 0.5 + d1))
        p2 = (self._pixel2 * (dX + 0.5 + d2))
        return p1, p2

    def calc_mask(self):
        """
        Detectors with gaps should overwrite this method with
        something actually calculating the mask!
        """
#        logger.debug("Detector.calc_mask is not implemented for generic detectors")
        return None

    ############################################################################
    # Few properties
    ############################################################################
    def get_mask(self):
        if self._mask is False:
            with self._sem:
                if self._mask is False:
                    self._mask = self.calc_mask()  # gets None in worse cases
                    if self._mask is not None:
                        self._mask_crc = crc32(self._mask)
        return self._mask
    def set_mask(self, mask):
        with self._sem:
            self._mask = mask
            if mask is not None:
                self._mask_crc = crc32(mask)
            else:
                self._mask_crc = None
    mask = property(get_mask, set_mask)
    def set_maskfile(self, maskfile):
        if fabio:
            with self._sem:
                self._mask = numpy.ascontiguousarray(fabio.open(maskfile).data,
                                                     dtype=numpy.int8)
                self._mask_crc = crc32(self._mask)
                self._maskfile = maskfile
        else:
            logger.error("FabIO is not available, unable to load the image to set the mask.")

    def get_maskfile(self):
        return self._maskfile
    maskfile = property(get_maskfile, set_maskfile)

    def get_pixel1(self):
        return self._pixel1
    def set_pixel1(self, value):
        if isinstance(value, float):
            self._pixel1 = value
        elif isinstance(value, (tuple, list)):
            self._pixel1 = float(value[0])
        else:
            self._pixel1 = float(value)
    pixel1 = property(get_pixel1, set_pixel1)

    def get_pixel2(self):
        return self._pixel2
    def set_pixel2(self, value):
        if isinstance(value, float):
            self._pixel2 = value
        elif isinstance(value, (tuple, list)):
            self._pixel2 = float(value[0])
        else:
            self._pixel2 = float(value)
    pixel2 = property(get_pixel2, set_pixel2)


class Pilatus(Detector):
    """
    Pilatus detector: generic description containing mask algorithm

    Sub-classed by Pilatus1M, Pilatus2M and Pilatus6M
    """
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    force_pixel = True
    def __init__(self, pixel1=172e-6, pixel2=172e-6, x_offset_file=None, y_offset_file=None):
        Detector.__init__(self, pixel1, pixel2)
        self.x_offset_file = x_offset_file
        self.y_offset_file = y_offset_file
        if self.x_offset_file and self.y_offset_file:
            if fabio:
                self.offset1 = fabio.open(self.y_offset_file).data
                self.offset2 = fabio.open(self.x_offset_file).data
            else:
                logging.error("FabIO is not available: no distortion correction for Pilatus detectors, sorry.")
                self.offset1 = None
                self.offset2 = None
        else:
            self.offset1 = None
            self.offset2 = None

    def __repr__(self):
        txt = "Detector %s\t PixelSize= %.3e, %.3e m" % \
                (self.name, self.pixel1, self.pixel2)
        if self.x_offset_file:
            txt += "\t delta_x= %s" % self.x_offset_file
        if self.y_offset_file:
            txt += "\t delta_y= %s" % self.y_offset_file
        return txt

    def get_splineFile(self):
        if self.x_offset_file and self.y_offset_file:
            return "%s,%s" % (self.x_offset_file, self.y_offset_file)

    def set_splineFile(self, splineFile=None):
        "In this case splinefile is a couple filenames"
        if splineFile is not None:
            try:
                files = splineFile.split(",")
                self.x_offset_file = [os.path.abspath(i) for i in files if "x" in i.lower()][0]
                self.y_offset_file = [os.path.abspath(i) for i in files if "y" in i.lower()][0]
            except Exception as error:
                logger.error("set_splineFile with %s gave error: %s" % (splineFile, error))
                self.x_offset_file = self.y_offset_file = self.offset1 = self.offset2 = None
                return
            if fabio:
                self.offset1 = fabio.open(self.y_offset_file).data
                self.offset2 = fabio.open(self.x_offset_file).data
            else:
                logging.error("FabIO is not available: no distortion correction for Pilatus detectors, sorry.")
                self.offset1 = None
                self.offset2 = None
        else:
            self._splineFile = None
    splineFile = property(get_splineFile, set_splineFile)

    def calc_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        if (self.max_shape[0] or self.max_shape[1]) is None:
            raise NotImplementedError("Generic Pilatus detector does not know"
                                      "the max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.MODULE_SIZE[0], self.max_shape[0],
                       self.MODULE_SIZE[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.MODULE_SIZE[1], self.max_shape[1],
                       self.MODULE_SIZE[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

    def calc_cartesian_positions(self, d1=None, d2=None):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)

        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if (d1 is None):
            d1 = numpy.outer(numpy.arange(self.max_shape[0]), numpy.ones(self.max_shape[1]))

        if (d2 is None):
            d2 = numpy.outer(numpy.ones(self.max_shape[0]), numpy.arange(self.max_shape[1]))

        if self.offset1 is None or self.offset2 is None:
            delta1 = delta2 = 0.
        else:
            if d2.ndim == 1:
                d1n = d1.astype(numpy.int32)
                d2n = d2.astype(numpy.int32)
                delta1 = self.offset1[d1n, d2n] / 100.0  # Offsets are in percent of pixel
                delta2 = self.offset2[d1n, d2n] / 100.0
            else:
                if d1.shape == self.offset1.shape:
                    delta1 = self.offset1 / 100.0  # Offsets are in percent of pixel
                    delta2 = self.offset2 / 100.0
                elif d1.shape[0] > self.offset1.shape[0]:  # probably working with corners
                    s0, s1 = self.offset1.shape
                    delta1 = numpy.zeros(d1.shape, dtype=numpy.int32)  # this is the natural type for pilatus CBF
                    delta2 = numpy.zeros(d2.shape, dtype=numpy.int32)
                    delta1[:s0, :s1] = self.offset1
                    delta2[:s0, :s1] = self.offset2
                    mask = numpy.where(delta1[-s0:, :s1] == 0)
                    delta1[-s0:, :s1][mask] = self.offset1[mask]
                    delta2[-s0:, :s1][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[-s0:, -s1:] == 0)
                    delta1[-s0:, -s1:][mask] = self.offset1[mask]
                    delta2[-s0:, -s1:][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[:s0, -s1:] == 0)
                    delta1[:s0, -s1:][mask] = self.offset1[mask]
                    delta2[:s0, -s1:][mask] = self.offset2[mask]
                    delta1 = delta1 / 100.0  # Offsets are in percent of pixel
                    delta2 = delta2 / 100.0  # former arrays were integers
                else:
                    logger.warning("Surprizing situation !!! please investigate: offset has shape %s and input array have %s" % (self.offset1.shape, d1, shape))
                    delta1 = delta2 = 0.
        # For pilatus,
        p1 = (self._pixel1 * (delta1 + 0.5 + d1))
        p2 = (self._pixel2 * (delta2 + 0.5 + d2))
        return p1, p2

class Pilatus100k(Pilatus):
    """
    Pilatus 100k detector
    """
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (195, 487)


class Pilatus200k(Pilatus):
    """
    Pilatus 200k detector
    """
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (407, 487)


class Pilatus300k(Pilatus):
    """
    Pilatus 300k detector
    """
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (619, 487)


class Pilatus300kw(Pilatus):
    """
    Pilatus 300k-wide detector
    """
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (195, 1475)


class Pilatus1M(Pilatus):
    """
    Pilatus 1M detector
    """
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (1043, 981)


class Pilatus2M(Pilatus):
    """
    Pilatus 2M detector
    """
    force_pixel = True
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (1679, 1475)


class Pilatus6M(Pilatus):
    """
    Pilatus 6M detector
    """
    force_pixel = True
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        Pilatus.__init__(self, pixel1, pixel2)
        self.max_shape = (2527, 2463)


class Fairchild(Detector):
    """
    Fairchild Condor 486:90 detector
    """
    force_pixel = True
    def __init__(self, pixel1=15e-6, pixel2=15e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.name = "Fairchild Condor 486:90"
        self.max_shape = (4096, 4096)

class Titan(Detector):
    """
    Titan CCD detector from Agilent. Mask not handled
    """
    force_pixel = True
    def __init__(self, pixel1=60e-6, pixel2=60e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.name = "Titan 2k x 2k"
        self.max_shape = (2048, 2048)




class Dexela2923(Detector):
    """
    Dexela CMOS family detector
    """
    force_pixel = True
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.name = "Dexela 2923"
        self.max_shape = (3888, 3072)


class FReLoN(Detector):
    """
    FReLoN detector:
    The spline is mandatory to correct for geometric distortion of the taper

    TODO: create automatically a mask that removes pixels out of the "valid reagion"
    """
    def __init__(self, splineFile=None):
        Detector.__init__(self, splineFile=splineFile)
        if splineFile:
            self.max_shape = (int(self.spline.ymax - self.spline.ymin),
                              int(self.spline.xmax - self.spline.xmin))
        else:
            self.max_shape = (2048, 2048)

    def calc_mask(self):
        """
        Returns a generic mask for Frelon detectors...
        All pixels which (center) turns to be out of the valid region are by default discarded
        """

        d1 = numpy.outer(numpy.arange(self.max_shape[0]), numpy.ones(self.max_shape[1])) + 0.5
        d2 = numpy.outer(numpy.ones(self.max_shape[0]), numpy.arange(self.max_shape[1])) + 0.5
        dX = self.spline.splineFuncX(d2, d1)
        dY = self.spline.splineFuncY(d2, d1)
        p1 = dY + d1
        p2 = dX + d2
        below_min = numpy.logical_or((p2 < self.spline.xmin), (p1 < self.spline.ymin))
        above_max = numpy.logical_or((p2 > self.spline.xmax), (p1 > self.spline.ymax))
        mask = numpy.logical_or(below_min, above_max)
        return mask

class Basler(Detector):
    """
    Basler camera are simple CCD camara over GigaE

    """
    force_pixel = True
    def __init__(self, pixel=3.75e-6):
        Detector.__init__(self, pixel1=pixel, pixel2=pixel)
        self.max_shape = (966, 1296)


class Xpad_flat(Detector):
    """
    Xpad detector: generic description for
    ImXPad detector with 8x7modules
    """
    MODULE_SIZE = (120, 80)
    MODULE_GAP = (3 + 3.57 * 1000 / 130, 3)  # in pixels
    force_pixel = True
    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.max_shape = (960, 560)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
                (self.name, self.pixel1, self.pixel2)

    def calc_mask(self):
        """
        Returns a generic mask for Xpad detectors...
        discards the first line and raw form all modules:
        those are 2.5x bigger and often mis - behaving
        """
        if (self.max_shape[0] or self.max_shape[1]) is None:
            raise NotImplementedError("Generic Xpad detector does not"
                                      " know the max size ...")
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

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)

        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.

        """
        if (d1 is None):
            c1 = numpy.arange(self.max_shape[0])
            for i in range(self.max_shape[0] // self.MODULE_SIZE[0]):
                c1[i * self.MODULE_SIZE[0],
                   (i + 1) * self.MODULE_SIZE[0]] += i * self.MODULE_GAP[0]
        else:
            c1 = d1 + (d1.astype(numpy.int64) // self.MODULE_SIZE[0])\
                * self.MODULE_GAP[0]
        if (d2 is None):
            c2 = numpy.arange(self.max_shape[1])
            for i in range(self.max_shape[1] // self.MODULE_SIZE[1]):
                c2[i * self.MODULE_SIZE[1],
                   (i + 1) * self.MODULE_SIZE[1]] += i * self.MODULE_GAP[1]
        else:
            c2 = d2 + (d2.astype(numpy.int64) // self.MODULE_SIZE[1])\
                * self.MODULE_GAP[1]

        p1 = self.pixel1 * (0.5 + c1)
        p2 = self.pixel2 * (0.5 + c2)
        return p1, p2


def _pixels_compute_center(pixels_size):
    """
    given a list of pixel size, this method return the center of each
    pixels. This method is generic.

    @param pixels_size: the size of the pixels.
    @type length: ndarray

    @return: the center-coordinates of each pixels 0..length
    @rtype: ndarray
    """
    center = pixels_size.cumsum()
    tmp = center.copy()
    center[1:] += tmp[:-1]
    center /= 2.

    return center

def _pixels_extract_coordinates(coordinates, pixels):
    """
    given a list of pixel coordinates, return the correspondig
    pixels coordinates extracted from the coodinates array.

    @param coodinates: the pixels coordinates
    @type coordinates: ndarray 1D (pixels -> coordinates)
    @param pixels: the list of pixels to extract.
    @type pixels: ndarray 1D(calibration) or 2D(integration)

    @return: the pixels coordinates
    @rtype: ndarray
    """
    return coordinates[pixels] if (pixels is not None) else coordinates

class ImXPadS140(Detector):
    """
    ImXPad detector: ImXPad s140 detector with 2x7modules
    """
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    MAX_SHAPE = (240, 560)  # max size of the detector
    PIXEL_SIZE = (130e-6, 130e-6)
    force_pixel = True
    name = "Imxpad S140"

    class __metaclass__(type):

        @lazy_property
        def COORDINATES(cls):
            """
            cache used to store the coordinates of the y, x, detector
            pixels. These array are compute only once for all
            instances.
            """
            return tuple(_pixels_compute_center(cls._pixels_size(n, m, p))
                         for n, m, p in zip(cls.MAX_SHAPE,
                                            cls.MODULE_SIZE,
                                            cls.PIXEL_SIZE))

    @staticmethod
    def _pixels_size(length, module_size, pixel_size):
        """
        given the length (in pixel) of the detector, the size of a
        module (in pixels) and the pixel_size (in meter). this method
        return the length of each pixels 0..length.

        @param length: the number of pixel to compute
        @type length: int
        @param module_size: the number of pixel of one module
        @type module_size: int
        @param pixel_size: the size of one pixels (meter per pixel)
        @type length: float

        @return: the coordinates of each pixels 0..length
        @rtype: ndarray
        """
        size = numpy.ones(length)
        n = length // module_size
        for i in range(1, n):
            size[i * module_size - 1] = 2.5
            size[i * module_size] = 2.5
        return pixel_size * size

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.name = "ImXPad S140"
        self.max_shape = self.MAX_SHAPE

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self.pixel1, self.pixel2)


    def calc_cartesian_positions(self, d1=None, d2=None):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)

        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.

        """
        return tuple(_pixels_extract_coordinates(coordinates, pixels)
                     for coordinates, pixels in zip(ImXPadS140.COORDINATES,
                                                    (d1, d2)))


class Perkin(Detector):
    """
    Perkin detector

    """
    force_pixel = True
    def __init__(self, pixel=200e-6):
        Detector.__init__(self, pixel, pixel)
        self.name = "Perkin detector"
        self.max_shape = (2048, 2048)

class RayonixMx225(Detector):
    """
    Rayonix mx225 2D detector
    """
    force_pixel = True
    def __init__(self):
        Detector.__init__(self, pixel1=73e-6, pixel2=73e-6)
        self.max_shape = (3072, 3072)
        self.name = "Rayonix mx225"

class RayonixMx300(Detector):
    """
    Rayonix mx300 2D detector
    """
    force_pixel = True
    def __init__(self):
        Detector.__init__(self, pixel1=73e-6, pixel2=73e-6)
        self.max_shape = (4096, 4096)
        self.name = "Rayonix mx300"

class RayonixMx325(Detector):
    """
    Rayonix mx325 2D detector
    """
    force_pixel = True
    def __init__(self):
        Detector.__init__(self, pixel1=79e-6, pixel2=79e-6)
        self.max_shape = (4096, 4096)
        self.name = "Rayonix mx325"

ALL_DETECTORS = {}
#Init time creation of the dict of all detectors
local_dict = locals()
for obj_name in dir():
    obj_class = local_dict.get(obj_name)
    if "isDetector" in dir(obj_class):
        try:
            obj_inst = obj_class()
        except:
            logger.debug("Unable to instanciate %s" % obj_name)
            pass
        else:
            ALL_DETECTORS[obj_name.lower()] = obj_class
            ALL_DETECTORS[obj_inst.name.lower().replace(" ", "_")] = obj_class

def detector_factory(name, config=None):
    """
    A kind of factory...
    @param name: name of a detector
    @type name: str
    @param config: configuration of the detector
    @type config: dict or JSON representation of it.

    @return: an instance of the right detector, set-up if possible
    @rtype: pyFAI.detectors.Detector
    """
    name = name.lower()
    if name in ALL_DETECTORS:
        mydet = ALL_DETECTORS[name]()
        if config is not None:
            mydet.set_config(config)
        return mydet
    else:
        msg = ("Detector %s is unknown !, "
               "please select one from %s" % (name, ALL_DETECTORS.keys()))
        logger.error(msg)
        raise RuntimeError(msg)

