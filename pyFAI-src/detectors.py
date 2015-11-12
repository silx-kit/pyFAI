# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
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
from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/11/2015"
__status__ = "stable"
__doc__ = """
Module containing the description of all detectors with a factory to instantiate them
"""


import logging
import numpy
import os
import posixpath
import threading

from . import io
from . import spline
from .utils import binning, expand2d

logger = logging.getLogger("pyFAI.detectors")

try:
    from .fastcrc import crc32
except ImportError:
    from zlib import crc32
try:
    from . import bilinear
except ImportError:
     bilinear = None
try:
    import fabio
except ImportError:
    fabio = None
try:
    from six import with_metaclass
except ImportError:
    from .third_party.six import with_metaclass


epsilon = 1e-6


class DetectorMeta(type):
    """
    Metaclass used to register all detector classes inheriting from Detector
    """
    # we use __init__ rather than __new__ here because we want
    # to modify attributes of the class *after* they have been
    # created
    def __init__(cls, name, bases, dct):
        # "Detector" is a bit peculiar: while abstract it may be needed by the GUI, so adding it to the repository
        if hasattr(cls, 'MAX_SHAPE') or name == "Detector":
            cls.registry[name.lower()] = cls
            if hasattr(cls, "aliases"):
                for alias in cls.aliases:
                    cls.registry[alias.lower().replace(" ", "_")] = cls
                    cls.registry[alias.lower().replace(" ", "")] = cls

        super(DetectorMeta, cls).__init__(name, bases, dct)


class Detector(with_metaclass(DetectorMeta, object)):
    """
    Generic class representing a 2D detector
    """
    force_pixel = False  # Used to specify pixel size should be defined by the class itself.
    aliases = []  # list of alternative names
    registry = {}  # list of  detectors ...
    uniform_pixel = True  # tells all pixels have the same size
    IS_FLAT = True  # this detector is flat

    @classmethod
    def factory(cls, name, config=None):
        """
        A kind of factory...

        @param name: name of a detector
        @type name: str
        @param config: configuration of the detector
        @type config: dict or JSON representation of it.

        @return: an instance of the right detector, set-up if possible
        @rtype: pyFAI.detectors.Detector
        """
        if os.path.isfile(name):
            return NexusDetector(name)
        name = name.lower()
        names = [name, name.replace(" ", "_")]
        for name in names:
            if name in cls.registry:
                mydet = cls.registry[name]()
                if config is not None:
                    mydet.set_config(config)
                return mydet
        else:
            msg = ("Detector %s is unknown !, "
                   "please check if the filename exists or select one from %s" % (name, cls.registry.keys()))
            logger.error(msg)
            raise RuntimeError(msg)

    def __init__(self, pixel1=None, pixel2=None, splineFile=None):
        """
        @param pixel1: size of the pixel in meter along the slow dimension (often Y)
        @type pixel1: float
        @param pixel2: size of the pixel in meter along the fast dimension (often X)
        @type pixel2: float
        @param splineFile: path to file containing the geometric correction.
        @type splineFile: str
        """
        self._pixel1 = None
        self._pixel2 = None
        self._pixel_corners = None

        if pixel1:
            self._pixel1 = float(pixel1)
        if pixel2:
            self._pixel2 = float(pixel2)
        if "MAX_SHAPE" in dir(self.__class__):
            self.max_shape = tuple(self.MAX_SHAPE)
        else:
            self.max_shape = None
        self.shape = self.max_shape
        self._binning = (1, 1)
        self._mask = False
        self._mask_crc = None
        self._maskfile = None
        self._splineFile = None
        self.spline = None
        self._dx = None
        self._dy = None
        self.flat = None
        self.dark = None
        self._splineCache = {}  # key=(dx,xpoints,ypoints) value: ndarray
        self._sem = threading.Semaphore()
        if splineFile:
            self.set_splineFile(splineFile)

    def __repr__(self):
        """Nice representation of the instance
        """
        if (self._pixel1 is None) or (self._pixel2 is None):
            return "Undefined detector"
        return "Detector %s\t Spline= %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self.splineFile, self._pixel1, self._pixel2)

    def __copy__(self):
        "@return a shallow copy of itself"
        new = self.__class__()
        numerical = ['_pixel1', '_pixel2', 'max_shape', 'shape', '_binning', '_mask_crc', '_maskfile']
        array = ['_mask', '_dx', '_dy', 'flat', 'dark']
        for key in numerical + array:
            new.__setattr__(key, self.__getattribute__(key))
        if self._splineFile:
            new.set_splineFile(self._splineFile)
        return new

    def __deepcopy__(self, memo):
        "@return a deep copy of itself"
        new = self.__class__()
        numerical = ['_pixel1', '_pixel2', 'max_shape', 'shape', '_binning', '_mask_crc', '_maskfile']
        for key in numerical:
            new.__setattr__(key, self.__getattribute__(key))
        array = ['_mask', '_dx', '_dy', 'flat', 'dark']
        for key in array:
            value = self.__getattribute__(key)
            if (value is None) or (value is False):
                new.__setattr__(key, value)
            elif "copy" in dir(value):
                new.__setattr__(key, value.copy())
            else:
                new.__setattr__(key, 1 * value)

        if self._splineFile:
            new.set_splineFile(self._splineFile)
        return new

    def set_config(self, config):
        """
        Sets the configuration of the detector. This implies:
        - Orientation: integers
        - Binning
        - ROI

        The configuration is either a python dictionary or a JSON string or a file containing this JSON configuration

        keys in that dictionary are :
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
            self.uniform_pixel = False
            self.max_shape = (int(self.spline.ymax - self.spline.ymin), int(self.spline.xmax - self.spline.xmin))
            # assume no binning
            self.shape = self.max_shape
            self._binning = (1, 1)
        else:
            self._splineFile = None
            self.spline = None
            self.uniform_pixel = True
    splineFile = property(get_splineFile, set_splineFile)

    def set_dx(self, dx=None):
        """
        set the pixel-wise displacement along X (dim2):
        """
        if dx is not None:
            if not self.max_shape:
                raise RuntimeError("Set detector shape before setting the distortion")
            if dx.shape == self.max_shape:
                self._dx = dx
            elif dx.shape == tuple(i + 1 for i in self.max_shape):
                if self._pixel_corners is None:
                    self.get_pixel_corners()
                d2 = numpy.outer(numpy.ones(self.shape[0] + 1), numpy.arange(self.shape[1] + 1))
                p2 = (self._pixel2 * (dx + d2))
                self._pixel_corners[:, :, 0, 2] = p2[:-1, :-1]
                self._pixel_corners[:, :, 1, 2] = p2[1:, :-1]
                self._pixel_corners[:, :, 2, 2] = p2[1:, 1:]
                self._pixel_corners[:, :, 3, 2] = p2[:-1, 1:]
            else:
                raise RuntimeError("detector shape:%s while distortionarray: %s" % (self.max_shape, dx.shape))
            self.uniform_pixel = False
        else:
            self._dx = None
            self.uniform_pixel = True

    def set_dy(self, dy=None):
        """
        set the pixel-wise displacement along Y (dim1):
        """
        if dy is not None:
            if not self.max_shape:
                raise RuntimeError("Set detector shape before setting the distortion")

            if dy.shape == self.max_shape:
                self._dy = dy
            elif dy.shape == tuple(i + 1 for i in self.max_shape):
                if self._pixel_corners is None:
                    self.get_pixel_corners()
                d1 = numpy.outer(numpy.arange(self.shape[0] + 1), numpy.ones(self.shape[1] + 1))
                p1 = (self._pixel1 * (dy + d1))
                self._pixel_corners[:, :, 0, 1] = p1[:-1, :-1]
                self._pixel_corners[:, :, 1, 1] = p1[1:, :-1]
                self._pixel_corners[:, :, 2, 1] = p1[1:, 1:]
                self._pixel_corners[:, :, 3, 1] = p1[:-1, 1:]
            else:
                raise RuntimeError("detector shape:%s while distortion array: %s" % (self.max_shape, dy.shape))
            self.uniform_pixel = False
        else:
            self._dy = None
            self.uniform_pixel = True

    def get_binning(self):
        return self._binning

    def set_binning(self, bin_size=(1, 1)):
        """
        Set the "binning" of the detector,

        @param bin_size: binning as integer or tuple of integers.
        @type bin_size: (int, int)
        """
        if "__len__" in dir(bin_size) and len(bin_size) >= 2:
            bin_size = int(round(float(bin_size[0]))), int(round(float(bin_size[1])))
        else:
            b = int(round(float(bin_size)))
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
            self.shape = (self.max_shape[0] // bin_size[0],
                          self.max_shape[1] // bin_size[1])
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

    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!
        Adapted to Nexus detector definition

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)
        @param center: retrieve the coordinate of the center of the pixel
        @param use_cython: set to False to test Python implementation
        @return: position in meter of the center of each pixels.
        @rtype: 3xndarray, the later being None if IS_FLAT

        d1 and d2 must have the same shape, returned array will have
        the same shape.

        pos_z is None for flat detectors
        """
        if self.shape:
            if (d1 is None) or (d2 is None):
                d1 = expand2d(numpy.arange(float(self.shape[0])), self.shape[1], False)
                d2 = expand2d(numpy.arange(float(self.shape[1])), self.shape[0], True)

        elif "ndim" in dir(d1):
            if d1.ndim == 2:
                self.shape = d1.shape
        elif "ndim" in dir(d2):
            if d2.ndim == 2:
                self.shape = d2.shape

        if center:
            # avoid += It modifies in place and segfaults
            d1c = d1 + 0.5
            d2c = d2 + 0.5
        else:
            d1c = d1
            d2c = d2

        if self._pixel_corners is not None:
            p3 = None
            if bilinear and use_cython:
                p1, p2, p3 = bilinear.calc_cartesian_positions(d1c.ravel(), d2c.ravel(),
                                                               self._pixel_corners,
                                                               is_flat=self.IS_FLAT)
                p1.shape = d1.shape
                p2.shape = d1.shape
                if p3 is not None:
                    p3.shape = d1.shape
            else:
                i1 = d1.astype(int).clip(0, self._pixel_corners.shape[0] - 1)
                i2 = d2.astype(int).clip(0, self._pixel_corners.shape[1] - 1)
                delta1 = d1 - i1
                delta2 = d2 - i2
                pixels = self._pixel_corners[i1, i2]
                A1 = pixels[:, :, 0, 1]
                A2 = pixels[:, :, 0, 2]
                B1 = pixels[:, :, 1, 1]
                B2 = pixels[:, :, 1, 2]
                C1 = pixels[:, :, 2, 1]
                C2 = pixels[:, :, 2, 2]
                D1 = pixels[:, :, 3, 1]
                D2 = pixels[:, :, 3, 2]
                # points A and D are on the same dim1 (Y), they differ in dim2 (X)
                # points B and C are on the same dim1 (Y), they differ in dim2 (X)
                # points A and B are on the same dim2 (X), they differ in dim1 (Y)
                # points C and D are on the same dim2 (X), they differ in dim1 (Y)

                p1 = A1 * (1.0 - delta1) * (1.0 - delta2) \
                   + B1 * delta1 * (1.0 - delta2) \
                   + C1 * delta1 * delta2 \
                   + D1 * (1.0 - delta1) * delta2
                p2 = A2 * (1.0 - delta1) * (1.0 - delta2) \
                   + B2 * delta1 * (1.0 - delta2) \
                   + C2 * delta1 * delta2 \
                   + D2 * (1.0 - delta1) * delta2
                if not self.IS_FLAT:
                    A0 = pixels[:, :, 0, 0]
                    B0 = pixels[:, :, 1, 0]
                    C0 = pixels[:, :, 2, 0]
                    D0 = pixels[:, :, 3, 0]
                    p3 = A0 * (1.0 - delta1) * (1.0 - delta2) \
                       + B0 * delta1 * (1.0 - delta2) \
                       + C0 * delta1 * delta2 \
                       + D0 * (1.0 - delta1) * delta2
            return p1, p2, p3

        elif self.spline is not None:
            if d2.ndim == 1:
                keyX = ("dX", tuple(d1), tuple(d2))
                keyY = ("dY", tuple(d1), tuple(d2))
                if keyX not in self._splineCache:
                    self._splineCache[keyX] = self.spline.splineFuncX(d2c, d1c, True).astype(numpy.float64)
                if keyY not in self._splineCache:
                    self._splineCache[keyY] = self.spline.splineFuncY(d2c, d1c, True).astype(numpy.float64)
                dX = self._splineCache[keyX]
                dY = self._splineCache[keyY]
            else:
                dX = self.spline.splineFuncX(d2c, d1c)
                dY = self.spline.splineFuncY(d2c, d1c)
        elif self._dx is not None:
            if self._binning == (1, 1):
                binned_x = self._dx
                binned_y = self._dy
            else:
                binned_x = binning(self._dx, self._binning)
                binned_y = binning(self._dy, self._binning)
            dX = numpy.interp(d2, numpy.arange(binned_x.shape[1]), binned_x, left=0, right=0)
            dY = numpy.interp(d1, numpy.arange(binned_y.shape[0]), binned_y, left=0, right=0)
        else:
            dX = 0.
            dY = 0.

        p1 = (self._pixel1 * (dY + d1c))
        p2 = (self._pixel2 * (dX + d2c))
        return p1, p2, None

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
            value = value
        elif isinstance(value, (tuple, list)):
            value = float(value[0])
        else:
            value = float(value)
        if self._pixel1:
            err = abs(value - self._pixel1) / self._pixel1
            if self.force_pixel and  (err > epsilon):
                logger.warning("Enforcing pixel size 1 for a detector %s" %
                               self.__class__.__name__)
        self._pixel1 = value
    pixel1 = property(get_pixel1, set_pixel1)

    def get_pixel2(self):
        return self._pixel2
    def set_pixel2(self, value):
        if isinstance(value, float):
            value = value
        elif isinstance(value, (tuple, list)):
            value = float(value[0])
        else:
            value = float(value)
        if self._pixel2:
            err = abs(value - self._pixel2) / self._pixel2
            if self.force_pixel and  (err > epsilon):
                logger.warning("Enforcing pixel size 2 for a detector %s" %
                               self.__class__.__name__)
        self._pixel2 = value
    pixel2 = property(get_pixel2, set_pixel2)

    def get_name(self):
        """
        Get a meaningful name for detector
        """
        if self.aliases:
            name = self.aliases[0]
        else:
            name = self.__class__.__name__
        return name
    name = property(get_name)

    def get_pixel_corners(self):
        """
        Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)

        Precision float32 is ok: precision of 1µm for a detector size of 1m


        @return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    d1 = expand2d(numpy.arange(self.shape[0] + 1.0) - 0.5, self.shape[1] + 1, False)
                    d2 = expand2d(numpy.arange(self.shape[1] + 1.0) - 0.5, self.shape[0] + 1, True)
                    p1, p2, p3 = self.calc_cartesian_positions(d1, d2)
                    self._pixel_corners = numpy.zeros((self.shape[0], self.shape[1], 4, 3), dtype=numpy.float32)
                    self._pixel_corners[:, :, 0, 1] = p1[:-1, :-1]
                    self._pixel_corners[:, :, 0, 2] = p2[:-1, :-1]
                    self._pixel_corners[:, :, 1, 1] = p1[1:, :-1]
                    self._pixel_corners[:, :, 1, 2] = p2[1:, :-1]
                    self._pixel_corners[:, :, 2, 1] = p1[1:, 1:]
                    self._pixel_corners[:, :, 2, 2] = p2[1:, 1:]
                    self._pixel_corners[:, :, 3, 1] = p1[:-1, 1:]
                    self._pixel_corners[:, :, 3, 2] = p2[:-1, 1:]
                    if p3 is not None:
                        # non flat detector
                        self._pixel_corners[:, :, 0, 0] = p3[:-1, :-1]
                        self._pixel_corners[:, :, 1, 0] = p3[1:, :-1]
                        self._pixel_corners[:, :, 2, 0] = p3[1:, 1:]
                        self._pixel_corners[:, :, 3, 0] = p3[:-1, 1:]
        return self._pixel_corners

    def save(self, filename):
        """
        Saves the detector description into a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html
        Main differences:

            * differentiate pixel center from pixel corner offsets
            * store all offsets are ndarray according to slow/fast dimention (not x, y)

        @param filename: name of the file on the disc
        """
        if not io.h5py:
            logger.error("h5py module missing: NeXus detectors not supported")
            raise RuntimeError("H5py module is missing")

        with io.Nexus(filename, "+") as nxs:
            det_grp = nxs.new_detector(name=self.name.replace(" ", "_"))
            det_grp["IS_FLAT"] = self.IS_FLAT
            det_grp["pixel_size"] = numpy.array([self.pixel1, self.pixel2], dtype=numpy.float32)
            if self.max_shape is not None:
                det_grp["max_shape"] = numpy.array(self.max_shape, dtype=numpy.int32)
            if self.shape is not None:
                det_grp["shape"] = numpy.array(self.shape, dtype=numpy.int32)
            if self.binning is not None:
                det_grp["binning"] = numpy.array(self._binning, dtype=numpy.int32)
            if self.flat is not None:
                dset = det_grp.create_dataset("flat", data=self.flat,
                                              compression="gzip", compression_opts=9, shuffle=True)
                dset.attrs["interpretation"] = "image"
            if self.mask is not None:
                dset = det_grp.create_dataset("mask", data=self.mask,
                                              compression="gzip", compression_opts=9, shuffle=True)
                dset.attrs["interpretation"] = "image"
            if not (self.uniform_pixel and self.IS_FLAT):
                # Get ready for the worse case: 4 corner per pixel, position 3D: z,y,x
                dset = det_grp.create_dataset("pixel_corners", data=self.get_pixel_corners(),
                                              compression="gzip", compression_opts=9, shuffle=True)
                dset.attrs["interpretation"] = "vertex"

    def guess_binning(self, data):
        """
        Guess the binning/mode depending on the image shape
        @param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        """
        if "shape" in dir(data):
            shape = data.shape
        else:
            shape = tuple(data[:2])
        if not self.force_pixel:
            if shape != self.max_shape:
                logger.warning("guess_binning is not implemented for %s detectors!\
                 and image size %s! is wrong, expected %s!" % (self.name, shape, self.shape))
        elif self.max_shape:
            bin1 = self.max_shape[0] // shape[0]
            bin2 = self.max_shape[1] // shape[1]
            res = self.max_shape[0] % shape[0] + self.max_shape[1] % shape[1]
            if res != 0:
                logger.warning("Impossible binning: max_shape is %s, requested shape %s" % (self.max_shape, shape))
            old_binning = self._binning
            self._binning = (bin1, bin2)
            self.shape = shape
            self._pixel1 *= (1.0 * bin1 / old_binning[0])
            self._pixel2 *= (1.0 * bin2 / old_binning[1])
            self._mask = False
            self._mask_crc = None
        else:
            logger.debug("guess_binning for generic detectors !")


class NexusDetector(Detector):
    """
    Class representing a 2D detector loaded from a NeXus file
    """
    def __init__(self, filename=None):
        Detector.__init__(self)
        if filename is not None:
            self.load(filename)
        self._filename = filename
        self.uniform_pixel = True

    def __repr__(self):
        return "%s detector from NeXus file: %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._filename, self._pixel1, self._pixel2)

    def load(self, filename):
        """
        Loads the detector description from a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html

        @param filename: name of the file on the disk
        """
        if not io.h5py:
            logger.error("h5py module missing: NeXus detectors not supported")
            raise RuntimeError("H5py module is missing")
        with io.Nexus(filename, "r") as nxs:
            det_grp = nxs.find_detector()
            name = posixpath.split(det_grp.name)[-1]
            self.aliases = [name.replace("_", " "), det_grp.name]
            if "IS_FLAT" in det_grp:
                self.IS_FLAT = det_grp["IS_FLAT"].value
            if "flat" in det_grp:
                self.flat = det_grp["flat"].value
            if "pixel_size" in det_grp:
                self.pixel1, self.pixel2 = det_grp["pixel_size"]
            if "binning" in det_grp:
                self._binning = tuple(i for i in det_grp["binning"].value)
            for what in ("max_shape", "shape"):
                if what in det_grp:
                    self.__setattr__(what, tuple(i for i in det_grp[what].value))
            if "mask" in det_grp:
                self.mask = det_grp["mask"].value
            if "pixel_corners" in det_grp:
                self._pixel_corners = det_grp["pixel_corners"].value
                self.uniform_pixel = False
            else:
                self.uniform_pixel = True

    @classmethod
    def sload(cls, filename):
        """
        Instantiate the detector description from a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html

        @param filename: name of the file on the disk
        @return: Detector instance
        """
        obj = cls()
        cls.load(filename)
        return obj

    def getPyFAI(self):
        """
        Helper method to serialize the description of a detector using the pyFAI way
        with everything in S.I units.

        @return: representation of the detector easy to serialize
        @rtype: dict
        """
        return {"detector": self._filename or self.name,
                "pixel1": self._pixel1,
                "pixel2": self._pixel2
                }

    def getFit2D(self):
        """
        Helper method to serialize the description of a detector using the Fit2d units

        @return: representation of the detector easy to serialize
        @rtype: dict
        """
        return {"pixelX": self._pixel2 * 1e6,
                "pixelY": self._pixel1 * 1e6
                }


class Pilatus(Detector):
    """
    Pilatus detector: generic description containing mask algorithm

    Sub-classed by Pilatus1M, Pilatus2M and Pilatus6M
    """
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    force_pixel = True

    def __init__(self, pixel1=172e-6, pixel2=172e-6, x_offset_file=None, y_offset_file=None):
        super(Pilatus, self).__init__(pixel1=pixel1, pixel2=pixel2)
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
                self.uniform_pixel = False
            except Exception as error:
                logger.error("set_splineFile with %s gave error: %s" % (splineFile, error))
                self.x_offset_file = self.y_offset_file = self.offset1 = self.offset2 = None
                self.uniform_pixel = True
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
            self.uniform_pixel = True
    splineFile = property(get_splineFile, set_splineFile)

    def calc_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Pilatus detector does not know "
                                      "its max size ...")
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
        if (d1 is None) or (d2 is None):
            d1 = numpy.outer(numpy.arange(self.max_shape[0]), numpy.ones(self.max_shape[1]))
            d2 = numpy.outer(numpy.ones(self.max_shape[0]), numpy.arange(self.max_shape[1]))

        if (self.offset1 is None) or (self.offset2 is None):
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
                    logger.warning("Surprizing situation !!! please investigate: offset has shape %s and input array have %s" % (self.offset1.shape, d1.shape))
                    delta1 = delta2 = 0.
        # For pilatus,
        p1 = (self._pixel1 * (delta1 + 0.5 + d1))
        p2 = (self._pixel2 * (delta2 + 0.5 + d2))
        return p1, p2, None


class Pilatus100k(Pilatus):
    """
    Pilatus 100k detector
    """
    MAX_SHAPE = 195, 487
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus100k, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus200k(Pilatus):
    """
    Pilatus 200k detector
    """
    MAX_SHAPE = (407, 487)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus200k, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus300k(Pilatus):
    """
    Pilatus 300k detector
    """
    MAX_SHAPE = (619, 487)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus300k, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus300kw(Pilatus):
    """
    Pilatus 300k-wide detector
    """
    MAX_SHAPE = (195, 1475)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus300kw, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus1M(Pilatus):
    """
    Pilatus 1M detector
    """
    MAX_SHAPE = (1043, 981)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus1M, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus2M(Pilatus):
    """
    Pilatus 2M detector
    """

    MAX_SHAPE = 1679, 1475
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus2M, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Pilatus6M(Pilatus):
    """
    Pilatus 6M detector
    """
    MAX_SHAPE = (2527, 2463)
    def __init__(self, pixel1=172e-6, pixel2=172e-6):
        super(Pilatus6M, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Eiger(Detector):
    """
    Eiger detector: generic description containing mask algorithm
    """
    MODULE_SIZE = (514, 1030)
    MODULE_GAP = (37, 10)
    force_pixel = True

    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)
        self.offset1 = self.offset2 = None

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def calc_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        if self.max_shape is None:
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
            d1 = expand2d(numpy.arange(float(self.max_shape[0])), self.shape[1], False)
        if (d2 is None):
            d2 = expand2d(numpy.arange(float(self.max_shape[1])), self.shape[0], True)

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
                    logger.warning("Surprising situation !!! please investigate: offset has shape %s and input array have %s" % (self.offset1.shape, d1.shape))
                    delta1 = delta2 = 0.
        # For pilatus,
        p1 = (self._pixel1 * (delta1 + 0.5 + d1))
        p2 = (self._pixel2 * (delta2 + 0.5 + d2))
        return p1, p2, None


class Eiger1M(Eiger):
    """
    Eiger 1M detector
    """
    MAX_SHAPE = (1065, 1030)
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Eiger.__init__(self, pixel1=pixel1, pixel2=pixel2)


class Eiger4M(Eiger):
    """
    Eiger 4M detector
    """
    MAX_SHAPE = (2167, 2070)
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Eiger.__init__(self, pixel1=pixel1, pixel2=pixel2)


class Eiger9M(Eiger):
    """
    Eiger 9M detector
    """
    MAX_SHAPE = (3269, 3110)
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Eiger.__init__(self, pixel1=pixel1, pixel2=pixel2)


class Eiger16M(Eiger):
    """
    Eiger 16M detector
    """
    MAX_SHAPE = (4371, 4150)
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        Eiger.__init__(self, pixel1=pixel1, pixel2=pixel2)


class Fairchild(Detector):
    """
    Fairchild Condor 486:90 detector
    """
    force_pixel = True
    uniform_pixel = True
    aliases = ["Fairchild", "Condor", "Fairchild Condor 486:90"]
    MAX_SHAPE = (4096, 4096)
    def __init__(self, pixel1=15e-6, pixel2=15e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)


class Titan(Detector):
    """
    Titan CCD detector from Agilent. Mask not handled
    """
    force_pixel = True
    MAX_SHAPE = (2048, 2048)
    aliases = ["Titan 2k x 2k", "OXD Titan", "Agilent Titan"]
    uniform_pixel = True
    def __init__(self, pixel1=60e-6, pixel2=60e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)


class Dexela2923(Detector):
    """
    Dexela CMOS family detector
    """
    force_pixel = True
    aliases = ["Dexela 2923"]
    MAX_SHAPE = (3888, 3072)
    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        super(Dexela2923, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)


class FReLoN(Detector):
    """
    FReLoN detector:
    The spline is mandatory to correct for geometric distortion of the taper

    TODO: create automatically a mask that removes pixels out of the "valid reagion"
    """
    def __init__(self, splineFile=None):
        super(FReLoN, self).__init__(splineFile=splineFile)
        if splineFile:
            self.max_shape = (int(self.spline.ymax - self.spline.ymin),
                              int(self.spline.xmax - self.spline.xmin))
            self.uniform_pixel = False
        else:
            self.max_shape = (2048, 2048)
            self.pixel1 = 50e-6
            self.pixel2 = 50e-6
        self.shape = self.max_shape

    def calc_mask(self):
        """
        Returns a generic mask for Frelon detectors...
        All pixels which (center) turns to be out of the valid region are by default discarded
        """
        if not self._splineFile:
            return
        d1 = numpy.outer(numpy.arange(self.shape[0]), numpy.ones(self.shape[1])) + 0.5
        d2 = numpy.outer(numpy.ones(self.shape[0]), numpy.arange(self.shape[1])) + 0.5
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
    aliases = ["aca1300"]
    MAX_SHAPE = (966, 1296)
    def __init__(self, pixel=3.75e-6):
        super(Basler, self).__init__(pixel1=pixel, pixel2=pixel)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)


class Mar345(Detector):

    """
    Mar345 Imaging plate detector

    In this detector, pixels are always square
    The valid image size are 2300, 2000, 1600, 1200, 3450, 3000, 2400, 1800
    """
    force_pixel = True
    MAX_SHAPE = (3450, 3450)
    # Valid image width with corresponding pixel size
    VALID_SIZE = {2300:150e-6,
                  2000:150e-6,
                  1600:150e-6,
                  1200:150e-6,
                  3450:100e-6,
                  3000:100e-6,
                  2400:100e-6,
                  1800:100e-6}

    aliases = ["MAR 345", "Mar3450"]
    def __init__(self, pixel1=100e-6, pixel2=100e-6):
        Detector.__init__(self, pixel1, pixel2)
        self.max_shape = (int(self.MAX_SHAPE[0] * 100e-6 / self.pixel1),
                          int(self.MAX_SHAPE[1] * 100e-6 / self.pixel2))
        self.shape = self.max_shape
#        self.mode = 1

    def calc_mask(self):
        c = [i // 2 for i in self.shape]
        x, y = numpy.ogrid[:self.shape[0], :self.shape[1]]
        mask = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2) > (c[0]) ** 2
        return mask

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def guess_binning(self, data):
        """
        Guess the binning/mode depending on the image shape
        @param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        """
        if "shape" in dir(data):
            shape = data.shape
        else:
            shape = data[:2]

        dim1, dim2 = shape
        self._pixel1 = self.VALID_SIZE[dim1]
        self._pixel2 = self.VALID_SIZE[dim1]
        self.max_shape = shape
        self.shape = shape
        self._binning = (1, 1)
        self._mask = False
        self._mask_crc = None


class ImXPadS10(Detector):
    """
    ImXPad detector: ImXPad s10 detector with 1x1modules
    """
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    MAX_SHAPE = (120, 80)  # max size of the detector
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_SIZE_RELATIVE = 2.5
    force_pixel = True
    aliases = ["Imxpad S10"]
    uniform_pixel = False

    @classmethod
    def _calc_pixels_size(cls, length, module_size, pixel_size):
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
            size[i * module_size - 1] = cls.BORDER_SIZE_RELATIVE
            size[i * module_size] = cls.BORDER_SIZE_RELATIVE
        # outer pixels have the normal size
#         size[0] = 1.0
#         size[-1] = 1.0
        return pixel_size * size

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)
        self._pixel_edges = None  # array of size max_shape+1: pixels are contiguous

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self.pixel1, self.pixel2)

    def calc_pixels_edges(self):
        """
        Calculate the position of the pixel edges
        """
        if self._pixel_edges is None:
            pixel_size1 = self._calc_pixels_size(self.MAX_SHAPE[0], self.MODULE_SIZE[0], self.PIXEL_SIZE[0])
            pixel_size2 = self._calc_pixels_size(self.MAX_SHAPE[1], self.MODULE_SIZE[1], self.PIXEL_SIZE[1])
            pixel_edges1 = numpy.zeros(self.MAX_SHAPE[0] + 1)
            pixel_edges2 = numpy.zeros(self.MAX_SHAPE[1] + 1)
            pixel_edges1[1:] = numpy.cumsum(pixel_size1)
            pixel_edges2[1:] = numpy.cumsum(pixel_size2)
            self._pixel_edges = pixel_edges1, pixel_edges2
        return self._pixel_edges

    def calc_mask(self):
        """
        Calculate the mask
        """
        dims = []
        for dim in [0, 1]:
            pos = numpy.zeros(self.MAX_SHAPE[dim], dtype=numpy.int8)
            n = self.MAX_SHAPE[dim] // self.MODULE_SIZE[dim]
            for i in range(1, n):
                pos[i * self.MODULE_SIZE[dim] - 1] = 1
                pos[i * self.MODULE_SIZE[dim]] = 1
            pos[0] = 1
            pos[-1] = 1
            dims.append(pos)
        # This is just an "outer sum"
        dim1, dim2 = dims
        dim1.shape = -1, 1
        dim1.strides = dim1.strides[0], 0
        dim2.shape = 1, -1
        dim2.strides = 0, dim2.strides[-1]
        return (dim1 + dim2) > 0

    def get_pixel_corners(self):
        """
        Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)

        Precision float32 is ok: precision of 1µm for a detector size of 1m


        @return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    edges1, edges2 = self.calc_pixels_edges()
                    p1 = expand2d(edges1, self.shape[1] + 1, False)
                    p2 = expand2d(edges2, self.shape[0] + 1, True)
#                     p3 = None
                    self._pixel_corners = numpy.zeros((self.shape[0], self.shape[1], 4, 3), dtype=numpy.float32)
                    self._pixel_corners[:, :, 0, 1] = p1[:-1, :-1]
                    self._pixel_corners[:, :, 0, 2] = p2[:-1, :-1]
                    self._pixel_corners[:, :, 1, 1] = p1[1:, :-1]
                    self._pixel_corners[:, :, 1, 2] = p2[1:, :-1]
                    self._pixel_corners[:, :, 2, 1] = p1[1:, 1:]
                    self._pixel_corners[:, :, 2, 2] = p2[1:, 1:]
                    self._pixel_corners[:, :, 3, 1] = p1[:-1, 1:]
                    self._pixel_corners[:, :, 3, 2] = p2[:-1, 1:]
#                     if p3 is not None:
#                         # non flat detector
#                         self._pixel_corners[:, :, 0, 0] = p3[:-1, :-1]
#                         self._pixel_corners[:, :, 1, 0] = p3[1:, :-1]
#                         self._pixel_corners[:, :, 2, 0] = p3[1:, 1:]
#                         self._pixel_corners[:, :, 3, 0] = p3[:-1, 1:]
        return self._pixel_corners

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
        edges1, edges2 = self.calc_pixels_edges()

        if (d1 is None) or (d2 is None):
            # Take the center of each pixel
            d1 = 0.5 * (edges1[:-1] + edges1[1:])
            d2 = 0.5 * (edges2[:-1] + edges2[1:])
            p1 = numpy.outer(d1, numpy.ones(self.shape[1]))
            p2 = numpy.outer(numpy.ones(self.shape[0]), d2)
        else:
            p1 = numpy.interp(d1 + 0.5, numpy.arange(self.MAX_SHAPE[0] + 1), edges1, edges1[0], edges1[-1])
            p2 = numpy.interp(d2 + 0.5, numpy.arange(self.MAX_SHAPE[1] + 1), edges2, edges2[0], edges2[-1])
        return p1, p2, None


class ImXPadS70(ImXPadS10):
    """
    ImXPad detector: ImXPad s70 detector with 1x7modules
    """
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    MAX_SHAPE = (120, 560)  # max size of the detector
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_SIZE_RELATIVE = 2.5
    force_pixel = True
    aliases = ["Imxpad S70"]
    PIXEL_EDGES = None  # array of size max_shape+1: pixels are contiguous

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        ImXPadS10.__init__(self, pixel1=pixel1, pixel2=pixel2)


class ImXPadS140(ImXPadS10):
    """
    ImXPad detector: ImXPad s140 detector with 2x7modules
    """
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    MAX_SHAPE = (240, 560)  # max size of the detector
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_PIXEL_SIZE_RELATIVE = 2.5
    force_pixel = True
    aliases = ["Imxpad S140"]

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        ImXPadS10.__init__(self, pixel1=pixel1, pixel2=pixel2)


class Xpad_flat(ImXPadS10):
    """
    Xpad detector: generic description for
    ImXPad detector with 8x7modules
    """
    MODULE_GAP = (3.57e-3, 0)  # in meter
    force_pixel = True
    MAX_SHAPE = (960, 560)
    uniform_pixel = False
    aliases = ["Xpad S540 flat", "d5"]
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_PIXEL_SIZE_RELATIVE = 2.5

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        super(Xpad_flat, self).__init__(pixel1=pixel1, pixel2=pixel2)
        self._pixel_corners = None

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
                (self.name, self.pixel1, self.pixel2)

    def calc_pixels_edges(self):
        """
        Calculate the position of the pixel edges, specific to the S540, d5 detector
        """
        if self._pixel_edges is None:
            # all pixel have the same size along the vertical axis, some pixels are larger along the horizontal one
            pixel_size1 = numpy.ones(self.MAX_SHAPE[0]) * self.PIXEL_SIZE[0]
            pixel_size2 = self._calc_pixels_size(self.MAX_SHAPE[1], self.MODULE_SIZE[1], self.PIXEL_SIZE[1])
            pixel_edges1 = numpy.zeros(self.MAX_SHAPE[0] + 1)
            pixel_edges2 = numpy.zeros(self.MAX_SHAPE[1] + 1)
            pixel_edges1[1:] = numpy.cumsum(pixel_size1)
            pixel_edges2[1:] = numpy.cumsum(pixel_size2)
            self._pixel_edges = pixel_edges1, pixel_edges2
        return self._pixel_edges


    def calc_mask(self):
        """
        Returns a generic mask for Xpad detectors...
        discards the first line and raw form all modules:
        those are 2.5x bigger and often mis - behaving
        """
        if self.max_shape is None:
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


    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!
        Adapted to Nexus detector definition

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)
        @param center: retrieve the coordinate of the center of the pixel
        @parm use_cython: set to False to test Numpy implementation
        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if (d1 is None) or (d2 is None):
#            d1, d2 = numpy.ogrid[:self.shape[0], :self.shape[1]]
            d1 = numpy.outer(numpy.arange(self.shape[0]), numpy.ones(self.shape[1]))
            d2 = numpy.outer(numpy.ones(self.shape[0]), numpy.arange(self.shape[1]))
        corners = self.get_pixel_corners()
        if center:
            # note += would make an increment in place which is bad (segfault !)
            d1 = d1 + 0.5
            d2 = d2 + 0.5
        if bilinear and use_cython:
            p1, p2, p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners)
            p1.shape = d1.shape
            p2.shape = d2.shape
        else:
            i1 = d1.astype(int).clip(0, corners.shape[0] - 1)
            i2 = d2.astype(int).clip(0, corners.shape[1] - 1)
            delta1 = d1 - i1
            delta2 = d2 - i2
            pixels = corners[i1, i2]
            A1 = pixels[:, :, 0, 1]
            A2 = pixels[:, :, 0, 2]
            B1 = pixels[:, :, 1, 1]
            B2 = pixels[:, :, 1, 2]
            C1 = pixels[:, :, 2, 1]
            C2 = pixels[:, :, 2, 2]
            D1 = pixels[:, :, 3, 1]
            D2 = pixels[:, :, 3, 2]
            # points A and D are on the same dim1 (Y), they differ in dim2 (X)
            # points B and C are on the same dim1 (Y), they differ in dim2 (X)
            # points A and B are on the same dim2 (X), they differ in dim1
            # p2 = mean(A2,B2) + delta2 * (mean(C2,D2)-mean(A2,C2))
            p1 = A1 * (1.0 - delta1) * (1.0 - delta2) \
               + B1 * delta1 * (1.0 - delta2) \
               + C1 * delta1 * delta2 \
               + D1 * (1.0 - delta1) * delta2
            p2 = A2 * (1.0 - delta1) * (1.0 - delta2) \
               + B2 * delta1 * (1.0 - delta2) \
               + C2 * delta1 * delta2 \
               + D2 * (1.0 - delta1) * delta2
        return p1, p2, None

    def get_pixel_corners(self):
        """
        Calculate the position of the corner of the pixels

        @return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    pixel_size1 = self._calc_pixels_size(self.MAX_SHAPE[0], self.MODULE_SIZE[0], self.PIXEL_SIZE[0])
                    pixel_size2 = self._calc_pixels_size(self.MAX_SHAPE[1], self.MODULE_SIZE[1], self.PIXEL_SIZE[1])
                    # half pixel offset
                    pixel_center1 = pixel_size1 / 2.0  # half pixel offset
                    pixel_center2 = pixel_size2 / 2.0
                    # size of all preceeding pixels
                    pixel_center1[1:] += numpy.cumsum(pixel_size1[:-1])
                    pixel_center2[1:] += numpy.cumsum(pixel_size2[:-1])
                    # gaps
                    for i in range(self.MAX_SHAPE[0] // self.MODULE_SIZE[0]):
                        pixel_center1[i * self.MODULE_SIZE[0]:
                           (i + 1) * self.MODULE_SIZE[0]] += i * self.MODULE_GAP[0]
                    for i in range(self.MAX_SHAPE[1] // self.MODULE_SIZE[1]):
                        pixel_center2[i * self.MODULE_SIZE[1]:
                           (i + 1) * self.MODULE_SIZE[1]] += i * self.MODULE_GAP[1]

                    pixel_center1.shape = -1, 1
                    pixel_center1.strides = pixel_center1.strides[0], 0

                    pixel_center2.shape = 1, -1
                    pixel_center2.strides = 0, pixel_center2.strides[1]

                    pixel_size1.shape = -1, 1
                    pixel_size1.strides = pixel_size1.strides[0], 0

                    pixel_size2.shape = 1, -1
                    pixel_size2.strides = 0, pixel_size2.strides[1]

                    corners = numpy.zeros((self.shape[0], self.shape[1], 4, 3), dtype=numpy.float32)
                    corners[:, :, 0, 1] = pixel_center1 - pixel_size1 / 2.0
                    corners[:, :, 0, 2] = pixel_center2 - pixel_size2 / 2.0
                    corners[:, :, 1, 1] = pixel_center1 + pixel_size1 / 2.0
                    corners[:, :, 1, 2] = pixel_center2 - pixel_size2 / 2.0
                    corners[:, :, 2, 1] = pixel_center1 + pixel_size1 / 2.0
                    corners[:, :, 2, 2] = pixel_center2 + pixel_size2 / 2.0
                    corners[:, :, 3, 1] = pixel_center1 - pixel_size1 / 2.0
                    corners[:, :, 3, 2] = pixel_center2 + pixel_size2 / 2.0
                    self._pixel_corners = corners
        return self._pixel_corners


class Perkin(Detector):
    """
    Perkin detector

    """
    aliases = ["Perkin detector", "Perkin Elmer"]
    force_pixel = True
    MAX_SHAPE = (4096, 4096)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 200e-6

    def __init__(self, pixel1=200e-6, pixel2=200e-6):
        super(Perkin, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(2 * pixel1 / self.DEFAULT_PIXEL1), int(2 * pixel2 / self.DEFAULT_PIXEL2))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))
        else:
            self.shape = (2048, 2048)
            self._binning = (2, 2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)


class Rayonix(Detector):
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 32e-6}
    MAX_SHAPE = (4096 , 4096)

    def __init__(self, pixel1=32e-6, pixel2=32e-6):
        super(Rayonix, self).__init__(pixel1=pixel1, pixel2=pixel2)
        binning = [1, 1]
        for b, p in self.BINNED_PIXEL_SIZE.items():
            if p == pixel1:
                binning[0] = b
            if p == pixel2:
                binning[1] = b
        self._binning = tuple(binning)
        self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, binning))

    def get_binning(self):
        return self._binning

    def set_binning(self, bin_size=(1, 1)):
        """
        Set the "binning" of the detector,

        @param bin_size: set the binning of the detector
        @type bin_size: int or (int, int)
        """
        if "__len__" in dir(bin_size) and len(bin_size) >= 2:
            bin_size = int(round(float(bin_size[0]))), int(round(float(bin_size[1])))
        else:
            b = int(round(float(bin_size)))
            bin_size = (b, b)
        if bin_size != self._binning:
            if (bin_size[0] in self.BINNED_PIXEL_SIZE) and (bin_size[1] in self.BINNED_PIXEL_SIZE):
                self._pixel1 = self.BINNED_PIXEL_SIZE[bin_size[0]]
                self._pixel2 = self.BINNED_PIXEL_SIZE[bin_size[1]]
            else:
                logger.warning("Binning factor (%sx%s) is not an official value for Rayonix detectors" % (bin_size[0], bin_size[1]))
                self._pixel1 = self.BINNED_PIXEL_SIZE[1] / float(bin_size[0])
                self._pixel2 = self.BINNED_PIXEL_SIZE[1] / float(bin_size[1])
            self._binning = bin_size
            self.shape = (self.max_shape[0] // bin_size[0],
                          self.max_shape[1] // bin_size[1])
    binning = property(get_binning, set_binning)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def guess_binning(self, data):
        """
        Guess the binning/mode depending on the image shape
        @param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        """
        if "shape" in dir(data):
            shape = data.shape
        else:
            shape = tuple(data[:2])
        bin1 = self.MAX_SHAPE[0] // shape[0]
        bin2 = self.MAX_SHAPE[1] // shape[1]
        self._binning = (bin1, bin2)
        self.shape = shape
        self.max_shape = shape
        self._pixel1 = self.BINNED_PIXEL_SIZE[bin1]
        self._pixel2 = self.BINNED_PIXEL_SIZE[bin2]
        self._mask = False
        self._mask_crc = None


class Rayonix133(Rayonix):
    """
    Rayonix 133 2D CCD detector detector also known as mar133

    Personnal communication from M. Blum

    What should be the default binning factor for those cameras ?

    Circular detector
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 32e-6,
                         2: 64e-6,
                         4: 128e-6,
                         8: 256e-6,
                         }
    MAX_SHAPE = (4096 , 4096)
    aliases = ["MAR133"]

    def __init__(self, pixel1=64e-6, pixel2=64e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        """Circular mask"""
        c = [i // 2 for i in self.shape]
        x, y = numpy.ogrid[:self.shape[0], :self.shape[1]]
        mask = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2) > (c[0]) ** 2
        return mask


class RayonixSx165(Rayonix):
    """
    Rayonix sx165 2d Detector also known as MAR165.

    Circular detector
    """
    BINNED_PIXEL_SIZE = {1: 39.5e-6,
                         2: 79e-6,
                         3: 118.616e-6,  # image shape is then 1364 not 1365 !
                         4: 158e-6,
                         8: 316e-6,
                         }
    MAX_SHAPE = (4096 , 4096)
    aliases = ["MAR165", "Rayonix Sx165"]
    force_pixel = True

    def __init__(self, pixel1=39.5e-6, pixel2=39.5e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        """Circular mask"""
        c = [i // 2 for i in self.shape]
        x, y = numpy.ogrid[:self.shape[0], :self.shape[1]]
        mask = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2) > (c[0]) ** 2
        return mask


class RayonixSx200(Rayonix):
    """
    Rayonix sx200 2d CCD Detector.

    Pixel size are personnal communication from M. Blum.
    """
    BINNED_PIXEL_SIZE = {1: 48e-6,
                         2: 96e-6,
                         3: 144e-6,
                         4: 192e-6,
                         8: 384e-6,
                         }
    MAX_SHAPE = (4096 , 4096)
    aliases = ["Rayonix sx200"]

    def __init__(self, pixel1=48e-6, pixel2=48e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixLx170(Rayonix):
    """
    Rayonix lx170 2d CCD Detector (2x1 CCDs).

    Nota: this is the same for lx170hs
    """
    BINNED_PIXEL_SIZE = {1:  44.2708e-6,
                         2:  88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10:442.7083e-6
                         }
    MAX_SHAPE = (1920, 3840)
    force_pixel = True
    aliases = ["Rayonix lx170"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx170(Rayonix):
    """
    Rayonix mx170 2d CCD Detector (2x2 CCDs).

    Nota: this is the same for mx170hs
    """
    BINNED_PIXEL_SIZE = {1:  44.2708e-6,
                         2:  88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10:442.7083e-6
                         }
    MAX_SHAPE = (3840, 3840)
    aliases = ["Rayonix mx170"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixLx255(Rayonix):
    """
    Rayonix lx255 2d Detector (3x1 CCDs)

    Nota: this detector is also called lx255hs
    """
    BINNED_PIXEL_SIZE = {1:  44.2708e-6,
                         2:  88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10:442.7083e-6
                         }
    MAX_SHAPE = (1920 , 5760)
    aliases = [ "Rayonix lx225"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx225(Rayonix):
    """
    Rayonix mx225 2D CCD detector detector

    Nota: this is the same definition for mx225he
    Personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1:  36.621e-6,
                         2:  73.242e-6,
                         3: 109.971e-6,
                         4: 146.484e-6,
                         8: 292.969e-6
                         }
    MAX_SHAPE = (6144, 6144)
    aliases = ["Rayonix mx225"]

    def __init__(self, pixel1=73.242e-6, pixel2=73.242e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx225hs(Rayonix):
    """
    Rayonix mx225hs 2D CCD detector detector

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1:  39.0625e-6,
                         2:  78.125e-6,
                         3: 117.1875e-6,
                         4: 156.25e-6,
                         5: 195.3125e-6,
                         6: 234.3750e-6,
                         8: 312.5e-6,
                         10:390.625e-6,
                         }
    MAX_SHAPE = (5760 , 5760)
    aliases = ["Rayonix mx225hs"]
    def __init__(self, pixel1=78.125e-6, pixel2=78.125e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx300(Rayonix):
    """
    Rayonix mx300 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1:  36.621e-6,
                         2:  73.242e-6,
                         3: 109.971e-6,
                         4: 146.484e-6,
                         8: 292.969e-6
                         }
    MAX_SHAPE = (8192, 8192)
    aliases = ["Rayonix mx300"]

    def __init__(self, pixel1=73.242e-6, pixel2=73.242e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx300hs(Rayonix):
    """
    Rayonix mx300hs 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1:   39.0625e-6,
                         2:   78.125e-6,
                         3:  117.1875e-6,
                         4:  156.25e-6,
                         5:  195.3125e-6,
                         6:  234.3750e-6,
                         8:  312.5e-6,
                         10: 390.625e-6
                         }
    MAX_SHAPE = (7680, 7680)
    aliases = ["Rayonix mx300hs"]

    def __init__(self, pixel1=78.125e-6, pixel2=78.125e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx340hs(Rayonix):
    """
    Rayonix mx340hs 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1:   44.2708e-6,
                         2:   88.5417e-6,
                         3:  132.8125e-6,
                         4:  177.0833e-6,
                         5:  221.3542e-6,
                         6:  265.625e-6,
                         8:  354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (7680 , 7680)
    aliases = ["Rayonix mx340hs"]

    def __init__(self, pixel1=88.5417e-6, pixel2=88.5417e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixSx30hs(Rayonix):
    """
    Rayonix sx30hs 2D CCD camera (1 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1:  15.625e-6,
                         2:  31.25e-6,
                         3:  46.875e-6,
                         4:  62.5e-6,
                         5:  78.125e-6,
                         6:  93.75e-6,
                         8: 125.0e-6,
                         10:156.25e-6
                         }
    MAX_SHAPE = (1920 , 1920)
    aliases = ["Rayonix Sx30hs"]

    def __init__(self, pixel1=15.625e-6, pixel2=15.625e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixSx85hs(Rayonix):
    """
    Rayonix sx85hs 2D CCD camera (1 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1:   44.2708e-6,
                         2:   88.5417e-6,
                         3:   132.8125e-6,
                         4:   177.0833e-6,
                         5:   221.3542e-6,
                         6:   265.625e-6,
                         8:   354.1667e-6,
                         10:  442.7083e-6
                         }
    MAX_SHAPE = (1920 , 1920)
    aliases = ["Rayonix Sx85hs"]
    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx425hs(Rayonix):
    """
    Rayonix mx425hs 2D CCD camera (5x5 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1:   44.2708e-6,
                         2:   88.5417e-6,
                         3:   132.8125e-6,
                         4:   177.0833e-6,
                         5:   221.3542e-6,
                         6:   265.625e-6,
                         8:   354.1667e-6,
                         10:  442.7083e-6
                         }
    MAX_SHAPE = (9600 , 9600)
    aliases = ["Rayonix mx425hs"]
    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx325(Rayonix):
    """
    Rayonix mx325 and mx325he 2D detector (4x4 CCD chips)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1:  39.673e-6,
                         2:  79.346e-6,
                         3: 119.135e-6,
                         4: 158.691e-6,
                         8: 317.383e-6
                         }
    MAX_SHAPE = (8192 , 8192)
    aliases = ["Rayonix mx325"]
    def __init__(self, pixel1=79.346e-6, pixel2=79.346e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)

class Aarhus(Detector):
    """
    Cylindrical detector made of a bent imaging-plate.
    Developped at the Danish university of Aarhus
    r = 1.2m or 0.3m

    The image has to be laid-out horizontally

    Nota: the detector is bending towards the sample, hence reducing the sample-detector distance.
    This is why z<0 (or p3<0)

    TODO: update cython code for 3d detectors
    use expand2d instead of outer product with ones
    """
    MAX_SHAPE = (1000 , 16000)
    IS_FLAT = False
    def __init__(self, pixel1=25e-6, pixel2=25e-6, radius=0.3):
        Detector.__init__(self, pixel1, pixel2)
        self.radius = radius
        self._pixel_corners = None


    def get_pixel_corners(self, use_cython=True):
        """
        Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)

        @return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    p1 = numpy.arange(self.shape[0] + 1.0) * self._pixel1
                    t2 = numpy.arange(self.shape[1] + 1.0) * (self._pixel2 / self.radius)
                    p2 = self.radius * numpy.sin(t2)
                    p3 = self.radius * (numpy.cos(t2) - 1.0)
                    if bilinear and use_cython:
                        d1 = expand2d(p1, self.shape[1] + 1, False)
                        d2 = expand2d(p2, self.shape[0] + 1, True)
                        d3 = expand2d(p3, self.shape[0] + 1, True)

                        corners = bilinear.convert_corner_2D_to_4D(3, d1, d2, d3)
                    else:
                        p1.shape = -1, 1
                        p1.strides = p1.strides[0], 0
                        p2.shape = 1, -1
                        p2.strides = 0, p2.strides[1]
                        p3.shape = 1, -1
                        p3.strides = 0, p3.strides[1]
                        corners = numpy.zeros((self.shape[0], self.shape[1], 4, 3), dtype=numpy.float32)
                        corners[:, :, 0, 0] = p3[:, :-1]
                        corners[:, :, 0, 1] = p1[:-1, :]
                        corners[:, :, 0, 2] = p2[:, :-1]
                        corners[:, :, 1, 0] = p3[:, :-1]
                        corners[:, :, 1, 1] = p1[1:, :]
                        corners[:, :, 1, 2] = p2[:, :-1]
                        corners[:, :, 2, 1] = p1[1:, :]
                        corners[:, :, 2, 2] = p2[:, 1:]
                        corners[:, :, 2, 0] = p3[:, 1:]
                        corners[:, :, 3, 0] = p3[:, 1:]
                        corners[:, :, 3, 1] = p1[:-1, :]
                        corners[:, :, 3, 2] = p2[:, 1:]
                    self._pixel_corners = corners
        return self._pixel_corners

    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!
        Adapted to Nexus detector definition

        @param d1: the Y pixel positions (slow dimension)
        @type d1: ndarray (1D or 2D)
        @param d2: the X pixel positions (fast dimension)
        @type d2: ndarray (1D or 2D)
        @param center: retrieve the coordinate of the center of the pixel
        @param use_cython: set to False to test Python implementeation
        @return: position in meter of the center of each pixels.
        @rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if (d1 is None) or d2 is None:
            d1 = expand2d(numpy.arange(float(self.shape[0])), self.shape[1], False)
            d2 = expand2d(numpy.arange(float(self.shape[1])), self.shape[0], True)
        corners = self.get_pixel_corners()
        if center:
            # avoid += It modifies in place and segfaults
            d1 = d1 + 0.5
            d2 = d2 + 0.5
        if bilinear and use_cython:
            p1, p2, p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners, is_flat=False)
            p1.shape = d1.shape
            p2.shape = d2.shape
            p3.shape = d2.shape
        else:
#         if True:
            i1 = d1.astype(int).clip(0, corners.shape[0] - 1)
            i2 = d2.astype(int).clip(0, corners.shape[1] - 1)
            delta1 = d1 - i1
            delta2 = d2 - i2
            pixels = corners[i1, i2]
            if pixels.ndim == 3:
                A0 = pixels[:, 0, 0]
                A1 = pixels[:, 0, 1]
                A2 = pixels[:, 0, 2]
                B0 = pixels[:, 1, 0]
                B1 = pixels[:, 1, 1]
                B2 = pixels[:, 1, 2]
                C0 = pixels[:, 2, 0]
                C1 = pixels[:, 2, 1]
                C2 = pixels[:, 2, 2]
                D0 = pixels[:, 3, 0]
                D1 = pixels[:, 3, 1]
                D2 = pixels[:, 3, 2]
            else:
                A0 = pixels[:, :, 0, 0]
                A1 = pixels[:, :, 0, 1]
                A2 = pixels[:, :, 0, 2]
                B0 = pixels[:, :, 1, 0]
                B1 = pixels[:, :, 1, 1]
                B2 = pixels[:, :, 1, 2]
                C0 = pixels[:, :, 2, 0]
                C1 = pixels[:, :, 2, 1]
                C2 = pixels[:, :, 2, 2]
                D0 = pixels[:, :, 3, 0]
                D1 = pixels[:, :, 3, 1]
                D2 = pixels[:, :, 3, 2]

            # points A and D are on the same dim1 (Y), they differ in dim2 (X)
            # points B and C are on the same dim1 (Y), they differ in dim2 (X)
            # points A and B are on the same dim2 (X), they differ in dim1 (Y)
            # points C and D are on the same dim2 (X), they differ in dim1 (Y)

            p1 = A1 * (1.0 - delta1) * (1.0 - delta2) \
               + B1 * delta1 * (1.0 - delta2) \
               + C1 * delta1 * delta2 \
               + D1 * (1.0 - delta1) * delta2
            p2 = A2 * (1.0 - delta1) * (1.0 - delta2) \
               + B2 * delta1 * (1.0 - delta2) \
               + C2 * delta1 * delta2 \
               + D2 * (1.0 - delta1) * delta2
            p3 = A0 * (1.0 - delta1) * (1.0 - delta2) \
               + B0 * delta1 * (1.0 - delta2) \
               + C0 * delta1 * delta2 \
               + D0 * (1.0 - delta1) * delta2
        return p1, p2, p3



ALL_DETECTORS = Detector.registry
detector_factory = Detector.factory
load = NexusDetector.sload


