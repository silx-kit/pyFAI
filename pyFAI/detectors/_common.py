# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
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
#

"""Description of all detectors with a factory to instantiate them"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/12/2018"
__status__ = "stable"


import logging
import numpy
import os
import posixpath
import threading
from collections import OrderedDict
import json

from .. import io
from .. import spline
from .. import utils
from .. import average
from ..utils import binning, expand2d, crc32
from ..third_party.six import with_metaclass
from ..utils.decorators import deprecated

logger = logging.getLogger(__name__)

try:
    import fabio
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    fabio = None
try:
    from ..ext import bilinear
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    bilinear = None


EPSILON = 1e-6


class DetectorMeta(type):
    """
    Metaclass used to register all detector classes inheriting from Detector
    """
    # we use __init__ rather than __new__ here because we want
    # to modify attributes of the class *after* they have been
    # created
    def __init__(cls, name, bases, dct):
        # "Detector" is a bit peculiar: while abstract it may be needed by the GUI, so adding it to the repository
        if name.startswith("_"):
            # It's not a public class
            return

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
    MANUFACTURER = None

    force_pixel = False  # Used to specify pixel size should be defined by the class itself.
    aliases = []  # list of alternative names
    registry = {}  # list of  detectors ...
    uniform_pixel = True  # tells all pixels have the same size
    IS_FLAT = True  # this detector is flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents, speeds-up calculation
    API_VERSION = "1.0"

    HAVE_TAPER = False
    """If true a spline file is mandatory to correct the geometry"""

    @classmethod
    def factory(cls, name, config=None):
        """
        A kind of factory...

        :param name: name of a detector
        :type name: str
        :param config: configuration of the detector
        :type config: dict or JSON representation of it.

        :return: an instance of the right detector, set-up if possible
        :rtype: pyFAI.detectors.Detector
        """
        if isinstance(name, Detector):
            # It's already a detector
            return name

        if os.path.isfile(name):
            # It's a filename
            return NexusDetector(name)

        # Search for the detector class
        import pyFAI.detectors
        detectorClass = None
        if hasattr(pyFAI.detectors, name):
            # It's a classname
            cls = getattr(pyFAI.detectors, name)
            if issubclass(cls, pyFAI.detectors.Detector):
                # Avoid code injection
                detectorClass = cls

        if detectorClass is None:
            # Search the name using the name database
            name = name.lower()
            names = [name, name.replace(" ", "_")]
            for name in names:
                if name in cls.registry:
                    detectorClass = cls.registry[name]
                    break

        if detectorClass is None:
            msg = ("Detector %s is unknown !, "
                   "please check if the filename exists or select one from %s" % (name, cls.registry.keys()))
            logger.error(msg)
            raise RuntimeError(msg)

        # Create the detector
        detector = None
        if config is not None:
            if not isinstance(config, dict):
                try:
                    config = json.loads(config)
                except Exception as err:  # IGNORE:W0703:
                    logger.error("Unable to parse config %s with JSON: %s, %s",
                                 name, config, err)
                    raise err
            try:
                detector = detectorClass(**config)
            except Exception as err:  # IGNORE:W0703:
                logger.error("Unable to configure detector %s with config: %s\n %s",
                             name, config, err)
                raise err
        else:
            detector = detectorClass()

        return detector

    def __init__(self, pixel1=None, pixel2=None, splineFile=None, max_shape=None):
        """
        :param pixel1: size of the pixel in meter along the slow dimension (often Y)
        :type pixel1: float
        :param pixel2: size of the pixel in meter along the fast dimension (often X)
        :type pixel2: float
        :param splineFile: path to file containing the geometric correction.
        :type splineFile: str
        :param max_shape: maximum size of the detector
        :type max_shape: 2-tuple of integrers
        """
        self._pixel1 = None
        self._pixel2 = None
        self._pixel_corners = None

        if pixel1:
            self._pixel1 = float(pixel1)
        if pixel2:
            self._pixel2 = float(pixel2)
        if (max_shape is None) and ("MAX_SHAPE" in dir(self.__class__)):
            self.max_shape = tuple(self.MAX_SHAPE)
        else:
            self.max_shape = max_shape
        self.shape = self.max_shape
        self._binning = (1, 1)
        self._mask = False
        self._mask_crc = None
        self._maskfile = None
        self._splineFile = None
        self.spline = None
        self._dx = None
        self._dy = None
        self._flatfield = None
        self._flatfield_crc = None  # not saved as part of HDF5 structure
        self._darkcurrent = None
        self._darkcurrent_crc = None  # not saved as part of HDF5 structure
        self.flatfiles = None  # not saved as part of HDF5 structure
        self.darkfiles = None  # not saved as part of HDF5 structure

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
        ":return: a shallow copy of itself"
        unmutable = ['_pixel1', '_pixel2', 'max_shape', 'shape', '_binning',
                     '_mask_crc', '_maskfile', "_splineFile", "_flatfield_crc",
                     "_darkcurrent_crc", "flatfiles", "darkfiles"]
        mutable = ['_mask', '_dx', '_dy', '_flatfield', "_darkcurrent"]
        new = self.__class__()
        for key in unmutable + mutable:
            new.__setattr__(key, self.__getattribute__(key))
        if self._splineFile:
            new.set_splineFile(self._splineFile)
        return new

    def __deepcopy__(self, memo=None):
        ":return: a deep copy of itself"
        unmutable = ['_pixel1', '_pixel2', 'max_shape', 'shape', '_binning',
                     '_mask_crc', '_maskfile', "_splineFile", "_flatfield_crc",
                     "_darkcurrent_crc", "flatfiles", "darkfiles"]
        mutable = ['_mask', '_dx', '_dy', '_flatfield', "_darkcurrent"]
        if memo is None:
            memo = {}
        new = self.__class__()
        memo[id(self)] = new
        for key in unmutable:
            old = self.__getattribute__(key)
            memo[id(old)] = old
            new.__setattr__(key, old)
        for key in mutable:
            value = self.__getattribute__(key)
            if (value is None) or (value is False):
                new_value = value
            elif "copy" in dir(value):
                new_value = value.copy()
            else:
                new_value = 1 * value
            memo[id(value)] = new_value
            new.__setattr__(key, new_value)
        if self._splineFile:
            new.set_splineFile(self._splineFile)
        return new

    def __eq__(self, other):
        """Equality checker for detector, used in tests

        Checks for pixel1, pixel2, binning, shape, max_shape.
        """
        if other is None:
            return False
        res = True
        for what in ["pixel1", "pixel2", "binning", "shape", "max_shape"]:
            res &= getattr(self, what) == getattr(other, what)
        return res

    def set_config(self, config):
        """
        Sets the configuration of the detector.

        The configuration is either a python dictionary or a JSON string or a
        file containing this JSON configuration

        keys in that dictionary are:  pixel1, pixel2, splineFile, max_shape

        :param config: string or JSON-serialized dict
        :return: self
        """
        if not isinstance(config, dict):
            try:
                config = json.loads(config)
            except Exception as err:  # IGNORE:W0703:
                logger.error("Unable to parse config %s with JSON: %s, %s",
                             config, err)
                raise err
        if not self.force_pixel:
            pixel1 = config.get("pixel1")
            pixel2 = config.get("pixel2")
            if pixel1:
                self.set_pixel1(pixel1)
            if pixel2:
                self.set_pixel2(pixel2)
            if "splineFile" in config:
                self.set_splineFile(config.get("splineFile"))
            if "max_shape" in config:
                self.max_shape = config.get("max_shape")
        return self

    def get_config(self):
        """Return the configuration with arguments to the constructor

        Derivative classes should implement this method
        if they change the constructor!

        :return: dict with param for serialization
        """
        dico = OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2),
                            ('max_shape', self.max_shape)))
        if self._splineFile:
            dico["splineFile"] = self._splineFile
        return dico

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
            self.max_shape = self.spline.getDetectorSize()
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

        :param bin_size: binning as integer or tuple of integers.
        :type bin_size: (int, int)
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

        :return: representation of the detector easy to serialize
        :rtype: dict
        """
        dico = OrderedDict((("detector", self.name),
                            ("pixel1", self._pixel1),
                            ("pixel2", self._pixel2),
                            ('max_shape', self.max_shape)))
        if self._splineFile:
            dico["splineFile"] = self._splineFile

        return dico

    def getFit2D(self):
        """
        Helper method to serialize the description of a detector using the Fit2d units

        :return: representation of the detector easy to serialize
        :rtype: dict
        """
        return {"pixelX": self._pixel2 * 1e6,
                "pixelY": self._pixel1 * 1e6,
                "splineFile": self._splineFile}

    def setPyFAI(self, **kwarg):
        """
        Twin method of getPyFAI: setup a detector instance according to a description

        :param kwarg: dictionary containing detector, pixel1, pixel2 and splineFile

        """
        if "detector" in kwarg:
            import pyFAI.detectors
            config = {}
            for key in ("pixel1", "pixel2", 'max_shape', "splineFile"):
                if key in kwarg:
                    config[key] = kwarg[key]
            self = pyFAI.detectors.detector_factory(kwarg["detector"], config)
        return self

    @classmethod
    def from_dict(cls, dico):
        """Creates a brand new detector from the description of the detector as
        a dict

        :param dico: JSON serializable dictionary
        :return: Detector instance
        """
        return cls.factory(dico.get("detector"), dico)

    def setFit2D(self, **kwarg):
        """
        Twin method of getFit2D: setup a detector instance according to a description

        :param kwarg: dictionary containing pixel1, pixel2 and splineFile

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

        :param d1: the Y pixel positions (slow dimension)
        :type d1: ndarray (1D or 2D)
        :param d2: the X pixel positions (fast dimension)
        :type d2: ndarray (1D or 2D)
        :param center: retrieve the coordinate of the center of the pixel, unless gives one corner
        :param use_cython: set to False to test Python implementation
        :return: position in meter of the center of each pixels.
        :rtype: 3xndarray, the later being None if IS_FLAT

        d1 and d2 must have the same shape, returned array will have
        the same shape.

        pos_z is None for flat detectors
        """
        if self.shape:
            if (d1 is None) or (d2 is None):
                d1 = expand2d(numpy.arange(self.shape[0]).astype(numpy.float32), self.shape[1], False)
                d2 = expand2d(numpy.arange(self.shape[1]).astype(numpy.float32), self.shape[0], True)

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
                A1 = pixels[..., 0, 1]
                A2 = pixels[..., 0, 2]
                B1 = pixels[..., 1, 1]
                B2 = pixels[..., 1, 2]
                C1 = pixels[..., 2, 1]
                C2 = pixels[..., 2, 2]
                D1 = pixels[..., 3, 1]
                D2 = pixels[..., 3, 2]
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
                    A0 = pixels[..., 0, 0]
                    B0 = pixels[..., 1, 0]
                    C0 = pixels[..., 2, 0]
                    D0 = pixels[..., 3, 0]
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

    def get_pixel_corners(self):
        """Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)

        Precision float32 is ok: precision of 1µm for a detector size of 1m

        :return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    # like numpy.ogrid
                    d1 = expand2d(numpy.arange(self.shape[0] + 1.0), self.shape[1] + 1, False)
                    d2 = expand2d(numpy.arange(self.shape[1] + 1.0), self.shape[0] + 1, True)
                    p1, p2, p3 = self.calc_cartesian_positions(d1, d2, center=False)
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

    def set_pixel_corners(self, ary):
        """Sets the position of pixel corners with some additional validation

        :param ary: This a 4D array which contains: number of lines,
                                                    number of columns,
                                                    corner index,
                                                    position in space Z, Y, X
        """
        if ary is None:
            # Leave as it is ... just reset the array
            self._pixel_corners = None
        else:
            ary = numpy.ascontiguousarray(ary, dtype=numpy.float32)
            # Validation for the array
            assert ary.ndim == 4
            assert ary.shape[3] == 3  # 3 coordinates in Z Y X
            assert ary.shape[2] >= 3  # at least 3 corners per pixel

            z = ary[..., 0]
            is_flat = (z.max() == z.min() == 0.0)
            with self._sem:
                self.IS_CONTIGUOUS = False
                self.IS_FLAT = is_flat
                self.uniform_pixel = False  # This enforces the usage of pixel_corners
                self._pixel_corners = ary

    def save(self, filename):
        """
        Saves the detector description into a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html
        Main differences:

            * differentiate pixel center from pixel corner offsets
            * store all offsets are ndarray according to slow/fast dimension (not x, y)

        :param filename: name of the file on the disc
        """
        if not io.h5py:
            logger.error("h5py module missing: NeXus detectors not supported")
            raise RuntimeError("H5py module is missing")

        with io.Nexus(filename, "+") as nxs:
            det_grp = nxs.new_detector(name=self.name.replace(" ", "_"))
            det_grp["API_VERSION"] = numpy.string_(self.API_VERSION)
            det_grp["IS_FLAT"] = self.IS_FLAT
            det_grp["IS_CONTIGUOUS"] = self.IS_CONTIGUOUS
            det_grp["pixel_size"] = numpy.array([self.pixel1, self.pixel2], dtype=numpy.float32)
            det_grp["force_pixel"] = self.force_pixel
            det_grp["force_pixel"].attrs["info"] = "The detector class specifies the pixel size"
            if self.max_shape is not None:
                det_grp["max_shape"] = numpy.array(self.max_shape, dtype=numpy.int32)
            if self.shape is not None:
                det_grp["shape"] = numpy.array(self.shape, dtype=numpy.int32)
            if self.binning is not None:
                det_grp["binning"] = numpy.array(self._binning, dtype=numpy.int32)
            if self.flatfield is not None:
                dset = det_grp.create_dataset("flatfield", data=self.flatfield,
                                              compression="gzip", compression_opts=9, shuffle=True)
                dset.attrs["interpretation"] = "image"
            if self.darkcurrent is not None:
                dset = det_grp.create_dataset("darkcurrent", data=self.darkcurrent,
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
        """Guess the binning/mode depending on the image shape

        If the binning changes, this enforces the reset of the mask.

        :param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        :return: True if the data fit the detector
        :rtype: bool
        """
        if hasattr(data, "shape"):
            shape = data.shape
        elif hasattr(data, "__len__"):
            shape = tuple(data[:2])
        else:
            logger.warning("No shape available to guess the binning: %s", data)
            self._binning = 1, 1
            return False

        if shape == self.shape:
            return True

        if not self.force_pixel:
            if shape != self.max_shape:
                logger.warning("guess_binning is not implemented for %s detectors!\
                 and image size %s is wrong, expected %s!" % (self.name, shape, self.shape))
                return False
            self._binning = 1, 1
            return True
        elif self.max_shape:
            bin1 = self.max_shape[0] // shape[0]
            bin2 = self.max_shape[1] // shape[1]
            if bin1 == 0 or bin2 == 0:
                # cancel
                logger.warning("Impossible binning: image bigger than the detector")
                return False
            res = self.max_shape[0] % shape[0] + self.max_shape[1] % shape[1]
            if res != 0:
                logger.warning("Impossible binning: max_shape is %s, requested shape %s", self.max_shape, shape)

            old_binning = self._binning
            self._binning = (bin1, bin2)
            self.shape = shape
            self._pixel1 *= (1.0 * bin1 / old_binning[0])
            self._pixel2 *= (1.0 * bin2 / old_binning[1])
            self._mask = False
            self._mask_crc = None
            return res == 0
        else:
            logger.debug("guess_binning for generic detectors !")
            self._binning = 1, 1
            return False

    def calc_mask(self):
        """Method calculating the mask for a given detector

        Detectors with gaps should overwrite this method with
        something actually calculating the mask!

        :return: the mask with valid pixel to 0
        :rtype: numpy ndarray of int8 or None
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
                        self._mask = numpy.ascontiguousarray(self._mask, numpy.int8)
                        self._mask_crc = crc32(self._mask)
        return self._mask

    def get_mask_crc(self):
        return self._mask_crc

    def set_mask(self, mask):
        with self._sem:
            if mask is None:
                self._mask = self._mask_crc = None
            else:
                mask = numpy.ascontiguousarray(mask, numpy.int8)
                # Mind the order: guess_binning deletes the mask
                self.guess_binning(mask)
                self._mask = mask
                self._mask_crc = crc32(self._mask)

    mask = property(get_mask, set_mask)

    def set_maskfile(self, maskfile):
        if fabio:
            mask = numpy.ascontiguousarray(fabio.open(maskfile).data,
                                           dtype=numpy.int8)
            self.set_mask(mask)
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
            if self.force_pixel and (err > EPSILON):
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
            if self.force_pixel and (err > EPSILON):
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

    def get_flatfield(self):
        return self._flatfield

    def get_flatfield_crc(self):
        return self._flatfield_crc

    def set_flatfield(self, flat):
        self._flatfield = flat
        self._flatfield_crc = crc32(flat) if flat is not None else None

    flatfield = property(get_flatfield, set_flatfield)

    @deprecated(reason="Not maintained", since_version="0.17")
    def set_flatfiles(self, files, method="mean"):
        """
        :param files: file(s) used to compute the flat-field.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str

        Set the flat field from one or mutliple files, averaged
        according to the method provided
        """
        if type(files) in utils.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_flatfield(None)
        elif len(files) == 1:
            if fabio is None:
                raise RuntimeError("FabIO is missing")
            self.set_flatfield(fabio.open(files[0]).data.astype(numpy.float32))
            self.flatfiles = files[0]
        else:
            self.set_flatfield(average.average_images(files, filter_=method, fformat=None, threshold=0))
            self.flatfiles = "%s(%s)" % (method, ",".join(files))

    def get_darkcurrent(self):
        return self._darkcurrent

    def get_darkcurrent_crc(self):
        return self._darkcurrent_crc

    def set_darkcurrent(self, dark):
        self._darkcurrent = dark
        self._darkcurrent_crc = crc32(dark) if dark is not None else None

    darkcurrent = property(get_darkcurrent, set_darkcurrent)

    @deprecated(reason="Not maintained", since_version="0.17")
    def set_darkfiles(self, files=None, method="mean"):
        """
        :param files: file(s) used to compute the dark.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str

        Set the dark current from one or mutliple files, avaraged
        according to the method provided
        """
        if type(files) in utils.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_darkcurrent(None)
        elif len(files) == 1:
            if fabio is None:
                raise RuntimeError("FabIO is missing")
            self.set_darkcurrent(fabio.open(files[0]).data.astype(numpy.float32))
            self.darkfiles = files[0]
        else:
            self.set_darkcurrent(average.average_images(files, filter_=method, fformat=None, threshold=0))
            self.darkfiles = "%s(%s)" % (method, ",".join(files))

    def __getnewargs_ex__(self):
        "Helper function for pickling detectors"
        return (self.pixel1, self.pixel2, self.splineFile, self.max_shape), {}

    def __getstate__(self):
        """Helper function for pickling detectors

        :return: the state of the object
        """
        state_blacklist = ('_sem',)
        state = self.__dict__.copy()
        for key in state_blacklist:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        """Helper function for unpickling detectors

        :param state: the state of the object
        """
        for statekey, statevalue in state.items():
            setattr(self, statekey, statevalue)
        self._sem = threading.Semaphore()


class NexusDetector(Detector):
    """
    Class representing a 2D detector loaded from a NeXus file
    """
    def __init__(self, filename=None):
        Detector.__init__(self)
        self.uniform_pixel = True
        self._filename = None
        if filename is not None:
            self.load(filename)

    def __repr__(self):
        return "%s detector from NeXus file: %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._filename, self._pixel1, self._pixel2)

    def load(self, filename):
        """
        Loads the detector description from a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html

        :param filename: name of the file on the disk
        """
        if not io.h5py:
            logger.error("h5py module missing: NeXus detectors not supported")
            raise RuntimeError("H5py module is missing")
        with io.Nexus(filename, "r") as nxs:
            det_grp = nxs.find_detector()
            if not det_grp:
                raise RuntimeError("No detector definition in this file %s" % filename)
            name = posixpath.split(det_grp.name)[-1]
            self.aliases = [name.replace("_", " "), det_grp.name]
            if "IS_FLAT" in det_grp:
                self.IS_FLAT = det_grp["IS_FLAT"].value
            if "IS_CONTIGUOUS" in det_grp:
                self.IS_CONTIGUOUS = det_grp["IS_CONTIGUOUS"].value
            if "flatfield" in det_grp:
                self.flatfield = det_grp["flatfield"].value
            if "darkcurrent" in det_grp:
                self.darkcurrent = det_grp["darkcurrent"].value
            if "force_pixel" in det_grp:
                self.force_pixel = det_grp["force_pixel"].value
            if "binning" in det_grp:
                self._binning = tuple(i for i in det_grp["binning"].value)
            if "pixel_size" in det_grp:
                self._pixel1, self._pixel2 = det_grp["pixel_size"].value
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
        # Populate shape and max_shape if needed
        if self.max_shape is None:
            if self.shape is None:
                if self.mask is not None:
                    self.shape = self.mask.shape
                elif self.darkcurrent is not None:
                    self.shape = self.darkcurrent.shape
                elif self.flatfield is not None:
                    self.shape = self.flatfield.shape
                else:
                    raise RuntimeError("Detector has no shape")
            if self._binning is None:
                self.max_shape = self.shape
            else:
                self.max_shape = tuple(i * j for i, j in zip(self.shape, self._binning))
        self._filename = filename

    def get_filename(self):
        """Returns the filename containing the description of this detector.

        :rtype: Enum[None|str]
        """
        return self._filename

    filename = property(get_filename)

    @classmethod
    def sload(cls, filename):
        """
        Instantiate the detector description from a NeXus file, adapted from:
        http://download.nexusformat.org/sphinx/classes/base_classes/NXdetector.html

        :param filename: name of the file on the disk
        :return: Detector instance
        """
        obj = cls()
        cls.load(filename)
        return obj

    def set_config(self, config):
        """set the config of the detector

        For Nexus detector, the only valid key is "filename"

        :param config: dict or JSON serialized dict
        :return: detector instance
        """
        if not isinstance(config, dict):
            try:
                config = json.loads(config)
            except Exception as err:  # IGNORE:W0703:
                logger.error("Unable to parse config %s with JSON: %s, %s",
                             config, err)
                raise err
        filename = config.get("filename")
        if os.path.exists(filename):
            self.load(filename)
        else:
            logger.error("Unable to configure Nexus detector, config: %s",
                         config)
        return self

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return {"filename": self._filename}

    def getPyFAI(self):
        """
        Helper method to serialize the description of a detector using the pyFAI way
        with everything in S.I units.

        :return: representation of the detector easy to serialize
        :rtype: dict
        """
        return {"detector": self._filename or self.name,
                "pixel1": self._pixel1,
                "pixel2": self._pixel2
                }

    def getFit2D(self):
        """
        Helper method to serialize the description of a detector using the Fit2d units

        :return: representation of the detector easy to serialize
        :rtype: dict
        """
        return {"pixelX": self._pixel2 * 1e6,
                "pixelY": self._pixel1 * 1e6
                }
