#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""
Description of the Pilatus detectors.
"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/10/2018"
__status__ = "production"


import numpy
import logging
logger = logging.getLogger(__name__)
import json
from collections import OrderedDict
from ._common import Detector
from pyFAI.utils import mathutil
try:
    from ..ext import bilinear
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    bilinear = None


class Fairchild(Detector):
    """
    Fairchild Condor 486:90 detector
    """
    MANUFACTURER = "Fairchild Semiconductor"

    force_pixel = True
    uniform_pixel = True
    aliases = ["Fairchild", "Condor", "Fairchild Condor 486:90"]
    MAX_SHAPE = (4096, 4096)

    def __init__(self, pixel1=15e-6, pixel2=15e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Titan(Detector):
    """
    Titan CCD detector from Agilent. Mask not handled
    """
    MANUFACTURER = ["Agilent", "Oxford Diffraction"]

    force_pixel = True
    MAX_SHAPE = (2048, 2048)
    aliases = ["Titan 2k x 2k", "Titan 2k x 2k", "OXD Titan", "Agilent Titan"]
    uniform_pixel = True

    def __init__(self, pixel1=60e-6, pixel2=60e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


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

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Basler(Detector):
    """
    Basler camera are simple CCD camara over GigaE

    """
    MANUFACTURER = "Basler"

    force_pixel = True
    aliases = ["aca1300"]
    MAX_SHAPE = (966, 1296)

    def __init__(self, pixel=3.75e-6):
        super(Basler, self).__init__(pixel1=pixel, pixel2=pixel)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return {"pixel": self._pixel1}

    def set_config(self, config):
        """Sets the configuration of the detector.

        The configuration is either a python dictionary or a JSON string or a
        file containing this JSON configuration

        keys in that dictionary are:  pixel

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
        pixel = config.get("pixel")
        if pixel:
            self.set_pixel1(pixel)
            self.set_pixel2(pixel)
        return self


class Perkin(Detector):
    """
    Perkin detector

    """
    MANUFACTURER = "Perkin Elmer"

    aliases = ["Perkin detector", "Perkin Elmer"]
    force_pixel = True
    MAX_SHAPE = (4096, 4096)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 200e-6

    def __init__(self, pixel1=200e-6, pixel2=200e-6):
        super(Perkin, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(2 * pixel1 / self.DEFAULT_PIXEL1), int(2 * pixel2 / self.DEFAULT_PIXEL2))
            self.shape = tuple(s // b for s, b in zip(self.max_shape, self._binning))
        else:
            self.shape = (2048, 2048)
            self._binning = (2, 2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Aarhus(Detector):
    """
    Cylindrical detector made of a bent imaging-plate.
    Developped at the Danish university of Aarhus
    r = 1.2m or 0.3m

    Credits:
    Private communication;
    B. B. Iversen,
    Center for Materials Crystallography & Dept. of Chemistry and iNANO,
    Aarhus University

    The image has to be laid-out horizontally

    Nota: the detector is bend towards the sample, hence reducing the sample-detector distance.
    This is why z<0 (or p3<0)

    TODO: update cython code for 3d detectors
    use expand2d instead of outer product with ones
    """
    MANUFACTURER = "Aarhus University"

    MAX_SHAPE = (1000, 16000)
    IS_FLAT = False
    force_pixel = True

    def __init__(self, pixel1=24.893e-6, pixel2=24.893e-6, radius=0.29989):
        Detector.__init__(self, pixel1, pixel2)
        self.radius = radius
        self._pixel_corners = None

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2),
                            ("radius", self.radius)))

    def set_config(self, config):
        """Sets the configuration of the detector.

        The configuration is either a python dictionary or a JSON string or a
        file containing this JSON configuration

        keys in that dictionary are:  pixel1, pixel2, radius

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
        pixel1 = config.get("pixel1")
        if pixel1:
            self.set_pixel1(pixel1)
        pixel2 = config.get("pixel2")
        if pixel2:
            self.set_pixel1(pixel2)
        radius = config.get("radius")
        if radius:
            self.radius = radius
            self._pixel_corners = None
        return self

    def get_pixel_corners(self, use_cython=True):
        """
        Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)

        :return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    p1 = (numpy.arange(self.shape[0] + 1.0) * self._pixel1).astype(numpy.float32)
                    t2 = numpy.arange(self.shape[1] + 1.0) * (self._pixel2 / self.radius)
                    p2 = (self.radius * numpy.sin(t2)).astype(numpy.float32)
                    p3 = (self.radius * (numpy.cos(t2) - 1.0)).astype(numpy.float32)
                    if bilinear and use_cython:
                        d1 = mathutil.expand2d(p1, self.shape[1] + 1, False)
                        d2 = mathutil.expand2d(p2, self.shape[0] + 1, True)
                        d3 = mathutil.expand2d(p3, self.shape[0] + 1, True)
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

        :param d1: the Y pixel positions (slow dimension)
        :type d1: ndarray (1D or 2D)
        :param d2: the X pixel positions (fast dimension)
        :type d2: ndarray (1D or 2D)
        :param center: retrieve the coordinate of the center of the pixel
        :param use_cython: set to False to test Python implementeation
        :return: position in meter of the center of each pixels.
        :rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if (d1 is None) or d2 is None:
            d1 = mathutil.expand2d(numpy.arange(self.shape[0]).astype(numpy.float32), self.shape[1], False)
            d2 = mathutil.expand2d(numpy.arange(self.shape[1]).astype(numpy.float32), self.shape[0], True)
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
            # points C and D are on the same dim2 (X), they differ in dim1 (
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
            # To ensure numerical consitency with cython procedure.
            p1 = p1.astype(numpy.float32)
            p2 = p2.astype(numpy.float32)
            p3 = p3.astype(numpy.float32)
        return p1, p2, p3


class Pixium(Detector):
    """PIXIUM 4700 detector

    High energy X ray diffraction using the Pixium 4700 flat panel detector
    J E Daniels, M Drakopoulos, et al.; Journal of Synchrotron Radiation 16(Pt 4):463-8 · August 2009
    """
    MANUFACTURER = "Thales Electronics"

    aliases = ["Pixium 4700", "Pixium 4700 detector", "Thales Electronics"]
    force_pixel = True
    MAX_SHAPE = (1910, 2480)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 154e-6

    def __init__(self, pixel1=308e-6, pixel2=308e-6):
        """Defaults to 2x2 binning
        """
        super(Pixium, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(round(pixel1 / self.DEFAULT_PIXEL1)),
                             int(round(pixel2 / self.DEFAULT_PIXEL2)))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Apex2(Detector):
    """BrukerApex2 detector

    Actually a derivative from the Fairchild detector with higher binning
    """
    MANUFACTURER = "Bruker"

    aliases = ["ApexII", "Bruker"]
    force_pixel = True
    MAX_SHAPE = (1024, 1024)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 60e-6

    def __init__(self, pixel1=120e-6, pixel2=120e-6):
        """Defaults to 2x2 binning
        """
        super(Apex2, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(round(pixel1 / self.DEFAULT_PIXEL1)),
                             int(round(pixel2 / self.DEFAULT_PIXEL2)))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class RaspberryPi5M(Detector):
    """5 Mpix detector from Raspberry Pi

    """
    aliases = ["Picam v1"]
    force_pixel = True
    MAX_SHAPE = (1944, 2592)

    def __init__(self, pixel1=1.4e-6, pixel2=1.4e-6):
        super(RaspberryPi5M, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class RaspberryPi8M(Detector):
    """8 Mpix detector from Raspberry Pi

    """
    aliases = ["Picam v2"]
    force_pixel = True
    MAX_SHAPE = (2464, 3280)

    def __init__(self, pixel1=1.12e-6, pixel2=1.12e-6):
        super(RaspberryPi8M, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))
