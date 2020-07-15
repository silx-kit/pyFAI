#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2020 European Synchrotron Radiation Facility, Grenoble, France
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
Description of detectors which are not flat.

Mainly cylindrical curved imaging-plates for now. 
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2020"
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


class CylindricalDetector(Detector):
    "Abstract base class for all cylindrical detecors"
    MANUFACTURER = None 
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

    def _get_compact_pixel_corners(self):
        "The core function which calculates the pixel corner coordinates"
        raise NotImplementedError("This is an abtract class")
    
    def get_pixel_corners(self, correct_binning=False, use_cython=True):
        """
        Calculate the position of the corner of the pixels

        This should be overwritten by class representing non-contiguous detector (Xpad, ...)
        :param correct_binning: If True, check that the produced array have the right shape regarding binning
        :param use_cython: set to False for testing
        :return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    p1, p2, p3 = self._get_compact_pixel_corners()
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
        if correct_binning and self._pixel_corners.shape[:2] != self.shape:
            return self._rebin_pixel_corners()
        else:
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
        
    
class Aarhus(CylindricalDetector):
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

    """
    MANUFACTURER = "Aarhus University"

    MAX_SHAPE = (1000, 16000)

    def __init__(self, pixel1=24.893e-6, pixel2=24.893e-6, radius=0.29989):
        CylindricalDetector.__init__(self, pixel1, pixel2, radius)
    
    def _get_compact_pixel_corners(self):
        "The core function which calculates the pixel corner coordinates"
        p1 = (numpy.arange(self.shape[0] + 1.0) * self._pixel1).astype(numpy.float32)
        t2 = numpy.arange(self.shape[1] + 1.0) * (self._pixel2 / self.radius)
        p2 = (self.radius * numpy.sin(t2)).astype(numpy.float32)
        p3 = (self.radius * (numpy.cos(t2) - 1.0)).astype(numpy.float32)
        return p1, p2, p3


class Rapid(CylindricalDetector):
    """
    Cylindrical detector: Rigaku R-axis RAPID II
    Unlike the Aarhus detector, the detectors is bent the other direction. 
    It covers 210°
    r = 127.26mm
    pixel size 100µm but can be binned 2x2

    Credits:
    Private communication;
    Dr. Jozef Bednarčík
    Department of Condensed Matter Physics
    Institute of Physics
    P.J. Šafárik University, Košice, Slovakia

    The image has to be laid-out horizontally

    Nota: the detector is bend towards the sample, hence reducing the sample-detector distance.
    This is why z<0 (or p3<0)
    """
    MANUFACTURER = "Rigaku"
    aliases = ["RapidII"]
    MAX_SHAPE = (2560, 4700)

    def __init__(self, pixel1=0.1e-3, pixel2=0.1e-3, radius=0.12726):
        CylindricalDetector.__init__(self, pixel1, pixel2, radius)

    def _get_compact_pixel_corners(self):
        "The core function which calculates the pixel corner coordinates"
        p1 = (numpy.arange(self.shape[0] + 1.0) * self._pixel1).astype(numpy.float32)
        t2 = numpy.arange(self.shape[1] + 1.0) * (self._pixel2 / self.radius)
        p2 = (self.radius * numpy.sin(t2)).astype(numpy.float32)
        p3 = (self.radius * (numpy.cos(t2) - 1.0)).astype(numpy.float32)
        return p1, p2, p3
        
