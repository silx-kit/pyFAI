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
Description of the `imXPAD <http://www.imxpad.com/>`_ detectors.
"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/01/2019"
__status__ = "production"

import functools
import json
import numpy
from collections import OrderedDict
from ._common import Detector
from pyFAI.utils import mathutil

import logging
logger = logging.getLogger(__name__)


try:
    from ..ext import bilinear
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    bilinear = None


class ImXPadS10(Detector):
    """
    ImXPad detector: ImXPad s10 detector with 1x1modules
    """
    MANUFACTURER = "ImXPad"

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

        :param length: the number of pixel to compute
        :type length: int
        :param module_size: the number of pixel of one module
        :type module_size: int
        :param pixel_size: the size of one pixels (meter per pixel)
        :type length: float

        :return: the coordinates of each pixels 0..length
        :rtype: ndarray
        """
        size = numpy.ones(length)
        n = length // module_size
        for i in range(1, n):
            size[i * module_size - 1] = cls.BORDER_SIZE_RELATIVE
            size[i * module_size] = cls.BORDER_SIZE_RELATIVE
        # outer pixels have the normal size
        # size[0] = 1.0
        # size[-1] = 1.0
        return pixel_size * size

    def __init__(self, pixel1=130e-6, pixel2=130e-6, max_shape=None, module_size=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
        self._pixel_edges = None  # array of size max_shape+1: pixels are contiguous
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self.pixel1, self.pixel2)

    def calc_pixels_edges(self):
        """
        Calculate the position of the pixel edges
        """
        if self._pixel_edges is None:
            pixel_size1 = self._calc_pixels_size(self.max_shape[0], self.module_size[0], self.PIXEL_SIZE[0])
            pixel_size2 = self._calc_pixels_size(self.max_shape[1], self.module_size[1], self.PIXEL_SIZE[1])
            pixel_edges1 = numpy.zeros(self.max_shape[0] + 1)
            pixel_edges2 = numpy.zeros(self.max_shape[1] + 1)
            pixel_edges1[1:] = numpy.cumsum(pixel_size1)
            pixel_edges2[1:] = numpy.cumsum(pixel_size2)
            self._pixel_edges = pixel_edges1, pixel_edges2
        return self._pixel_edges

    def calc_mask(self):
        """
        Calculate the mask
        """
        dims = []
        for dim in (0, 1):
            pos = numpy.zeros(self.max_shape[dim], dtype=numpy.int8)
            n = self.max_shape[dim] // self.module_size[dim]
            for i in range(1, n):
                pos[i * self.module_size[dim] - 1] = 1
                pos[i * self.module_size[dim]] = 1
            pos[0] = 1
            pos[-1] = 1
            dims.append(numpy.atleast_2d(pos))
        # This is just an "outer_or"
        mask = numpy.logical_or(dims[0].T, dims[1])
        return mask.astype(numpy.int8)

    def get_pixel_corners(self, d1=None, d2=None):
        """
        Calculate the position of the corner of the pixels

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
                    edges1, edges2 = self.calc_pixels_edges()
                    p1 = mathutil.expand2d(edges1, self.shape[1] + 1, False)
                    p2 = mathutil.expand2d(edges2, self.shape[0] + 1, True)
                    # p3 = None
                    self._pixel_corners = numpy.zeros((self.shape[0], self.shape[1], 4, 3), dtype=numpy.float32)
                    self._pixel_corners[:, :, 0, 1] = p1[:-1, :-1]
                    self._pixel_corners[:, :, 0, 2] = p2[:-1, :-1]
                    self._pixel_corners[:, :, 1, 1] = p1[1:, :-1]
                    self._pixel_corners[:, :, 1, 2] = p2[1:, :-1]
                    self._pixel_corners[:, :, 2, 1] = p1[1:, 1:]
                    self._pixel_corners[:, :, 2, 2] = p2[1:, 1:]
                    self._pixel_corners[:, :, 3, 1] = p1[:-1, 1:]
                    self._pixel_corners[:, :, 3, 2] = p2[:-1, 1:]
                    # if p3 is not None:
                    #     # non flat detector
                    #    self._pixel_corners[:, :, 0, 0] = p3[:-1, :-1]
                    #     self._pixel_corners[:, :, 1, 0] = p3[1:, :-1]
                    #     self._pixel_corners[:, :, 2, 0] = p3[1:, 1:]
                    #     self._pixel_corners[:, :, 3, 0] = p3[:-1, 1:]
        return self._pixel_corners

    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        :param d1: the Y pixel positions (slow dimension)
        :type d1: ndarray (1D or 2D)
        :param d2: the X pixel positions (fast dimension)
        :type d2: ndarray (1D or 2D)

        :return: position in meter of the center of each pixels.
        :rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.

        """
        edges1, edges2 = self.calc_pixels_edges()

        if (d1 is None) or (d2 is None):
            if center:
                # Take the center of each pixel
                d1 = 0.5 * (edges1[:-1] + edges1[1:])
                d2 = 0.5 * (edges2[:-1] + edges2[1:])
            else:
                # take the lower corner
                d1 = edges1[:-1]
                d2 = edges2[:-1]
            p1 = numpy.outer(d1, numpy.ones(self.shape[1]))
            p2 = numpy.outer(numpy.ones(self.shape[0]), d2)
        else:
            if center:
                # Not +=: do not mangle in place arrays
                d1 = d1 + 0.5
                d2 = d2 + 0.5
            p1 = numpy.interp(d1, numpy.arange(self.max_shape[0] + 1), edges1, edges1[0], edges1[-1])
            p2 = numpy.interp(d2, numpy.arange(self.max_shape[1] + 1), edges2, edges2[0], edges2[-1])
        return p1, p2, None

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        dico = {}
        if ((self.max_shape is not None) and
                ("MAX_SHAPE" in dir(self.__class__)) and
                (tuple(self.max_shape) != tuple(self.__class__.MAX_SHAPE))):
            dico["max_shape"] = self.max_shape
        if ((self.module_size is not None) and
                (tuple(self.module_size) != tuple(self.__class__.MODULE_SIZE))):
            dico["module_size"] = self.module_size
        return dico

    def set_config(self, config):
        """set the config of the detector

        For Xpad detector, possible keys are: max_shape, module_size

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

        # pixel size is enforced by the detector itself
        if "max_shape" in config:
            self.max_shape = tuple(config["max_shape"])
        module_size = config.get("module_size")
        if module_size is not None:
            self.module_size = tuple(module_size)
        return self


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
    IS_CONTIGUOUS = False
    force_pixel = True
    MAX_SHAPE = (960, 560)
    uniform_pixel = False
    aliases = ["Xpad S540 flat", "d5"]
    MODULE_SIZE = (120, 80)  # number of pixels per module (y, x)
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_PIXEL_SIZE_RELATIVE = 2.5

    def __init__(self, pixel1=130e-6, pixel2=130e-6, max_shape=None, module_size=None):
        super(Xpad_flat, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
        self._pixel_corners = None
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % (self.name, self.pixel1, self.pixel2)

    def calc_pixels_edges(self):
        """
        Calculate the position of the pixel edges, specific to the S540, d5 detector
        """
        if self._pixel_edges is None:
            # all pixel have the same size along the vertical axis, some pixels are larger along the horizontal one
            pixel_size1 = numpy.ones(self.max_shape[0]) * self._pixel1
            pixel_size2 = self._calc_pixels_size(self.max_shape[1], self.module_size[1], self._pixel2)
            pixel_edges1 = numpy.zeros(self.max_shape[0] + 1)
            pixel_edges2 = numpy.zeros(self.max_shape[1] + 1)
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
        # working in dim0 = Y
        for i in range(0, self.max_shape[0], self.module_size[0]):
            mask[i, :] = 1
            mask[i + self.module_size[0] - 1, :] = 1
        # working in dim1 = X
        for i in range(0, self.max_shape[1], self.module_size[1]):
            mask[:, i] = 1
            mask[:, i + self.module_size[1] - 1] = 1
        return mask

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
        :param use_cython: set to False to test Numpy implementation
        :return: position in meter of the center of each pixels.
        :rtype: ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if self.shape:
            if (d1 is None) or (d2 is None):
                d1 = mathutil.expand2d(numpy.arange(self.shape[0]).astype(numpy.float32), self.shape[1], False)
                d2 = mathutil.expand2d(numpy.arange(self.shape[1]).astype(numpy.float32), self.shape[0], True)
        corners = self.get_pixel_corners()
        if center:
            # note += would make an increment in place which is bad (segfault !)
            d1 = d1 + 0.5
            d2 = d2 + 0.5
        if bilinear and use_cython:
            p1, p2, _p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners)
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
            # To ensure numerical consitency with cython procedure.
            p1 = p1.astype(numpy.float32)
            p2 = p2.astype(numpy.float32)
        return p1, p2, None

    def get_pixel_corners(self):
        """
        Calculate the position of the corner of the pixels

        :return:  4D array containing:
                    pixel index (slow dimension)
                    pixel index (fast dimension)
                    corner index (A, B, C or D), triangles or hexagons can be handled the same way
                    vertex position (z,y,x)
        """
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    pixel_size1 = numpy.ones(self.max_shape[0]) * self._pixel1
                    pixel_size2 = self._calc_pixels_size(self.max_shape[1], self.module_size[1], self._pixel2)
                    # half pixel offset
                    pixel_center1 = pixel_size1 / 2.0  # half pixel offset
                    pixel_center2 = pixel_size2 / 2.0
                    # size of all preceeding pixels
                    pixel_center1[1:] += numpy.cumsum(pixel_size1[:-1])
                    pixel_center2[1:] += numpy.cumsum(pixel_size2[:-1])
                    # gaps
                    for i in range(self.max_shape[0] // self.module_size[0]):
                        pixel_center1[i * self.module_size[0]:
                                      (i + 1) * self.module_size[0]] += i * self.MODULE_GAP[0]
                    for i in range(self.max_shape[1] // self.module_size[1]):
                        pixel_center2[i * self.module_size[1]:
                                      (i + 1) * self.module_size[1]] += i * self.MODULE_GAP[1]

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


class Cirpad(ImXPadS10):
    MAX_SHAPE = (11200, 120)
    IS_FLAT = False
    IS_CONTIGUOUS = False
    force_pixel = True
    uniform_pixel = False
    aliases = ["CirPAD", "XCirpad"]
    MEDIUM_MODULE_SIZE = (560, 120)
    MODULE_SIZE = (80, 120)  # number of pixels per module (y, x)
    PIXEL_SIZE = (130e-6, 130e-6)
    DIFFERENT_PIXEL_SIZE = 2.5
    ROT = [0, 0, 6.74]

    # static functions used in order to define the Cirpad
    @staticmethod
    def _M(theta, u):
        """
        :param theta: the axis value in radian
        :type theta: float
        :param u: the axis vector [x, y, z]
        :type u: [float, float, float]
        :return: the rotation matrix
        :rtype: numpy.ndarray (3, 3)
        """
        c = numpy.cos(theta)
        one_minus_c = 1 - c
        s = numpy.sin(theta)
        return [[c + u[0] ** 2 * one_minus_c,
                 u[0] * u[1] * one_minus_c - u[2] * s,
                 u[0] * u[2] * one_minus_c + u[1] * s],
                [u[0] * u[1] * one_minus_c + u[2] * s,
                 c + u[1] ** 2 * one_minus_c,
                 u[1] * u[2] * one_minus_c - u[0] * s],
                [u[0] * u[2] * one_minus_c - u[1] * s,
                 u[1] * u[2] * one_minus_c + u[0] * s,
                 c + u[2] ** 2 * one_minus_c]]

    @staticmethod
    def _rotation(md, rot):
        shape = md.shape
        axe = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Maybe a parameter
        P = functools.reduce(numpy.dot, [Cirpad._M(numpy.radians(rot[i]), axe[i]) for i in range(len(rot))])
        try:
            nmd = numpy.transpose(numpy.reshape(numpy.tensordot(P, numpy.reshape(numpy.transpose(md), (3, shape[0] * shape[1] * 4)), axes=1), (3, 4, shape[1], shape[0])))
        except IndexError:
            nmd = numpy.transpose(numpy.tensordot(P, numpy.transpose(md), axes=1))
        return(nmd)

    @staticmethod
    def _translation(md, u):
        return md + u

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        ImXPadS10.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def _calc_pixels_size(self, length, module_size, pixel_size):
        size = numpy.ones(length)
        n = (length // module_size)
        for i in range(1, n):
            size[i * module_size - 1] = self.DIFFERENT_PIXEL_SIZE
            size[i * module_size] = self.DIFFERENT_PIXEL_SIZE
        return pixel_size * size

    def _passage(self, corners, rot):
        deltaZ, deltaY = 0.0, 0.0
        nmd = self._rotation(corners, rot)
        # Size in mm of the chip in the Y direction (including 10px gap)
        size_Y = ((560.0 + 3 * 6 + 10) * 0.13 / 1000)
        for i in range(1, int(round(numpy.abs(rot[2]) / 6.74))):
            deltaZ = deltaZ + numpy.sin(numpy.deg2rad(rot[2]))
        for i in range(int(round(numpy.abs(rot[2]) / 6.74))):
            deltaY = deltaY + numpy.cos(numpy.deg2rad(rot[2] - 6.74 * (i + 1)))
        return self._translation(nmd, [size_Y * deltaZ, size_Y * deltaY, 0])

    def _get_pixel_corners(self):
        pixel_size1 = self._calc_pixels_size(self.MEDIUM_MODULE_SIZE[0],
                                             self.MODULE_SIZE[0],
                                             self.PIXEL_SIZE[0])
        pixel_size2 = (numpy.ones(self.MEDIUM_MODULE_SIZE[1]) * self.PIXEL_SIZE[1]).astype(numpy.float32)
        # half pixel offset
        pixel_center1 = pixel_size1 / 2.0  # half pixel offset
        pixel_center2 = pixel_size2 / 2.0
        # size of all preceeding pixels
        pixel_center1[1:] += numpy.cumsum(pixel_size1[:-1])
        pixel_center2[1:] += numpy.cumsum(pixel_size2[:-1])

        pixel_center1.shape = -1, 1
        pixel_center1.strides = pixel_center1.strides[0], 0

        pixel_center2.shape = 1, -1
        pixel_center2.strides = 0, pixel_center2.strides[1]

        pixel_size1.shape = -1, 1
        pixel_size1.strides = pixel_size1.strides[0], 0

        pixel_size2.shape = 1, -1
        pixel_size2.strides = 0, pixel_size2.strides[1]

        # Position of the first module
        corners = numpy.zeros((self.MEDIUM_MODULE_SIZE[0], self.MEDIUM_MODULE_SIZE[1], 4, 3), dtype=numpy.float32)
        corners[:, :, 0, 1] = pixel_center1 - pixel_size1 / 2.0
        corners[:, :, 0, 2] = pixel_center2 - pixel_size2 / 2.0
        corners[:, :, 1, 1] = pixel_center1 + pixel_size1 / 2.0
        corners[:, :, 1, 2] = pixel_center2 - pixel_size2 / 2.0
        corners[:, :, 2, 1] = pixel_center1 + pixel_size1 / 2.0
        corners[:, :, 2, 2] = pixel_center2 + pixel_size2 / 2.0
        corners[:, :, 3, 1] = pixel_center1 - pixel_size1 / 2.0
        corners[:, :, 3, 2] = pixel_center2 + pixel_size2 / 2.0

        # modules = [self._passage(corners, [self.ROT[0], self.ROT[1], self.ROT[2] * i]) for i in range(20)]
        modules = list()
        dz, dy = 0.0, 0.0
        module_size = ((560.0 + 3 * 6 + 10) * 0.13 / 1000)
        for m in range(20):
            # rotation
            rot = numpy.array(self.ROT)
            rot[2] *= m

            # translation
            dy += numpy.cos(numpy.deg2rad(-rot[2]))
            u = numpy.array([module_size * dz, module_size * dy, 0])
            dz -= numpy.sin(numpy.deg2rad(rot[2]))

            # compute
            module = self._rotation(corners, rot)
            module = self._translation(module, u)
            modules.append(module)

        result = numpy.concatenate(modules, axis=0)
        result = numpy.ascontiguousarray(result, result.dtype)
        return result

    def get_pixel_corners(self):
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    self._pixel_corners = self._get_pixel_corners()
        return self._pixel_corners

    # TODO !!!
    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        if (d1 is None) or d2 is None:
            d1 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[0]).astype(numpy.float32), self.MAX_SHAPE[1], False)
            d2 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[1]).astype(numpy.float32), self.MAX_SHAPE[0], True)
        corners = self.get_pixel_corners()
        if center:
            # avoid += It modifies in place and segfaults
            d1 = d1 + 0.5
            d2 = d2 + 0.5
        if False and use_cython:
            p1, p2, p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners, is_flat=False)
            p1.shape = d1.shape
            p2.shape = d2.shape
            p3.shape = d2.shape
        else:  # TODO verified
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


class Cirpad2Module(ImXPadS70):
    """
    ImXPad detector: ImXPad s70 detector with 1x7modules
    """
    MODULE_SIZE = (80, 120)  # number of pixels per module (y, x)
    MAX_SHAPE = (560, 120)  # max size of the detector
    PIXEL_SIZE = (130e-6, 130e-6)
    BORDER_SIZE_RELATIVE = 2.5
    force_pixel = True
    aliases = ["Cirpad2Module"]
    PIXEL_EDGES = None  # array of size max_shape+1: pixels are contiguous

    def __init__(self, pixel1=130e-6, pixel2=130e-6):
        super(Cirpad2Module, self).__init__(pixel1=pixel1, pixel2=pixel2)


class Cirpad2(Detector):
    MAX_SHAPE = (11200, 120)  # max size of the detector as the 20 detector
    IS_FLAT = False
    IS_CONTIGUOUS = False
    force_pixel = True
    uniform_pixel = False
    aliases = ["Cirpad2"]
    MEDIUM_MODULE_SIZE = (560, 120)  # size of one module, as one detector
    MODULE_SIZE = (80, 120)  # number of pixels per chip (y, x)
    PIXEL_SIZE = (130e-6, 130e-6)
    DIFFERENT_PIXEL_SIZE = 2.5

    # static functions used in order to define the Cirpad
    @staticmethod
    def _M(theta, u):
        """
        :param theta: the axis value in radian
        :type theta: float
        :param u: the axis vector [x, y, z]
        :type u: [float, float, float]
        :return: the rotation matrix
        :rtype: numpy.ndarray (3, 3)
        """
        c = numpy.cos(theta)
        one_minus_c = 1 - c
        s = numpy.sin(theta)
        return [[c + u[0] ** 2 * one_minus_c,
                 u[0] * u[1] * one_minus_c - u[2] * s,
                 u[0] * u[2] * one_minus_c + u[1] * s],
                [u[0] * u[1] * one_minus_c + u[2] * s,
                 c + u[1] ** 2 * one_minus_c,
                 u[1] * u[2] * one_minus_c - u[0] * s],
                [u[0] * u[2] * one_minus_c - u[1] * s,
                 u[1] * u[2] * one_minus_c + u[0] * s,
                 c + u[2] ** 2 * one_minus_c]]

    @staticmethod
    def _rotation(md, rot):
        shape = md.shape
        axe = numpy.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Maybe a parameter
        P = functools.reduce(numpy.dot, [Cirpad2._M(numpy.radians(rot[i]), axe[i]) for i in range(len(rot))])
        try:
            nmd = numpy.transpose(numpy.reshape(numpy.tensordot(P, numpy.reshape(numpy.transpose(md), (3, shape[0] * shape[1] * 4)), axes=1), (3, 4, shape[1], shape[0])))
        except IndexError:
            nmd = numpy.transpose(numpy.tensordot(P, numpy.transpose(md), axes=1))
        return(nmd)

    def __init__(self, pixel1=130e-6, pixel2=130e-6, dist=0, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=self.MAX_SHAPE)
        from .. import geometry
        self.modules = list()
        self.modules_geometry = list()
        self.modules_param = list()
        deltaZ = 0
        deltaY = 0
        # init 20 modules as 20 detectors.
        for i in range(20):
            module = Cirpad2Module()
            mdgeometry = geometry.Geometry(dist=dist, poni1=poni1, poni2=poni2,
                                           rot1=rot1, rot2=rot2, rot3=rot3,
                                           pixel1=pixel1, pixel2=pixel2, detector=module)
            self.modules.append(module)
            self.modules_geometry.append(mdgeometry)
            self.modules_param.append([0.65 + deltaZ, deltaY, 0, 0, numpy.deg2rad(-i * 6.74), 0])
            deltaZ += 0.0043
            deltaY -= 0.0017
            # deltaZ -= numpy.sin(numpy.deg2rad(-i*6.74))
            # deltaY -= numpy.cos(numpy.deg2rad(-i*6.74))

    def get_config(self):
        """Return the configuration with arguments to the constructor
        :return: dict with param for serialization
        """
        return OrderedDict((("distance", self.dist),
                            ("poni1", self.poni1),
                            ("poni2", self.poni2),
                            ("rot1", self.rot1),
                            ("rot2", self.rot2),
                            ("rot3", self.rot3)))

    def _calc_pixels_size(self, length, module_size, pixel_size):
        size = numpy.ones(length)
        n = (length // module_size)
        for i in range(1, n):
            size[i * module_size - 1] = self.DIFFERENT_PIXEL_SIZE
            size[i * module_size] = self.DIFFERENT_PIXEL_SIZE
        return pixel_size * size

    def _get_pixel_corners(self):
        pixel_size1 = self._calc_pixels_size(self.MEDIUM_MODULE_SIZE[0],
                                             self.MODULE_SIZE[0],
                                             self.PIXEL_SIZE[0])
        pixel_size2 = (numpy.ones(self.MEDIUM_MODULE_SIZE[1]) * self.PIXEL_SIZE[1]).astype(numpy.float32)
        # half pixel offset
        pixel_center1 = pixel_size1 / 2.0  # half pixel offset
        pixel_center2 = pixel_size2 / 2.0
        # size of all preceeding pixels
        pixel_center1[1:] += numpy.cumsum(pixel_size1[:-1])
        pixel_center2[1:] += numpy.cumsum(pixel_size2[:-1])

        pixel_center1.shape = -1, 1
        pixel_center1.strides = pixel_center1.strides[0], 0

        pixel_center2.shape = 1, -1
        pixel_center2.strides = 0, pixel_center2.strides[1]

        pixel_size1.shape = -1, 1
        pixel_size1.strides = pixel_size1.strides[0], 0

        pixel_size2.shape = 1, -1
        pixel_size2.strides = 0, pixel_size2.strides[1]

        # Position of the first module
        corners = numpy.zeros((self.MEDIUM_MODULE_SIZE[0], self.MEDIUM_MODULE_SIZE[1], 4, 3), dtype=numpy.float32)
        corners[:, :, 0, 1] = pixel_center1 - pixel_size1 / 2.0
        corners[:, :, 0, 2] = pixel_center2 - pixel_size2 / 2.0
        corners[:, :, 1, 1] = pixel_center1 + pixel_size1 / 2.0
        corners[:, :, 1, 2] = pixel_center2 - pixel_size2 / 2.0
        corners[:, :, 2, 1] = pixel_center1 + pixel_size1 / 2.0
        corners[:, :, 2, 2] = pixel_center2 + pixel_size2 / 2.0
        corners[:, :, 3, 1] = pixel_center1 - pixel_size1 / 2.0
        corners[:, :, 3, 2] = pixel_center2 + pixel_size2 / 2.0

        modules_position = list()
        for param, geometry in zip(self.modules_param, self.modules_geometry):
            zyx = geometry.calc_pos_zyx(d0=0, d1=0, d2=0, param=param, corners=True)
            modules_position.append(numpy.moveaxis(zyx, 0, -1))
            """
            c0 = set_modules_position[i][0,0,0,:]
            c1 = set_modules_position[i][559,0,0,:]
            c2 = set_modules_position[i][0,119,0,:]
            size1 = numpy.sqrt((c0[0]-c1[0])**2 + (c0[1]-c1[1])**2 +(c0[2]-c1[2])**2 )
            size2 = numpy.sqrt((c0[0]-c2[0])**2 + (c0[1]-c2[1])**2 +(c0[2]-c2[2])**2 )
            print(i, size1, size2)
            """
        result = numpy.concatenate(modules_position, axis=0)
        result = numpy.ascontiguousarray(result, result.dtype)
        return result

    def get_pixel_corners(self):
        if self._pixel_corners is None:
            with self._sem:
                if self._pixel_corners is None:
                    self._pixel_corners = self._get_pixel_corners()
        return self._pixel_corners

    # TODO !!!
    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        if (d1 is None) or d2 is None:
            d1 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[0]).astype(numpy.float32), self.MAX_SHAPE[1], False)
            d2 = mathutil.expand2d(numpy.arange(self.MAX_SHAPE[1]).astype(numpy.float32), self.MAX_SHAPE[0], True)
        corners = self.get_pixel_corners()
        if center:
            # avoid += It modifies in place and segfaults
            d1 = d1 + 0.5
            d2 = d2 + 0.5
        if False and use_cython:
            p1, p2, p3 = bilinear.calc_cartesian_positions(d1.ravel(), d2.ravel(), corners, is_flat=False)
            p1.shape = d1.shape
            p2.shape = d2.shape
            p3.shape = d2.shape
        else:  # TODO verified
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
