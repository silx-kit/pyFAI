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
Detectors manufactured by PSI, those may be different from the one from Dectris
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "2021 European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/03/2021"
__status__ = "production"

import numpy
import logging
from ._common import Detector
from ..utils import mathutil
logger = logging.getLogger(__name__)


class Jungfrau(Detector):
    """
    Raw Jungfrau module without sub-module pixel expension applied.
    """
    MANUFACTURER = "PSI"

    MODULE_SIZE = (256, 256)  # number of pixels per module (y, x)
    MAX_SHAPE = (512, 1024)  # max size of the detector
    PIXEL_SIZE = (75e-6, 75e-6)
    BORDER_SIZE_RELATIVE = 2.0
    force_pixel = True
    aliases = ["Jungfrau 500k"]
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
        return pixel_size * size

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None):
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

    def get_pixel_corners(self, correct_binning=False):
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
                    self._pixel_corners[:,:, 0, 1] = p1[:-1,:-1]
                    self._pixel_corners[:,:, 0, 2] = p2[:-1,:-1]
                    self._pixel_corners[:,:, 1, 1] = p1[1:,:-1]
                    self._pixel_corners[:,:, 1, 2] = p2[1:,:-1]
                    self._pixel_corners[:,:, 2, 1] = p1[1:, 1:]
                    self._pixel_corners[:,:, 2, 2] = p2[1:, 1:]
                    self._pixel_corners[:,:, 3, 1] = p1[:-1, 1:]
                    self._pixel_corners[:,:, 3, 2] = p2[:-1, 1:]
                    # if p3 is not None:
                    #     # non flat detector
                    #    self._pixel_corners[:, :, 0, 0] = p3[:-1, :-1]
                    #     self._pixel_corners[:, :, 1, 0] = p3[1:, :-1]
                    #     self._pixel_corners[:, :, 2, 0] = p3[1:, 1:]
                    #     self._pixel_corners[:, :, 3, 0] = p3[:-1, 1:]
        if correct_binning and self._pixel_corners.shape[:2] != self.shape:
            return self._rebin_pixel_corners()
        else:
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
        # if the detctor has been tweaked with an ASCII geometry ... fall-back on the classical method:
        if self._pixel_corners is not None:
            return Detector.calc_cartesian_positions(self, d1=d1, d2=d2, center=center, use_cython=use_cython)

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


class Jungfrau_16M_cor(Jungfrau):
    """Jungfrau 16 corrected for double-sized pixels
    """
    MODULE_SIZE = ((512 + 2), 1024 + 6)  # number of pixels per module (y, x)
    MAX_SHAPE = ((512 + 2) * 32, 1024 + 6)  # max size of the detector
    force_pixel = True
    aliases = ["Jungfrau 16M cor"]

    @staticmethod
    def load_geom(geom_fname):
        """"Load module geometry from ASCII file
        
        Stollen from Alejandro Homs' code
        """
        import re
        geom_re = re.compile('m(?P<mod>[0-9]+)/(?P<par>[^ \t=]+)[ \t]*='
                         '[ \t]*(?P<val>.+)')
        module_geom = {}
        with open(geom_fname) as ifile:
            for l in ifile:
                m = geom_re.match(l)
                if not m:
                    continue
                mod = int(m.group('mod'))
                mod_data = module_geom.setdefault(mod, {})
                val = m.group('val')
                if ' ' in val:
                    val = val.split()
                    if val[0].endswith('x') and val[1].endswith('y'):
                        val = [v[:-1] for v in val]
                else:
                    val = [val]
                key = m.group('par')
                if key.startswith("min_") or key.startswith("max_"):
                    mod_data[key] = int(val[0])
                elif key.startswith("corner"):
                    mod_data[key] = float(val[0])
                else:
                    mod_data[key] = [float(v) for v in val]
        return module_geom

    def init_from_geometry(self, filename):
        """initialize the detector from "geom" file produced at  PSI"""
        config = self.load_geom(filename)
        shape0 = 0
        shape1 = 0
        for m in config.values():
            shape0 = max(shape0, m.get("max_ss", 0))
            shape1 = max(shape1, m.get("max_fs", 0))
        self.MAX_SHAPE = (shape0 + 1, shape1 + 1)

        position_array = numpy.zeros(self.MAX_SHAPE + (4, 3), dtype=numpy.float32)

        for module in config.values():
            slab = position_array[module["min_ss"]: 1 + module["max_ss"], module["min_fs"]: 1 + module["max_fs"]]
            ss_edges = numpy.arange(2 + module["max_ss"] - module["min_ss"], dtype=numpy.int32)
            fs_edges = numpy.arange(2 + module["max_fs"] - module["min_fs"], dtype=numpy.int32)
            p1 = mathutil.expand2d(ss_edges, fs_edges.size, False)
            p2 = mathutil.expand2d(fs_edges, ss_edges.size, True)
            indexes = numpy.vstack([p2.ravel(), p1.ravel()])  # XY
            mat = numpy.array([module["fs"], module["ss"]], dtype=numpy.float64)
            position_xy = mat.dot(indexes) + numpy.array([[module["corner_x"]], [module["corner_y"]]])
            p2, p1 = position_xy.reshape((2,) + p1.shape)
            slab[:,:, 0, 1] = p1[:-1,:-1]
            slab[:,:, 0, 2] = p2[:-1,:-1]
            slab[:,:, 1, 1] = p1[1:,:-1]
            slab[:,:, 1, 2] = p2[1:,:-1]
            slab[:,:, 2, 1] = p1[1:, 1:]
            slab[:,:, 2, 2] = p2[1:, 1:]
            slab[:,:, 3, 1] = p1[:-1, 1:]
            slab[:,:, 3, 2] = p2[:-1, 1:]
        self._pixel_corners = (position_array * self.pixel1).astype(numpy.float32)
        self.IS_CONTIGUOUS = False

    def calc_mask(self):
        "Mask out sub-module junctions"
        mask = numpy.zeros(self.MAX_SHAPE, dtype=numpy.int8)
        mask[255:self.MAX_SHAPE[0] - 2:self.MODULE_SIZE[0]] = 1
        mask[256:self.MAX_SHAPE[0] - 2:self.MODULE_SIZE[0]] = 1
        mask[257:self.MAX_SHAPE[0] - 2:self.MODULE_SIZE[0]] = 1
        mask[258:self.MAX_SHAPE[0] - 2:self.MODULE_SIZE[0]] = 1

        for i in range(0, self.MODULE_SIZE[1], 258):
            mask[:, i + 255:self.MAX_SHAPE[1] - 2:self.MODULE_SIZE[1]] = 1
            mask[:, i + 256:self.MAX_SHAPE[1] - 2:self.MODULE_SIZE[1]] = 1
            mask[:, i + 257:self.MAX_SHAPE[1] - 2:self.MODULE_SIZE[1]] = 1
            mask[:, i + 258:self.MAX_SHAPE[1] - 2:self.MODULE_SIZE[1]] = 1
        return mask

