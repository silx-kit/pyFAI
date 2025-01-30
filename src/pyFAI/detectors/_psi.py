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
__date__ = "25/06/2024"
__status__ = "production"

import numpy
import logging
from ._common import Detector, to_eng
from ._dectris import _Dectris
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

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None, orientation=0):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        self._pixel_edges = None  # array of size max_shape+1: pixels are contiguous
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size

    def __repr__(self):
        return f"Detector {self.name}%s\t PixelSize= {to_eng(self.pixel1)}m, {to_eng(self.pixel2)}m"

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
            d1, d2 = self._reorder_indexes_from_orientation(d1, d2, center)
            if center:
                # Not +=: do not mangle in place arrays
                d1 = d1 + 0.5
                d2 = d2 + 0.5
            p1 = numpy.interp(d1, numpy.arange(self.max_shape[0] + 1), edges1, edges1[0], edges1[-1])
            p2 = numpy.interp(d2, numpy.arange(self.max_shape[1] + 1), edges2, edges2[0], edges2[-1])
        return p1, p2, None

class Jungfrau4M(_Dectris):
    """
    Jungfrau 4M module without sub-module pixel expension applied.
    """
    MANUFACTURER = "PSI"
    MODULE_SIZE = (514, 1030)  # number of pixels per module (y, x)
    MAX_SHAPE = (2164, 2068)  # max size of the detector
    MODULE_GAP = (36, 8)
    PIXEL_SIZE = (75e-6, 75e-6)
    force_pixel = True
    aliases = ["Jungfrau 4M"]
    uniform_pixel = True

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None, orientation=0):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size
        self.offset1 = self.offset2 = None


class Jungfrau1M(Jungfrau4M):
    """
    Jungfrau 1M module without sub-module pixel expension applied.
    """
    MANUFACTURER = "PSI"
    MODULE_SIZE = (514, 1030)  # number of pixels per module (y, x)
    MAX_SHAPE = (1064, 1030)  # max size of the detector
    MODULE_GAP = (36, 8)
    PIXEL_SIZE = (75e-6, 75e-6)
    force_pixel = True
    aliases = ["Jungfrau 1M"]
    uniform_pixel = True


class Jungfrau8M(Jungfrau):
    """
    Jungfrau 8M module composed of 16 modules, 12 horizontals and 4 vertical

    To simplyfy the layout, one considers the chips (256x256)
    thus there are 128 chips (8 per modules)
    """
    MANUFACTURER = "PSI"
    MODULE_SIZE = (256, 256)
    MAX_SHAPE = (3333, 3212)  # max size of the detector
    PIXEL_SIZE = (75e-6, 75e-6)
    force_pixel = True
    aliases = ["Jungfrau 8M"]
    uniform_pixel = True
    module_positions = [[1, 607], [1, 866], [1, 1124], [1, 1382],
                        [69, 1646], [69, 1905], [69, 2163], [69, 2421],
                        [259, 607], [259, 866], [259, 1124],[259, 1382],
                        [328, 1646],[328, 1905], [328, 2163],[328, 2421],
                        [550, 607], [550, 866], [550, 1124], [550, 1382],
                        [619, 1646], [619, 1905], [619, 2163], [619, 2421],
                        [667, 2698], [667, 2957], [809, 607], [809, 866],
                        [809, 1124], [809, 1382], [856-259, 69], [856-259, 328],
                        [856, 69], [856, 328],
                        [878, 1646], [878, 1905], [878, 2163], [878, 2421],
                        [926, 2698], [926, 2957],
                        [926+259, 2698], [926+259, 2957],
                        [1100, 607], [1100, 866], [1100, 1124], [1100, 1382],
                        [1114, 69], [1114, 328],
                        [1169, 1646], [1169, 1905], [1169, 2163], [1169, 2421],
                        [1359, 607], [1359, 866], [1359, 1124], [1359, 1382],
                        [1372, 69], [1372, 328],
                        [1428, 1646], [1428, 1905], [1428, 2163], [1428, 2421],
                        [1442, 2698], [1442, 2957],
                        [1636, 1], [1636, 259],
                        [1650, 538], [1650, 797], [1650, 1055], [1650, 1313],
                        [1706, 2629], [1706, 2888],
                        [1719, 1577], [1719, 1836], [1719, 2094], [1719, 2352],
                        [1895, 1], [1895, 259],
                        [1909, 538], [1909, 797], [1909, 1055], [1909, 1313],
                        [1965, 2629], [1965, 2888],
                        [1978, 1577], [1978, 1836], [1978, 2094], [1978, 2352],
                        [2153, 1], [2153, 259],
                        [2200, 538], [2200, 797], [2200, 1055], [2200, 1313],
                        [2223, 2629], [2223, 2888],
                        [2269, 1577], [2269, 1836], [2269, 2094], [2269, 2352],
                        [2411, 1], [2411, 259],
                        [2459, 538], [2459, 797], [2459, 1055], [2459, 1313],
                        [2481, 2629], [2481, 2629+259],
                        [2528, 1577], [2528, 1836], [2528, 2094], [2528, 2352],
                        [2750, 538], [2750, 797], [2750, 1055], [2750, 1313],
                        [2819, 1577], [2819, 1836], [2819, 2094], [2819, 2352],
                        [3009, 538], [3009, 538+259], [3009, 1055], [3009, 1313],
                        [3078, 1577], [3078, 1836], [3078, 2094], [3078, 2352]]

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None, orientation=0):
        Jungfrau.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, module_size=module_size, orientation=orientation)

    def calc_mask(self):
        mask = numpy.ones(self.max_shape, dtype=numpy.int8)
        for i, j in self.module_positions:
            mask[i:i+self.module_size[0], j:j+self.module_size[1]] = 0
        return mask

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
