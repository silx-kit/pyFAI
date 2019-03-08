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
Description of the `Dectris <https://www.dectris.com/>`_ detectors.
"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/03/2019"
__status__ = "production"


import os
import numpy
import logging
import json
from collections import OrderedDict
from ._common import Detector
from ..utils import expand2d
logger = logging.getLogger(__name__)

try:
    import fabio
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    fabio = None

logger = logging.getLogger(__name__)


class _Dectris(Detector):

    MANUFACTURER = "Dectris"


class Eiger(_Dectris):
    """
    Eiger detector: generic description containing mask algorithm
    
    Nota: 512k modules (514*1030) are made of 2x4 submodules of 256*256 pixels. 
    Two missing pixels are interpolated at each sub-module boundary which explains 
    the +2 and the +6 pixels.    
    """
    MODULE_SIZE = (514, 1030)
    MODULE_GAP = (37, 10)
    force_pixel = True

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size
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
        for i in range(self.module_size[0], self.max_shape[0],
                       self.module_size[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.module_size[1], self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

    def calc_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        :param d1: the Y pixel positions (slow dimension)
        :type d1: ndarray (1D or 2D)
        :param d2: the X pixel positions (fast dimension)
        :type d2: ndarray (1D or 2D)

        :return: p1, p2 position in meter of the center of each pixels.
        :rtype: 2-tuple of numpy.ndarray

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if self.shape:
            if (d1 is None) or (d2 is None):
                d1 = expand2d(numpy.arange(self.shape[0]).astype(numpy.float32), self.shape[1], False)
                d2 = expand2d(numpy.arange(self.shape[1]).astype(numpy.float32), self.shape[0], True)

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
                    logger.warning("Surprising situation !!! please investigate: offset has shape %s and input array have %s", self.offset1.shape, d1.shape)
                    delta1 = delta2 = 0.
        if center:
            # Eiger detectors images are re-built to be contiguous
            delta1 += 0.5
            delta2 += 0.5
        # For Eiger,
        p1 = (self._pixel1 * (delta1 + d1))
        p2 = (self._pixel2 * (delta2 + d2))
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

        For Eiger detector, possible keys are: max_shape, module_size

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


class Eiger500k(Eiger):
    """
    Eiger 1M detector
    """
    MAX_SHAPE = (514, 1030)
    aliases = ["Eiger 500k"]


class Eiger1M(Eiger):
    """
    Eiger 1M detector
    """
    MAX_SHAPE = (1065, 1030)
    aliases = ["Eiger 1M"]


class Eiger4M(Eiger):
    """
    Eiger 4M detector
    """
    MAX_SHAPE = (2167, 2070)
    aliases = ["Eiger 4M"]


class Eiger9M(Eiger):
    """
    Eiger 9M detector
    """
    MAX_SHAPE = (3269, 3110)
    aliases = ["Eiger 9M"]


class Eiger16M(Eiger):
    """
    Eiger 16M detector
    """
    MAX_SHAPE = (4371, 4150)
    aliases = ["Eiger 16M"]


class Mythen(_Dectris):
    "Mythen dtrip detector from Dectris"
    aliases = ["Mythen 1280"]
    force_pixel = True
    MAX_SHAPE = (1, 1280)

    def __init__(self, pixel1=8e-3, pixel2=50e-6):
        super(Mythen, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Pilatus(_Dectris):
    """
    Pilatus detector: generic description containing mask algorithm

    Sub-classed by Pilatus1M, Pilatus2M and Pilatus6M
    """
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    force_pixel = True

    def __init__(self, pixel1=172e-6, pixel2=172e-6, max_shape=None, module_size=None,
                 x_offset_file=None, y_offset_file=None):
        super(Pilatus, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size
        self.set_offset_files(x_offset_file, y_offset_file)

    def __repr__(self):
        txt = "Detector %s\t PixelSize= %.3e, %.3e m" % (self.name, self.pixel1, self.pixel2)
        if self.x_offset_file:
            txt += "\t delta_x= %s" % self.x_offset_file
        if self.y_offset_file:
            txt += "\t delta_y= %s" % self.y_offset_file
        return txt

    def set_offset_files(self, x_offset_file=None, y_offset_file=None):
        self.x_offset_file = x_offset_file
        self.y_offset_file = y_offset_file
        if self.x_offset_file and self.y_offset_file:
            if fabio:
                self.offset1 = fabio.open(self.y_offset_file).data
                self.offset2 = fabio.open(self.x_offset_file).data
                self.uniform_pixel = False
            else:
                logging.error("FabIO is not available: no distortion correction for Pilatus detectors, sorry.")
                self.offset1 = None
                self.offset2 = None
                self.uniform_pixel = True
        else:
            self.offset1 = None
            self.offset2 = None
            self.uniform_pixel = True

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
                logger.error("set_splineFile with %s gave error: %s", splineFile, error)
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
        for i in range(self.module_size[0], self.max_shape[0],
                       self.module_size[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.module_size[1], self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

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
        if self.shape and ((d1 is None) or (d2 is None)):
            d1 = expand2d(numpy.arange(self.shape[0]).astype(numpy.float32), self.shape[1], False)
            d2 = expand2d(numpy.arange(self.shape[1]).astype(numpy.float32), self.shape[0], True)

        if (self.offset1 is None) or (self.offset2 is None):
            delta1 = delta2 = 0.
        else:
            if d2.ndim == 1:
                d1n = d1.astype(numpy.int32)
                d2n = d2.astype(numpy.int32)
                delta1 = -self.offset1[d1n, d2n] / 100.0  # Offsets are in percent of pixel and negative
                delta2 = -self.offset2[d1n, d2n] / 100.0
            else:
                if d1.shape == self.offset1.shape:
                    delta1 = -self.offset1 / 100.0  # Offsets are in percent of pixel and negative
                    delta2 = -self.offset2 / 100.0
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
                    delta1 = -delta1 / 100.0  # Offsets are in percent of pixel and negative
                    delta2 = -delta2 / 100.0  # former arrays were integers
                else:
                    logger.warning("Surprizing situation !!! please investigate:"
                                   " offset has shape %s and input array have %s",
                                   self.offset1.shape, d1.shape)
                    delta1 = delta2 = 0.
        # For Pilatus,
        if center:
            # Account for the pixel center: pilatus detector are contiguous
            delta1 += 0.5
            delta2 += 0.5
        p1 = (self._pixel1 * (delta1 + d1))
        p2 = (self._pixel2 * (delta2 + d2))
        return p1, p2, None

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        dico = OrderedDict()
        if ((self.max_shape is not None) and
                ("MAX_SHAPE" in dir(self.__class__)) and
                (tuple(self.max_shape) != tuple(self.__class__.MAX_SHAPE))):
            dico["max_shape"] = self.max_shape
        if ((self.module_size is not None) and
                (tuple(self.module_size) != tuple(self.__class__.MODULE_SIZE))):
            dico["module_size"] = self.module_size
        if self.x_offset_file is not None:
            dico["x_offset_file"] = self.x_offset_file
        if self.y_offset_file is not None:
            dico["y_offset_file"] = self.y_offset_file
        return dico

    def set_config(self, config):
        """set the config of the detector

        For Eiger detector, possible keys are: max_shape, module_size, x_offset_file, y_offset_file

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
        self.set_offset_files(config.get("x_offset_file"), config.get("y_offset_file"))
        return self


class Pilatus100k(Pilatus):
    """
    Pilatus 100k detector
    """
    MAX_SHAPE = (195, 487)
    aliases = ["Pilatus 100k"]


class Pilatus200k(Pilatus):
    """
    Pilatus 200k detector
    """
    MAX_SHAPE = (407, 487)
    aliases = ["Pilatus 200k"]


class Pilatus300k(Pilatus):
    """
    Pilatus 300k detector
    """
    MAX_SHAPE = (619, 487)
    aliases = ["Pilatus 300k"]


class Pilatus300kw(Pilatus):
    """
    Pilatus 300k-wide detector
    """
    MAX_SHAPE = (195, 1475)
    aliases = ["Pilatus 300kw"]


class Pilatus1M(Pilatus):
    """
    Pilatus 1M detector
    """
    MAX_SHAPE = (1043, 981)
    aliases = ["Pilatus 1M"]


class Pilatus2M(Pilatus):
    """
    Pilatus 2M detector
    """

    MAX_SHAPE = 1679, 1475
    aliases = ["Pilatus 2M"]


class Pilatus6M(Pilatus):
    """
    Pilatus 6M detector
    """
    MAX_SHAPE = (2527, 2463)
    aliases = ["Pilatus 6M"]


class PilatusCdTe(Pilatus):
    """
    Pilatus CdTe detector: Like the Pilatus with an extra 3 pixel in the middle
    of every module (vertically)
    """
    def calc_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Pilatus detector does not know "
                                      "its max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.module_size[0], self.max_shape[0],
                       self.module_size[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.module_size[1], self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        # Small gaps in the middle of the module
        for i in range(self.module_size[1] // 2, self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i - 1: i + 2] = 1

        return mask


class PilatusCdTe300k(PilatusCdTe):
    """
    Pilatus CdTe 300k detector
    """
    MAX_SHAPE = (619, 487)
    aliases = ["Pilatus CdTe 300k", "Pilatus 300k CdTe", "Pilatus300k CdTe", "Pilatus300kCdTe"]


class PilatusCdTe300kw(PilatusCdTe):
    """
    Pilatus CdTe 300k-wide detector
    """
    MAX_SHAPE = (195, 1475)
    aliases = ["Pilatus CdTe 300kw", "Pilatus 300kw CdTe", "Pilatus300kw CdTe", "Pilatus300kwCdTe"]


class PilatusCdTe1M(PilatusCdTe):
    """
    Pilatus CdTe 1M detector
    """
    MAX_SHAPE = (1043, 981)
    aliases = ["Pilatus CdTe 1M", "Pilatus 1M CdTe", "Pilatus1M CdTe", "Pilatus1MCdTe"]


class PilatusCdTe2M(PilatusCdTe):
    """
    Pilatus CdTe 2M detector
    """
    MAX_SHAPE = 1679, 1475
    aliases = ["Pilatus CdTe 2M", "Pilatus 2M CdTe", "Pilatus2M CdTe", "Pilatus2MCdTe"]
