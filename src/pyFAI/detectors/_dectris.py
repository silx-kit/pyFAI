#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2024 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2024"
__status__ = "production"

import os
import numpy
import logging
import json
from collections import OrderedDict
from ._common import Detector, Orientation, to_eng
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
    # This detector does not exist but those are place-holder
    MODULE_SIZE = (64, 128)
    MODULE_GAP = (9, 11)
    force_pixel = True
    DUMMY = -2
    DELTA_DUMMY = 1.5
    ORIENTATION = 3 # should be 2, Personal communication from Dectris: origin top-left looking from the sample to the detector, thus flip-rl

    def calc_mask(self):
        """
        Returns a generic mask for module based detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Dectris detector does not know"
                                      "its max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.module_size[0], self.max_shape[0],
                       self.module_size[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0],:] = 1
        # workinng in dim1 = X
        for i in range(self.module_size[1], self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask


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

    def __init__(self, pixel1=75e-6, pixel2=75e-6, max_shape=None, module_size=None, orientation=0):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size
        self.offset1 = self.offset2 = None

    def __repr__(self):
        txt = f"Detector {self.name}\t PixelSize= {to_eng(self._pixel1)}m, {to_eng(self._pixel2)}m"
        if self.orientation:
            txt += f"\t {self.orientation.name} ({self.orientation.value})"
        return txt

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
                r1, r2 = self._calc_pixel_index_from_orientation(center)
                delta = 0 if center else 1
                d1 = expand2d(r1, self.shape[1] + delta, False)
                d2 = expand2d(r2, self.shape[0] + delta, True)
            else:
                d1, d2 = self._reorder_indexes_from_orientation(d1, d2, center)

        if self.offset1 is None or self.offset2 is None:
            delta1 = delta2 = 0.
        else:
            #TODO: this does not take into account the orientation of the detector !
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
                    delta1[:s0,:s1] = self.offset1
                    delta2[:s0,:s1] = self.offset2
                    mask = numpy.where(delta1[-s0:,:s1] == 0)
                    delta1[-s0:,:s1][mask] = self.offset1[mask]
                    delta2[-s0:,:s1][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[-s0:, -s1:] == 0)
                    delta1[-s0:, -s1:][mask] = self.offset1[mask]
                    delta2[-s0:, -s1:][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[:s0, -s1:] == 0)
                    delta1[:s0, -s1:][mask] = self.offset1[mask]
                    delta2[:s0, -s1:][mask] = self.offset2[mask]
                    delta1 = delta1 / 100.0  # Offsets are in percent of pixel
                    delta2 = delta2 / 100.0  # former arrays were integers
                else:
                    logger.warning("Surprising situation !!! please investigate: offset has shape %s and input array have %s",
                                   self.offset1.shape, d1.shape)
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
        dico = {"orientation": self.orientation}
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
        self._orientation = Orientation(config.get("orientation", 3))
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


class Eiger2(Eiger):
    MODULE_SIZE = (512, 1028)
    MODULE_GAP = (38, 12)


class Eiger2_250k(Eiger2):
    """
    Eiger2 250k detector
    """
    MAX_SHAPE = (512, 512)
    aliases = ["Eiger2 250k"]


class Eiger2_500k(Eiger2):
    """
    Eiger2 500k detector
    """
    MAX_SHAPE = (512, 1028)
    aliases = ["Eiger2 500k"]


class Eiger2_1M(Eiger2):
    """
    Eiger2 1M detector
    """
    MAX_SHAPE = (1062, 1028)
    aliases = ["Eiger2 1M"]


class Eiger2_1MW(Eiger2):
    """
    Eiger2 1M-Wide detector
    """
    MAX_SHAPE = (512, 2068)
    aliases = ["Eiger2 1M-W"]


class Eiger2_2MW(Eiger2):
    """
    Eiger2 2M-Wide detector
    """
    MAX_SHAPE = (512, 4148)
    aliases = ["Eiger2 2M-W"]


class Eiger2_4M(Eiger2):
    """
    Eiger2 4M detector
    """
    MAX_SHAPE = (2162, 2068)
    aliases = ["Eiger2 4M"]


class Eiger2_9M(Eiger2):
    """
    Eiger2 9M detector
    """
    MAX_SHAPE = (3262, 3108)
    aliases = ["Eiger2 9M"]


class Eiger2_16M(Eiger2):
    """
    Eiger2 16M detector
    """
    MAX_SHAPE = (4362, 4148)
    aliases = ["Eiger2 16M"]


class Eiger2CdTe(Eiger2):
    """
    Eiger2 CdTe detector: Like the Eiger2 with an extra 2-pixel gap in the middle
    of every module (vertically)
    """

    def calc_mask(self):
        """
        Mask out an extra 2 pixels in the middle of each module
        """
        mask = super().calc_mask()
        # Add the small gaps in the middle of the module
        for i in range(self.module_size[1] // 2, self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i - 1: i + 1] = 1

        return mask


class Eiger2CdTe_500k(Eiger2CdTe):
    """
    Eiger2 CdTe 500k detector
    """
    MAX_SHAPE = (512, 1028)
    aliases = ["Eiger2 CdTe 500k"]


class Eiger2CdTe_1M(Eiger2CdTe):
    """
    Eiger2 CdTe 1M detector
    """
    MAX_SHAPE = (1062, 1028)
    aliases = ["Eiger2 CdTe 1M"]


class Eiger2CdTe_1MW(Eiger2CdTe):
    """
    Eiger2 CdTe 1M-Wide detector
    """
    MAX_SHAPE = (512, 2068)
    aliases = ["Eiger2 CdTe 1M-W"]


class Eiger2CdTe_2MW(Eiger2CdTe):
    """
    Eiger2 CdTe 2M-Wide detector
    """
    MAX_SHAPE = (512, 4148)
    aliases = ["Eiger2 CdTe 2M-W"]


class Eiger2CdTe_4M(Eiger2CdTe):
    """
    Eiger2 CdTe 4M detector
    """
    MAX_SHAPE = (2162, 2068)
    aliases = ["Eiger2 CdTe 4M"]


class Eiger2CdTe_9M(Eiger2CdTe):
    """
    Eiger2 CdTe 9M detector
    """
    MAX_SHAPE = (3262, 3108)
    aliases = ["Eiger2 CdTe 9M"]


class Eiger2CdTe_16M(Eiger2CdTe):
    """
    Eiger2 CdTe 16M detector
    """
    MAX_SHAPE = (4362, 4148)
    aliases = ["Eiger2 CdTe 16M"]


class Mythen(_Dectris):
    "Mythen strip detector from Dectris"
    aliases = ["Mythen 1280"]
    force_pixel = True
    MAX_SHAPE = (1, 1280)

    def __init__(self, pixel1=8e-3, pixel2=50e-6, orientation=0):
        super(Mythen, self).__init__(pixel1=pixel1, pixel2=pixel2, orientation=orientation)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return {"pixel1": self._pixel1,
                "pixel2": self._pixel2,
                "orientation": self.orientation or 3}

    def calc_mask(self):
        "Mythen have no masks"
        return None


class Pilatus(_Dectris):
    """
    Pilatus detector: generic description containing mask algorithm

    Sub-classed by Pilatus1M, Pilatus2M and Pilatus6M
    """
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)
    force_pixel = True


    def __init__(self, pixel1=172e-6, pixel2=172e-6, max_shape=None, module_size=None,
                 x_offset_file=None, y_offset_file=None, orientation=0):
        super(Pilatus, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size
        self.set_offset_files(x_offset_file, y_offset_file)

    def __repr__(self):
        txt = f"Detector {self.name}\t PixelSize= {to_eng(self._pixel1)}m, {to_eng(self._pixel2)}m"
        if self.orientation > 0:
            txt += f"\t {self.orientation.name} ({self.orientation.value})"
        if self.x_offset_file:
            txt += f"\t delta_x= {self.x_offset_file}"
        if self.y_offset_file:
            txt += f"\t delta_y= {self.y_offset_file}"
        return txt

    def set_offset_files(self, x_offset_file=None, y_offset_file=None):
        self.x_offset_file = x_offset_file
        self.y_offset_file = y_offset_file
        if self.x_offset_file and self.y_offset_file:
            if fabio:
                with fabio.open(self.y_offset_file) as fimgy:
                    self.offset1 = fimgy.data
                with fabio.open(self.x_offset_file) as fimgx:
                    self.offset2 = fimgx.data
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
                with fabio.open(self.y_offset_file) as fimgy:
                    self.offset1 = fimgy.data
                with fabio.open(self.x_offset_file) as fimgx:
                    self.offset2 = fimgx.data
            else:
                logging.error("FabIO is not available: no distortion correction for Pilatus detectors, sorry.")
                self.offset1 = None
                self.offset2 = None

        else:
            self._splineFile = None
            self.uniform_pixel = True

    splineFile = property(get_splineFile, set_splineFile)

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
        if self.shape:
            if ((d1 is None) or (d2 is None)):
                r1, r2 = self._calc_pixel_index_from_orientation(center)
                delta = 0 if center else 1
                d1 = expand2d(r1, self.shape[1] + delta, False)
                d2 = expand2d(r2, self.shape[0] + delta, True)
            else:
                d1, d2 = self._reorder_indexes_from_orientation(d1, d2, center)

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
                    delta1[:s0,:s1] = self.offset1
                    delta2[:s0,:s1] = self.offset2
                    mask = numpy.where(delta1[-s0:,:s1] == 0)
                    delta1[-s0:,:s1][mask] = self.offset1[mask]
                    delta2[-s0:,:s1][mask] = self.offset2[mask]
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
        dico = {"orientation": self.orientation or 3}
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
        self._orientation = Orientation(config.get("orientation", 0))
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


class Pilatus900k(Pilatus):
    """
    Pilatus 900k detector, assembly of 3x3 modules
    Available at NSLS-II 12-ID.

    This is different from the "Pilatus CdTe 900kw" available at ESRF ID06-LVP which is 1x9 modules
    """
    MAX_SHAPE = (619, 1475)
    aliases = ["Pilatus 900k"]


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
        Mask out an extra 3 pixel in the middle of each module
        """
        mask = super().calc_mask()
        # Add the small gaps in the middle of the module
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


class PilatusCdTe900kw(PilatusCdTe):
    """
    Pilatus CdTe 900k-wide detector, assembly of 1x9 modules
    Available at ESRF ID06-LVP

    This differes from the "Pilatus 900k" detector, assembly of 3x3 modules, available at NSLS-II 12-ID.
    """
    MAX_SHAPE = (195, 4439)
    aliases = ["Pilatus CdTe 900kw", "Pilatus 900kw CdTe", "Pilatus900kw CdTe", "Pilatus900kwCdTe"]


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


class Pilatus4(_Dectris):
    """
    Pilatus4 detector: generic description containing mask algorithm

    Sub-classed by Pilatus4_1M, Pilatus4_2M and Pilatus_4M
    """
    MODULE_SIZE = (255, 513)
    MODULE_GAP = (20, 7)
    force_pixel = True

    def __init__(self, pixel1=150e-6, pixel2=150e-6, max_shape=None, orientation=0):
        super(Pilatus4, self).__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        self.module_size = tuple(self.MODULE_SIZE)

    def __repr__(self):
        txt = f"Detector {self.name}\t PixelSize= {to_eng(self._pixel1)}m, {to_eng(self._pixel2)}m"
        if self.orientation:
            txt += f"\t {self.orientation.name} ({self.orientation.value})"
        return txt


class Pilatus4_1M(Pilatus4):
    MAX_SHAPE = 1080, 1033
    aliases = ["Pilatus4 1M"]


class Pilatus4_2M(Pilatus4):
    MAX_SHAPE = 1630, 1553
    aliases = ["Pilatus4 2M"]


class Pilatus4_4M(Pilatus4):
    MAX_SHAPE = 2180, 2073
    aliases = ["Pilatus4 4M"]


class Pilatus4_260k(Pilatus4):
    MAX_SHAPE = 530, 513
    aliases = ["Pilatus4 260k"]


class Pilatus4_260kw(Pilatus4):
    MAX_SHAPE = 255, 1033
    aliases = ["Pilatus4 260kw"]


class Pilatus4_CdTe(Pilatus):
    """
    Pilatus CdTe detector: Like the Pilatus4 with an extra gap of 1 pixel in the middle
    of every module (vertically)
    """

    def calc_mask(self):
        """
        Mask out an extra 3 pixel in the middle of each module
        """
        mask = super().calc_mask()
        # Add the small gaps in the middle of the module
        for i in range(self.module_size[1] // 2, self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i] = 1

        return mask


class Pilatus4_CdTe_1M(Pilatus4_CdTe):
    MAX_SHAPE = 1080, 1033
    aliases = ["Pilatus4 1M CdTe"]


class Pilatus4_CdTe_2M(Pilatus4_CdTe):
    MAX_SHAPE = 1630, 1553
    aliases = ["Pilatus4 2M CdTe"]


class Pilatus4_CdTe_4M(Pilatus4_CdTe):
    MAX_SHAPE = 2180, 2073
    aliases = ["Pilatus4 4M CdTe"]


class Pilatus4_CdTe_260k(Pilatus4_CdTe):
    MAX_SHAPE = 530, 513
    aliases = ["Pilatus4 260k CdTe"]


class Pilatus4_CdTe_260kw(Pilatus4_CdTe):
    MAX_SHAPE = 255, 1033
    aliases = ["Pilatus4 260kw CdTe"]
