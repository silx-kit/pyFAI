#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2022-2025 European Synchrotron Radiation Facility, Grenoble, France
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
Description of the detector taken from `X-spectrum <https://x-spectrum.de/products/lambda/>`_ detectors.
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/02/2025"
__status__ = "production"

import numpy
import logging
from ._common import Detector, to_eng, SensorConfig
logger = logging.getLogger(__name__)


#Define sensors used in X-Spectrum detectors
Si300 = SensorConfig.from_dict({"material": "Si", "thickness": 300e-6})
Si500 = SensorConfig.from_dict({"material": "Si", "thickness": 500e-6})
GaAs500 = SensorConfig.from_dict({"material": "GaAs", "thickness": 500e-6})
CdTe1000 = SensorConfig.from_dict({"material": "CdTe", "thickness": 1000e-6})


class _Lambda(Detector):

    MANUFACTURER = "X-Spectrum"
    # This detector does not exist but those are place-holder
    MODULE_SIZE = (256, 256)
    MODULE_GAP = (4, 4)
    DUMMY = 0
    force_pixel = True
    SENSORS = (Si300, Si500, GaAs500, CdTe1000)

    def __init__(self, pixel1=55e-6, pixel2=55e-6, max_shape=None, module_size=None, orientation=0, sensor:SensorConfig|None=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation, sensor=sensor)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size

    def __repr__(self):
        txt = f"Detector {self.name}\t PixelSize= {to_eng(self._pixel1)}m, {to_eng(self._pixel2)}m"
        if self.orientation:
            txt+=f"\t {self.orientation.name} ({self.orientation.value})"
        if self.sensor:
            txt += f"\t {self.sensor}"
        return txt

    def calc_mask(self):
        """
        Returns a generic mask for module based detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Lambda detector does not know"
                                      "its max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        # working in dim0 = Y
        for i in range(self.module_size[0], self.max_shape[0],
                       self.module_size[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0],:] = 1
        # working in dim1 = X
        for i in range(self.module_size[1], self.max_shape[1],
                       self.module_size[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask


class Lambda60k(_Lambda):
    """
    LAMBDA 60k detector
    """
    MAX_SHAPE = (256, 256)
    aliases = ["Lambda 60k"]


class Lambda250k(_Lambda):
    """
    LAMBDA 250k detector
    """
    MAX_SHAPE = (516, 516)
    aliases = ["Lambda 250k"]


class Lambda750k(_Lambda):
    """
    LAMBDA 750k detector
    """
    MAX_SHAPE = (516, 1556)
    aliases = ["Lambda 750k"]


class Lambda2M(_Lambda):
    """
    LAMBDA 2M detector
    """
    MAX_SHAPE = (1556, 1556)
    aliases = ["Lambda 2M"]


class Lambda7M5(_Lambda):
    """
    LAMBDA 7.5M detector
    """
    MAX_SHAPE = (2596, 2596)
    aliases = ["Lambda 7.5M"]


class Lambda10M(_Lambda):
    """
    LAMBDA 10M detector
    """
    MAX_SHAPE = (2596, 4676)
    aliases = ["Lambda 10M"]

class Lambda9M(_Lambda):
    """
    LAMBDA 9M detector
    """
    MAX_SHAPE = (3868, 3227)
    aliases = ["Lambda 9M"]

    def calc_mask(self):
        """
        Returns a generic mask for module based detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Lambda detector does not know"
                                      "its max size ...")
        mask = numpy.zeros(self.max_shape, dtype=numpy.int8)
        logger.warning("Lambda9M mask is detector specific, no pixel are actually masked")
        return mask
