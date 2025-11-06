#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2025 European Synchrotron Radiation Facility, Grenoble, France
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
Detectors manufactured by ESRF
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/11/2025"
__status__ = "production"


import numpy
import logging
from ._common import Detector, SensorConfig, ModuleDetector
from ..utils.decorators import deprecated_args
logger = logging.getLogger(__name__)

#Define sensor used in Maxipix detectors
Si500 = SensorConfig.from_dict({"material": "Si", "thickness": 500e-6})


try:
    import fabio
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    fabio = None


class FReLoN(Detector):
    """
    FReLoN detector:
    The spline is mandatory to correct for geometric distortion of the taper

    TODO: create automatically a mask that removes pixels out of the "valid region"
    """
    MANUFACTURER = "ESRF"
    MAX_SHAPE = (2048, 2048)
    HAVE_TAPER = True

    @deprecated_args({"splinefile":"splineFile"}, since_version="2025.10")
    def __init__(self,
                splinefile: str|None=None,
                orientation=0):
        super().__init__(splinefile=splinefile, orientation=orientation)
        if splinefile:
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
        if not self._splinefile:
            return
        d1 = numpy.outer(numpy.arange(self.shape[0]), numpy.ones(self.shape[1])) + 0.5
        d2 = numpy.outer(numpy.ones(self.shape[0]), numpy.arange(self.shape[1])) + 0.5
        dX = self.spline.splineFuncX(d2, d1)
        dY = self.spline.splineFuncY(d2, d1)
        p1 = dY + d1
        p2 = dX + d2
        below_min = numpy.logical_or((p2 < self.spline.xmin), (p1 < self.spline.ymin))
        above_max = numpy.logical_or((p2 > self.spline.xmax), (p1 > self.spline.ymax))
        mask = numpy.logical_or(below_min, above_max).astype(numpy.int8)
        return mask



class Maxipix(ModuleDetector):
    """
    ESRF Maxipix detector: generic description containing mask algorithm

    Sub-classed by Maxipix2x2 and Maxipix5x1
    """
    MANUFACTURER = "ESRF"
    MODULE_SIZE = (256, 256)
    MODULE_GAP = (4, 4)
    MAX_SHAPE = (256, 256)
    force_pixel = True
    PIXEL_SIZE = (55e-6, 55e-6)
    aliases = ["Maxipix 1x1", "Maxipix1x1"]
    SENSORS = (Si500,)

    def __init__(self, pixel1=55e-6, pixel2=55e-6, max_shape=None, module_size=None, orientation=0, sensor:SensorConfig|None=None):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape,
                        module_size=module_size, orientation=orientation, sensor=sensor)
        self.uniform_pixel = True

    def calc_mask(self):
        """
        Returns a generic mask for Maxipix detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Maxipix detector does not know "
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

    

class Maxipix2x2(Maxipix):
    """
    Maxipix 2x2 detector
    """
    MAX_SHAPE = (516, 516)
    aliases = ["Maxipix 2x2"]


class Maxipix5x1(Maxipix):
    """
    Maxipix 5x1 detector
    """
    MAX_SHAPE = (256, 1296)
    aliases = ["Maxipix 5x1"]
