#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2022-2022 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "21/03/2022"
__status__ = "production"

import numpy
import logging
from ._common import Detector
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class _Lambda(Detector):

    MANUFACTURER = "X-Spectrum"
    # This detector does not exist but those are place-holder
    MODULE_SIZE = (516, 516)
    MODULE_GAP = (6, 6)
    force_pixel = True

    def __init__(self, pixel1=55e-6, pixel2=55e-6, max_shape=None, module_size=None):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
        if (module_size is None) and ("MODULE_SIZE" in dir(self.__class__)):
            self.module_size = tuple(self.MODULE_SIZE)
        else:
            self.module_size = module_size

    def __repr__(self):
        return f"Detector {self.name}\t PixelSize= {self._pixel1:.3e}, {self._pixel2:.3e} m"

    def calc_mask(self):
        """
        Returns a generic mask for module based detectors...
        """
        if self.max_shape is None:
            raise NotImplementedError("Generic Lambda detector does not know"
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
    MAX_SHAPE = (516, 1554)
    aliases = ["Lambda 750k"]


class Lambda2M(_Lambda):
    """
    LAMBDA 2M detector
    """
    MAX_SHAPE = (1554, 1554)
    aliases = ["Lambda 2M"]
