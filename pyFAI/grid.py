# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import logging
import numpy

from . import detectors
import fabio
logger = logging.getLogger(__name__)


class Grid(object):
    """
    This class handles a regular grid in front of a detector to calibrate the
    geometrical distortion of the detector
    """
    def __init__(self, detector, image, mask=None, pitch=None, invert=False):
        """
        :param detector: instance of Detector or its name
        :param image: 2d array representing the image
        :param mask:
        :param pitch: 2-tuple representing the grid spacing in (y, x) coordinates, in meter
        :param invert: set to true if the image of the grid has regular dark spots (instead of bright points)
        """
        if isinstance(detector, detectors.Detector):
            self.detector = detectors.detector_factory(detector)
        else:
            self.detector = detector

        if isinstance(image, numpy.ndarray):
            self.image = image
        else:
            self.image = fabio.open(image).data

        if mask is not None:
            if isinstance(mask, numpy.ndarray):
                self.mask = mask
            else:
                self.mask = fabio.open(mask).data.astype(bool)
            if self.detector.mask is not None:
                self.mask = numpy.logical_or(self.detector.mask, self.mask)
        else:
            self.mask = numpy.zeros_like(self.image, bool)
        if invert:
            self.image = self.image.max() - self.image
        self.pitch = tuple(pitch[0], pitch[-1])

    def threshold(self, level=None, percentile=None):
        """
        Segment the image with a single threshold
        """
        if percentile and not level:
            data = self.image[self.mask]
            data.sort()
            level = data[int(len(data) * percentile / 100.)]
        raise NotImplementedError("TODO")
