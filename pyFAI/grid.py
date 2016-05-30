# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/02/2016"
__status__ = "development"
__docformat__ = 'restructuredtext'

import logging
import numpy

from . import detectors
try:
    from .third_party import six
except ImportError:
    import six
StringTypes = (six.binary_type, six.text_type)
import fabio
logger = logging.getLogger("pyFAI.grid")


class Grid(object):
    """
    This class handles a regular grid in front of a detector to calibrate the 
    geometrical distortion of the detector 
    """
    def __init__(self, detector, image, mask=None, pitch=None, invert=False):
        """
        @param detector: instance of Detector or its name
        @parma image: 2d array representing the image  
        @param mask:
        @param pitch: 2-tuple representing the grid spacing in (y, x) coordinates, in meter
        @param invert: set to true if the image of the grid has regular dark spots (instead of bright points) 
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
        
        @param 
        @param 
        """
        if percentile and not level:
            data = self.image[self.mask]
            data.sort()
            level = data[int(len(data) * percentile / 100.)]
        raise NotImplemented("TODO")
