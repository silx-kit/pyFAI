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
Detectors with hexagonal pixel shape
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/06/2020"
__status__ = "production"
 

import numpy
import logging
import json
from ._common import Detector
logger = logging.getLogger(__name__)


class HexDetector(Detector):
    """
    Abstract class for regular hexagonal-pixel detectors

    This is characterized by the pitch, distance between 2 pixels 
    
    Pixels are aligned horizontally with the provided pitch
    Vertically, 2 raws are separated by pitch*sqrt(3)/2 
    Odd raws are offsetted horizontally by pitch/2 
    """
    uniform_pixel = False # ensures we use the array of position !
    IS_CONTIGUOUS = False
    IS_FLAT = True
    
    @staticmethod
    def build_pixel_coordinates(shape, pitch=1):
        """Build the 4D array with pixel coordinates for a detector composed of hexagonal-pixels
        
        :param shape: 2-tuple with size of the detector in number of pixels (y, x)
        :param pitch: the distance between two pixels
        :return: array with pixel coordinates 
        """
        assert len(shape) == 2
        ary = numpy.zeros(shape+(6, 3))
        a = 1
        sqrt3 = numpy.sqrt(3.0)
        h = sqrt3/2
        r = numpy.linspace(0, 2, 7, endpoint=True)[:-1] - 0.5
        c = numpy.exp((0+1j)*r*numpy.pi) / sqrt3
        c += complex(-c.real.min(), -c.imag.min())
        
        px = numpy.atleast_3d(numpy.outer(numpy.ones(shape[0]), numpy.arange(shape[1])))
        py = numpy.atleast_3d(numpy.outer(numpy.arange(shape[0]), numpy.ones(shape[1])))*h
        cxy = px + complex(0, 1)*py
        cplx = cxy + c[None, None, :]  
        ary[..., 1] = cplx.imag
        ary[..., 2] = cplx.real
        ary[1::2, ... , 2] += 0.5
        return ary*pitch
     
    def __init__(self, pitch=None, pixel1=None, pixel2=None, max_shape=None):
        if pitch:
            pitch = float(pitch)
            h = pitch*numpy.sqrt(3.0)/2.0
        else: #fallback on standard signature
            pitch = pixel2
            h = pixel1
        Detector.__init__(self, pixel1=h, pixel2=pitch, max_shape=max_shape)
        self.set_pixel_corners(self.build_pixel_coordinates(self.max_shape, self.pitch))        

    @property
    def pitch(self):
        "This is the distance between 2 pixel in the hexagonal pattern"
        return self.pixel2

    def __repr__(self):
        return f"Hexagonal-pixel detector {self.name}\t Pitch= {self.pitch:.3e} m"


class Pixirad1(HexDetector):
    MAX_SHAPE = (476, 512)  # max size of the detector
    MANUFACTURER = "Pixirad"
    aliases = ["Pixirad-1"]
    def __init__(self, pitch=60e-6, pixel1=None, pixel2=None, max_shape=None):
        HexDetector.__init__(self, pitch=pitch, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)


class Pixirad2(HexDetector):
    MAX_SHAPE = (476, 1024)  # max size of the detector
    MANUFACTURER = "Pixirad"
    aliases = ["Pixirad-2"]
    def __init__(self, pitch=60e-6, pixel1=None, pixel2=None, max_shape=None):
        HexDetector.__init__(self, pitch=pitch, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)


class Pixirad4(HexDetector):
    MAX_SHAPE = (476, 2048)  # max size of the detector
    MANUFACTURER = "Pixirad"
    aliases = ["Pixirad-4"]
    def __init__(self, pitch=60e-6, pixel1=None, pixel2=None, max_shape=None):
        HexDetector.__init__(self, pitch=pitch, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)


class Pixirad8(HexDetector):
    MAX_SHAPE = (476, 4096)  # max size of the detector
    MANUFACTURER = "Pixirad"
    aliases = ["Pixirad-8"]
    def __init__(self, pitch=60e-6, pixel1=None, pixel2=None, max_shape=None):
        HexDetector.__init__(self, pitch=pitch, pixel1=pixel1, pixel2=pixel2, max_shape=max_shape)
