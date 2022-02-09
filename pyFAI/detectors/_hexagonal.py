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


def build_pixel_coordinates(shape, pitch=1):
    """Build the 4D array with pixel coordinates for a detector composed of hexagonal-pixels
    
    :param shape: 2-tuple with size of the detector in number of pixels (y, x)
    :param pitch: the distance between two pixels
    :return: array with pixel coordinates 
    """
    assert len(shape) == 2
    ary = numpy.zeros(shape+(6, 3))
    a = 1
    sqrt3 = numpy.sqrt(3)
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
     
    
    