#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023 European Synchrotron Radiation Facility, Grenoble, France
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
Utilities, mainly for mask manipulation
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "30/01/2023"
__status__ = "production"

import numpy


def _search_gap(mask, dim=0):
    shape = mask.shape
    m0 = numpy.sum(mask, axis=dim, dtype="int") == shape[dim]
    if m0.any():
        m0 = numpy.asarray(m0, "int8")
        d0=m0[1:]-m0[:-1]
        starts = numpy.where(d0==1)[0]
        stops = numpy.where(d0==-1)[0]
        if  (len(starts) == 0):
            starts = numpy.array([-1])
        if  (len(stops) == 0):
            stops = numpy.array([len(m0)-1])
        if (stops[0]<starts[0]):
            starts = numpy.concatenate(([-1], starts))
        if (stops[-1]<starts[-1]):
            stops = numpy.concatenate((stops, [len(m0)-1]))
        r0 = [ (start+1, stop+1) for start,stop  in zip(starts, stops)]
    else:
        r0 = []
    return r0


def search_gaps(mask):
    """Provide a list of gaps in vertical (dim1) and horizontal (dim2) directions.
    :param mask: 2D array with the mask
    """
    assert mask.ndim == 2
    mask = numpy.asarray(mask, dtype="bool")
    return _search_gap(mask, 1), _search_gap(mask, 0)


def build_gaps(shape, gaps):
    """Build a mask image from the dist of gaps provided by `search_gaps`

    :param shape: shape of the image
    :param gaps: 2-tuple of start/end
    """
    g0, g1 = gaps
    mask = numpy.zeros(shape, dtype=numpy.int8)
    for start,stop in g0:
        mask[start:stop] = 1
    for start,stop in g1:
        mask[:, start:stop] = 1
    return mask
