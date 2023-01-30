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

import os
import numpy
from scipy import ndimage


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

def decompose_detector(mask):
    """
    Decompose the detector as a list of panels

    :param mask: detector mask
    """
    labels, nlabels = ndimage.label(1-build_gaps(mask.shape, search_gaps(mask)))
    res = []
    for i in range(1, nlabels+1):
        w = numpy.where(labels==i)
        res.append((w[0].min(), w[0].max()+1, w[1].min(), w[1].max()+1))
    return res

def crystfel_mask(mask):
    """
    Generate a text with the mask description in CrystFEL format
    """
    assert mask.ndim == 2
    shape = mask.shape
    res = ["; Define a mask with the gaps of the detector"]
    g1, g2 = search_gaps(mask)
    for i, j in enumerate(g1):
        res+=[f"badregionH{i}/min_fs = 0",
              f"badregionH{i}/max_fs = {shape[1]-1}",
              f"badregionH{i}/min_ss = {j[0]}",
              f"badregionH{i}/max_ss = {j[1]-1}",
              " "
              ]
    for i, j in enumerate(g2):
        res+=[f"badregionV{i}/min_fs = {j[0]}",
              f"badregionV{i}/max_fs = {j[1]-1}",
              f"badregionV{i}/min_ss = 0",
              f"badregionV{i}/max_ss = {shape[0]-1}",
              " "]
    return os.linesep.join(res)

def crystfel_detector(detector):
    """
    Generate a CrystFEL detector definiton as text file
    """
    res = [crystfel_mask(detector.mask),"", ";Panel description"]
    corners = detector.get_pixel_corners()
    for i,j in enumerate(decompose_detector(detector.mask)):
        min_ss,max_ss,min_fs,max_fs = j
        max_ss -= 1
        max_fs -= 1
        corner_x = corners[min_ss, min_fs, 0, 2]
        corner_y = corners[min_ss, min_fs, 0, 1]
        ss = corners[max_ss, min_fs] - corners[min_ss, min_fs]
        fs = corners[min_ss, max_fs] - corners[min_ss, min_fs]
        # print(min_ss,max_ss,min_fs,max_fs, fs, ss)
        ss = ss.mean(axis=0)/detector.pixel1/(max_ss-min_ss)
        fs = fs.mean(axis=0)/detector.pixel2/(max_fs-min_fs)
        res+=[
    f"panel{i}/fs = {fs[2]:+f}x {fs[1]:+f}y",
    f"panel{i}/ss = {ss[2]:+f}x {ss[1]:+f}y",
    f"panel{i}/min_ss = {min_ss}",
    f"panel{i}/min_fs = {min_fs}",
    f"panel{i}/max_ss = {max_ss}",
    f"panel{i}/max_fs = {max_fs}",
    f"panel{i}/corner_x = {corner_x:+f}",
    f"panel{i}/corner_y = {corner_y:+f}",
    ""
                ]
    return os.linesep.join(res)
