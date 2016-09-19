#!/usr/bin/python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015 European Synchrotron Radiation Facility, Grenoble, France
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

"test suite for preprocessing corrections"

from __future__ import absolute_import, division, print_function

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/09/2016"


import unittest
import numpy
import logging
import sys
import fabio
logger = getLogger(__file__)

from ..ext import preproc


class TestPreproc(unittest.TestCase):
    def test(self):
        """
        The final pattern should look like a 4x4 square with 1 and -1 elsewhere.
        """
        shape = 8, 8
        size = shape[0] * shape[1]
        target = numpy.ones(shape)
        target[:2, :] = 0
        target[-2:, :] = 0
        target[:, -2:] = 0
        target[:, :2] = 0
        mask = numpy.zeros(shape, "int8")
        mask[:2, :] = 1
        dark = numpy.random.poisson(10, size).reshape(shape)
        flat = 1.0 + numpy.random.random(shape)
        scale = 10
        raw = scale * flat + dark
        raw[-2:, :] = numpy.NaN
        dummy = -1
        raw[:, :2] = dummy
        flat[:, -2:] = dummy
        res = preproc.preproc(raw)
        # add some tests with various levels of conditionning

        res = preproc.preproc(raw, dark, flat, dummy=dummy, mask=mask, normalization_factor=scale)
        print(raw)
        print(numpy.round(res))
