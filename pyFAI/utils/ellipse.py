# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""This modules contains a function to fit without refinement an ellipse
on a set of points ....
"""

from __future__ import division, print_function

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/01/2018"
__status__ = "production"
__docformat__ = 'restructuredtext'

import numpy
from collections import namedtuple
Ellipse = namedtuple("Ellipse", ["center_1", "center_2", "angle", "half_long_axis", "half_short_axis"])


def fit_ellipse(pty, ptx):
    """Fit an ellipse

    inspired from
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    :param pty: point coordinates in the slow dimension (y)
    :param ptx: point coordinates in the fast dimension (x)
    """
    x = ptx[:, numpy.newaxis]
    y = pty[:, numpy.newaxis]
    D = numpy.hstack((x * x, x * y, y * y, x, y, numpy.ones_like(x)))
    S = numpy.dot(D.T, D)
    C = numpy.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = numpy.linalg.eig(numpy.dot(numpy.linalg.inv(S), C))
    n = numpy.argmax(numpy.abs(E))
    res = V[:, n]
    b, c, d, f, g, a = res[1] / 2, res[2], res[3] / 2, res[4] / 2, res[5], res[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    if b == 0:
        if a > c:
            angle = 0
        else:
            angle = numpy.pi / 2
    else:
        if a > c:
            angle = numpy.arctan2(2 * b, (a - c)) / 2
        else:
            angle = numpy.pi / 2 + numpy.arctan2(2 * b, (a - c)) / 2
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * numpy.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * numpy.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = numpy.sqrt(up / down1)
    res2 = numpy.sqrt(up / down2)
    return Ellipse(y0, x0, angle, max(res1, res2), min(res1, res2))
