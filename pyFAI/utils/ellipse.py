# !/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2021 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/02/2021"
__status__ = "production"
__docformat__ = 'restructuredtext'

import numpy
import logging
from math import sqrt, atan2, pi
from collections import namedtuple

_logger = logging.getLogger(__name__)

Ellipse = namedtuple("Ellipse", ["center_1", "center_2", "angle", "half_long_axis", "half_short_axis"])


def fit_ellipse(pty, ptx, _allow_delta=True):
    """Fit an ellipse

    Math from 
    https://mathworld.wolfram.com/Ellipse.html #15

    inspired from
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html

    :param pty: point coordinates in the slow dimension (y)
    :param ptx: point coordinates in the fast dimension (x)
    :raise ValueError: If the ellipse can't be fitted
    """
    x = ptx[:, numpy.newaxis]
    y = pty[:, numpy.newaxis]
    D = numpy.hstack((x * x, x * y, y * y, x, y, numpy.ones_like(x)))
    S = numpy.dot(D.T, D)
    try:
        inv = numpy.linalg.inv(S)
    except numpy.linalg.LinAlgError:
        if not _allow_delta:
            raise ValueError("Ellipse can't be fitted: singular matrix")
        # Try to do the same with a delta
        delta = 100
        ellipse = fit_ellipse(pty + delta, ptx + delta, _allow_delta=False)
        y0, x0, angle, wlong, wshort = ellipse
        return Ellipse(y0 - delta, x0 - delta, angle, wlong, wshort)

    C = numpy.zeros([6, 6], dtype=numpy.float64)
    C[0, 2] = C[2, 0] = 2.0
    C[1, 1] = -1.0
    E, V = numpy.linalg.eig(numpy.dot(inv, C))

    # First of all, sieve out all infinite and complex eigenvalues and come back to the Real world
    m = numpy.logical_and(numpy.isfinite(E), numpy.isreal(E))
    E, V = E[m].real, V[:, m].real

    # Ensures a>0, invert eigenvectors concerned
    V[:, V[0] < 0] = -V[:, V[0] < 0]
    # See https://mathworld.wolfram.com/Ellipse.html #15
    # Eigenvector must meet constraint (ac - b^2)>0 to be valid.
    A = V[0]
    B = V[1] / 2.0
    C = V[2]
    D = V[3] / 2.0
    F = V[4] / 2.0
    G = V[5]

    # Condition 1: Delta = det((a b d)(b c f)(d f g)) !=0
    Delta = A * (C * G - F * F) - G * B * B + D * (2 * B * F - C * D)
    # Condition 2: J>0
    J = (A * C - B * B)

    # Condition 3: Delta/(A+C)<0, replaces by Delta*(A+C)<0, less warnings
    m = numpy.logical_and(J > 0, Delta != 0)
    m = numpy.logical_and(m, Delta * (A + C) < 0)

    n = numpy.where(m)[0]
    if len(n) == 0:
        raise ValueError("Ellipse can't be fitted: No Eigenvalue match all 3 criteria")
    else:
        n = n[0]
    a = A[n]
    b = B[n]
    c = C[n]
    d = D[n]
    f = F[n]
    g = G[n]

    # Calculation of the center:
    denom = b * b - a * c
    x0 = (c * d - b * f) / denom
    y0 = (a * f - b * d) / denom

    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    a2 = up / down1
    b2 = up / down2
    if a2 <= 0 or b2 <= 0:
        raise ValueError("Ellipse can't be fitted, negative sqrt")

    res1 = sqrt(a2)
    res2 = sqrt(b2)

    if a == c:
        angle = 0  # we have a circle
    elif res2 > res1:
        res1, res2 = res2, res1
        angle = 0.5 * (pi + atan2(2 * b, (a - c)))
    else:
        angle = 0.5 * (pi + atan2(2 * b, (a - c)))
    return Ellipse(y0, x0, angle, res1, res2)
