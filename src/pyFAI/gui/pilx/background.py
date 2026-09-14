#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Baseline estimation with asymmetrically reweighted penalized least squares.

Implements the aRPLS algorithm of:

    Baek, S.-J., Park, A., Ahn, Y.-J., Choo, J., "Baseline correction using
    asymmetrically reweighted penalized least squares smoothing", Analyst,
    2015, 140, 250-257. DOI: 10.1039/C4AN01061B.

The estimator is the same one used by ``pybaselines.whittaker.arpls`` but
depends only on numpy and scipy: the penalty matrix``lam * D.T @ D`` of the
second-order difference operator is pentadiagonal and is solved with a
symmetric banded solver, which makes a full map of baselines cheap enough to
compute in the GUI process.
"""

__author__ = "European Synchrotron Radiation Facility"
__license__ = "MIT"
__date__ = "14/09/2026"
__status__ = "development"

import numpy
from scipy import linalg
from scipy.special import expit


def auto_smoothness(size):
    """GSAS-like automatic smoothness.

    :param int size: number of samples of the signal
    :return: exponent of the smoothing parameter (the used lambda is ``10**exponent``)
    """
    return min(10, float(int(10 * numpy.log10(size) ** 1.5) - 9.5) / 10.)


def arpls(y, smoothness=None, mask=None, max_iter=10, tol=1e-3):
    """Estimate the aRPLS baseline of a 1D signal.

    :param y: 1D array of measured values
    :param smoothness: exponent of the smoothing parameter; when None,
        :func:`auto_smoothness` is used (lambda = 10**exponent)
    :param mask: optional boolean array; False entries keep weight zero and
        do not contribute to the fit
    :param max_iter: maximum number of weight updates
    :param tol: relative weight-change convergence criterion
    :return: baseline of the same length as ``y``
    """
    y = numpy.asarray(y, dtype=float)
    if y.size < 3:
        raise ValueError("aRPLS requires at least 3 samples")
    if smoothness is None:
        smoothness = auto_smoothness(y.size)
    lam = 10.0 ** smoothness
    size = y.size

    # lam * D.T @ D for the second-order difference operator D of shape
    # (size-2, size): pentadiagonal, main [1, 5, 6, ..., 6, 5, 1] (for
    # size == 3 the only interior entry is 4), first off-diagonal
    # [-2, -4, ..., -4, -2], second off-diagonal all ones.
    main = numpy.full(size, 6.0)
    main[0] = 1.0
    main[-1] = 1.0
    if size == 3:
        main[1] = 4.0
    else:
        main[1] = 5.0
        main[-2] = 5.0
    sub1 = numpy.full(size - 1, -4.0)
    sub1[0] = -2.0
    sub1[-1] = -2.0
    ab = numpy.zeros((3, size))
    ab[0] = lam * main
    ab[1, : size - 1] = lam * sub1
    ab[2, : size - 2] = lam
    main_diagonal = ab[0].copy()

    if mask is None:
        user_weights = numpy.ones(size)
    else:
        user_weights = numpy.asarray(mask, dtype=float)
        user_weights = (user_weights > 0).astype(float)

    weights = user_weights.copy()
    for _ in range(max_iter + 1):
        ab[0] = main_diagonal + weights
        baseline = linalg.solveh_banded(
            ab, weights * y, lower=True, overwrite_b=True, check_finite=False
        )
        residual = y - baseline
        valid = user_weights > 0
        neg_residual = residual[valid & (residual < 0)]
        if neg_residual.size < 2:
            break
        std = neg_residual.std(ddof=1)
        if std == 0:
            break
        center = 2.0 * std - neg_residual.mean()
        arpls_weights = numpy.ones(size)
        arpls_weights[valid] = expit(
            -(2.0 / std) * (residual[valid] - center)
        )
        new_weights = user_weights * arpls_weights
        if numpy.linalg.norm(new_weights - weights) < tol * numpy.linalg.norm(
            weights
        ):
            weights = new_weights
            break
        weights = new_weights
    return baseline
