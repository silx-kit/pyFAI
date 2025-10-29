#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2025 European Synchrotron Radiation Facility, Grenoble, France
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
Utilities, mainly for image treatment
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2025"
__status__ = "production"

import logging
import math
import numpy
import time
import scipy.ndimage
from scipy.signal import peak_widths

logger = logging.getLogger(__name__)
try:
    from ..ext import relabel as _relabel
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    _relabel = None

EPS32 = 1.0 + numpy.finfo(numpy.float32).eps


def deg2rad(dd: float, disc: bool = True) -> float:
    """
    Convert degrees to radian in the range [-π->π[ or [0->2π[

    :param dd: angle in degrees
    :param disc: position of the discontinuity at π if True, else 2π if False
    :return: angle in radians in the selected range
    """
    # range [0:2pi[
    rp = (dd / 180.0) % 2.0
    if disc:  # range [-pi:pi[
        if rp >= 1.0:
            rp -= 2.0
    return rp * math.pi


def rad2rad(r: float, disc: bool = True):
    """
    Transform radians in the range [-π->π[ or [0->2π[

    :param r: angle in radians
    :param disc: position of the discontinuity at π if True, else 2π if False
    :return: angle in radians in the selected range
    """
    # Set r between (0,2pi)
    r = r % (2 * math.pi)
    if disc:
        if r > math.pi:
            r = r - 2 * math.pi
    return r


def expand2d(vect: numpy.ndarray, size2: int, vertical: bool = True) -> numpy.ndarray:
    """
    This expands a vector to a 2d-array.

    The result is the same as:

    .. code-block:: python

        if vertical:
            numpy.outer(numpy.ones(size2), vect)
        else:
            numpy.outer(vect, numpy.ones(size2))

    This is a ninja optimization: replace \\*1 with a memcopy, saves 50% of
    time at the ms level.

    :param vect: 1d vector
    :param size2: size of the expanded dimension
    :param vertical: if False the vector is expanded to the first dimension.
        If True, it is expanded to the second dimension.
    :return: 2d-array
    """
    size1 = vect.size
    size2 = int(size2)
    if vertical:
        out = numpy.empty((size2, size1), vect.dtype)
        q = vect.reshape(1, -1)
        q.strides = 0, vect.strides[0]
    else:
        out = numpy.empty((size1, size2), vect.dtype)
        q = vect.reshape(-1, 1)
        q.strides = vect.strides[0], 0
    out[:, :] = q
    return out


def gaussian(M: int, std: float) -> numpy.ndarray:
    """
    Return a Gaussian window of length M with standard-deviation std.

    This differs from the scipy.signal.gaussian implementation as:
    - The default for sym=False (needed for gaussian filtering without shift)
    - This implementation is normalized

    :param M: length of the windows (int)
    :param std: standatd deviation sigma

    The FWHM is 2*numpy.sqrt(2 * numpy.pi)*std

    """
    x = numpy.arange(M) - M / 2.0
    return numpy.exp(-((x / std) ** 2) / 2.0) / std / numpy.sqrt(2 * numpy.pi)


def gaussian_filter(
    input_img: numpy.ndarray,
    sigma: float | tuple,
    mode: str = "reflect",
    cval: float = 0.0,
    use_scipy: bool = True,
) -> numpy.ndarray:
    """
    2-dimensional Gaussian filter implemented with FFT

    :param input_img:    input array to filter
    :type input_img: array-like
    :param sigma: standard deviation for Gaussian kernel.
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
    :type sigma: scalar or sequence of scalars
    :param mode: {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    :param cval: scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default is 0.0
    """
    if use_scipy:
        res = scipy.ndimage.gaussian_filter(input_img, sigma, mode=(mode or "reflect"))
    else:
        if isinstance(sigma, (list, tuple)):
            sigma = (float(sigma[0]), float(sigma[1]))
        else:
            sigma = (float(sigma), float(sigma))
        k0 = int(math.ceil(4.0 * float(sigma[0])))
        k1 = int(math.ceil(4.0 * float(sigma[1])))

        if mode != "wrap":
            input_img = expand(input_img, (k0, k1), mode, cval)
        s0, s1 = input_img.shape
        g0 = gaussian(s0, sigma[0])
        g1 = gaussian(s1, sigma[1])
        g0 = numpy.concatenate((g0[s0 // 2 :], g0[: s0 // 2]))  # faster than fftshift
        g1 = numpy.concatenate((g1[s1 // 2 :], g1[: s1 // 2]))  # faster than fftshift
        g2 = numpy.outer(g0, g1)
        fftIn = numpy.fft.ifft2(
            numpy.fft.fft2(input_img) * numpy.fft.fft2(g2).conjugate()
        )
        res = fftIn.real.astype(numpy.float32)
        if mode != "wrap":
            res = res[k0:-k0, k1:-k1]
    return res


def shift(input_img: numpy.ndarray, shift_val: tuple) -> numpy.ndarray:
    """
    Shift an array like  scipy.ndimage.interpolation.shift(input_img, shift_val, mode="wrap", order=0) but faster

    :param input_img: 2d numpy array
    :param shift_val: 2-tuple of integers
    :return: shifted image
    """
    re = numpy.zeros_like(input_img)
    s0, s1 = input_img.shape
    d0 = shift_val[0] % s0
    d1 = shift_val[1] % s1
    r0 = (-d0) % s0
    r1 = (-d1) % s1
    re[d0:, d1:] = input_img[:r0, :r1]
    re[:d0, d1:] = input_img[r0:, :r1]
    re[d0:, :d1] = input_img[:r0, r1:]
    re[:d0, :d1] = input_img[r0:, r1:]
    return re


def dog(s1, s2, shape=None) -> numpy.ndarray:
    """
    2D difference of gaussian
    typically 1 to 10 parameters

    :param s1: width (sigma) of first gaussian
    :param s2: width (sigma) of second gaussian
    :param shape: kernel size, 2-tuple of integers
    """
    if shape is None:
        maxi = max(s1, s2) * 5
        u, v = numpy.ogrid[-maxi : maxi + 1, -maxi : maxi + 1]
    else:
        u, v = numpy.ogrid[
            -shape[0] // 2 : shape[0] - shape[0] // 2,
            -shape[1] // 2 : shape[1] - shape[1] // 2,
        ]
    r2 = u * u + v * v
    centered = (
        numpy.exp(-r2 / (2.0 * s1) ** 2) / 2.0 / numpy.pi / s1
        - numpy.exp(-r2 / (2.0 * s2) ** 2) / 2.0 / numpy.pi / s2
    )
    return centered


def dog_filter(
    input_img: numpy.ndarray,
    sigma1: float,
    sigma2: float,
    mode: str = "reflect",
    cval: float = 0.0,
) -> numpy.ndarray:
    """
    2-dimensional Difference of Gaussian filter implemented with FFT

    :param input_img:    input_img array to filter
    :type input_img: array-like
    :param sigma: standard deviation for Gaussian kernel.
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
    :type sigma: scalar or sequence of scalars
    :param mode: {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    :param cval: scalar, optional
            Value to fill past edges of input if ``mode`` is 'constant'. Default is 0.0
    :return: image filtered with a DoG
    """

    if 1:  # try:
        sigma = max(sigma1, sigma2)
        if mode != "wrap":
            input_img = expand(input_img, sigma, mode, cval)
        s0, s1 = input_img.shape
        if isinstance(sigma, (list, tuple)):
            k0 = int(math.ceil(4.0 * float(sigma[0])))
            k1 = int(math.ceil(4.0 * float(sigma[1])))
        else:
            k0 = k1 = int(math.ceil(4.0 * float(sigma)))

        res = numpy.fft.ifft2(
            numpy.fft.fft2(input_img.astype(complex))
            * numpy.fft.fft2(
                shift(dog(sigma1, sigma2, (s0, s1)), (s0 // 2, s1 // 2)).astype(complex)
            ).conjugate()
        )
        if mode == "wrap":
            return res
        else:
            return res[k0:-k0, k1:-k1]


def expand(
    input_img: numpy.ndarray, sigma: float, mode: str = "constant", cval: float = 0.0
) -> numpy.ndarray:
    """Expand array a with its reflection on boundaries

    :param a: 2D array
    :param sigma: float or 2-tuple of floats.
    :param mode: "constant", "nearest", "reflect" or "mirror"
    :param cval: filling value used for constant, 0.0 by default

    Nota: sigma is the half-width of the kernel. For gaussian convolution it is adviced that it is 4*sigma_of_gaussian
    """
    s0, s1 = input_img.shape
    dtype = input_img.dtype
    if isinstance(sigma, (list, tuple)):
        k0 = int(math.ceil(float(sigma[0])))
        k1 = int(math.ceil(float(sigma[1])))
    else:
        k0 = k1 = int(math.ceil(float(sigma)))
    if k0 > s0 or k1 > s1:
        raise RuntimeError(
            "Makes little sense to apply a kernel (%i,%i)larger than the image (%i,%i)"
            % (k0, k1, s0, s1)
        )
    output = numpy.zeros((s0 + 2 * k0, s1 + 2 * k1), dtype=dtype) + float(cval)
    output[k0 : k0 + s0, k1 : k1 + s1] = input_img
    if mode == "mirror":
        # 4 corners
        output[s0 + k0 :, s1 + k1 :] = input_img[-2 : -k0 - 2 : -1, -2 : -k1 - 2 : -1]
        output[:k0, :k1] = input_img[k0 - 0 : 0 : -1, k1 - 0 : 0 : -1]
        output[:k0, s1 + k1 :] = input_img[k0 - 0 : 0 : -1, s1 - 2 : s1 - k1 - 2 : -1]
        output[s0 + k0 :, :k1] = input_img[s0 - 2 : s0 - k0 - 2 : -1, k1 - 0 : 0 : -1]
        # 4 sides
        output[k0 : k0 + s0, :k1] = input_img[:s0, k1 - 0 : 0 : -1]
        output[:k0, k1 : k1 + s1] = input_img[k0 - 0 : 0 : -1, :s1]
        output[-k0:, k1 : s1 + k1] = input_img[-2 : s0 - k0 - 2 : -1, :]
        output[k0 : s0 + k0, -k1:] = input_img[:, -2 : s1 - k1 - 2 : -1]
    elif mode == "reflect":
        # 4 corners
        output[s0 + k0 :, s1 + k1 :] = input_img[-1 : -k0 - 1 : -1, -1 : -k1 - 1 : -1]
        output[:k0, :k1] = input_img[k0 - 1 :: -1, k1 - 1 :: -1]
        output[:k0, s1 + k1 :] = input_img[k0 - 1 :: -1, s1 - 1 : s1 - k1 - 1 : -1]
        output[s0 + k0 :, :k1] = input_img[s0 - 1 : s0 - k0 - 1 : -1, k1 - 1 :: -1]
        # 4 sides
        output[k0 : k0 + s0, :k1] = input_img[:s0, k1 - 1 :: -1]
        output[:k0, k1 : k1 + s1] = input_img[k0 - 1 :: -1, :s1]
        output[-k0:, k1 : s1 + k1] = input_img[: s0 - k0 - 1 : -1, :]
        output[k0 : s0 + k0, -k1:] = input_img[:, : s1 - k1 - 1 : -1]
    elif mode == "nearest":
        # 4 corners
        output[s0 + k0 :, s1 + k1 :] = input_img[-1, -1]
        output[:k0, :k1] = input_img[0, 0]
        output[:k0, s1 + k1 :] = input_img[0, -1]
        output[s0 + k0 :, :k1] = input_img[-1, 0]
        # 4 sides
        output[k0 : k0 + s0, :k1] = expand2d(input_img[:, 0], k1, False)
        output[:k0, k1 : k1 + s1] = expand2d(input_img[0, :], k0)
        output[-k0:, k1 : s1 + k1] = expand2d(input_img[-1, :], k0)
        output[k0 : s0 + k0, -k1:] = expand2d(input_img[:, -1], k1, False)
    elif mode == "wrap":
        # 4 corners
        output[s0 + k0 :, s1 + k1 :] = input_img[:k0, :k1]
        output[:k0, :k1] = input_img[-k0:, -k1:]
        output[:k0, s1 + k1 :] = input_img[-k0:, :k1]
        output[s0 + k0 :, :k1] = input_img[:k0, -k1:]
        # 4 sides
        output[k0 : k0 + s0, :k1] = input_img[:, -k1:]
        output[:k0, k1 : k1 + s1] = input_img[-k0:, :]
        output[-k0:, k1 : s1 + k1] = input_img[:k0, :]
        output[k0 : s0 + k0, -k1:] = input_img[:, :k1]
    elif mode == "constant":
        # Nothing to do
        pass

    else:
        raise RuntimeError("Unknown expand mode: %s" % mode)
    return output


def relabel(
    label: numpy.ndarray,
    data: numpy.ndarray,
    blured: numpy.ndarray,
    max_size: int = None,
) -> numpy.ndarray:
    """
    Relabel limits the number of region in the label array.
    They are ranked relatively to their max(I0)-max(blur(I0))

    :param label: a label array coming out of ``scipy.ndimage.measurement.label``
    :param data: an array containing the raw data
    :param blured: an array containing the blurred data
    :param max_size: the max number of label wanted
    :return: array like label
    """
    if _relabel:
        max_label = label.max()
        _a, _b, _c, d = _relabel.countThem(label, data, blured)
        count = d
        sortCount = count.argsort()
        invSortCount = sortCount[-1::-1]
        invCutInvSortCount = numpy.zeros(max_label + 1, dtype=int)
        for i, j in enumerate(list(invSortCount[:max_size])):
            invCutInvSortCount[j] = i
        return invCutInvSortCount[label]
    else:
        logger.warning("relabel Cython module is not available...")
        return label


def binning(
    input_img: numpy.ndarray, binsize: int | tuple, norm: bool = True
) -> numpy.ndarray:
    """Perform a 2D binning of the image

    :param input_img: input ndarray
    :param binsize: int or 2-tuple representing the size of the binning
    :param norm: if False, do average instead of sum
    :return: binned input ndarray
    """
    inputSize = input_img.shape
    outputSize = []
    if len(inputSize) != 2:
        raise RuntimeError("input image is not 2D")
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    for i, j in zip(inputSize, binsize):
        if i % j:
            raise RuntimeError(
                "input_img shape should be a multiple of the binning size"
            )
        outputSize.append(i // j)

    if numpy.array(binsize).prod() < 50:
        out = numpy.zeros(tuple(outputSize))
        for i in range(binsize[0]):
            for j in range(binsize[1]):
                out += input_img[i :: binsize[0], j :: binsize[1]]
    else:
        temp = input_img.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    if not norm:
        out /= binsize[0] * binsize[1]
    return out


def unbinning(
    binnedArray: numpy.ndarray, binsize: int | tuple, norm=True
) -> numpy.ndarray:
    """Opposit operation of binning: go from (n,m)->(2n,2m)

    :param binnedArray: input ndarray
    :param binsize: 2-tuple representing the size of the binning
    :param norm: if True (default) decrease the intensity by binning factor. If False, it is non-conservative
    :return: unBinned input ndarray
    """
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    outputShape = []
    for i, j in zip(binnedArray.shape, binsize):
        outputShape.append(i * j)
    out = numpy.zeros(tuple(outputShape), dtype=binnedArray.dtype)
    for i in range(binsize[0]):
        for j in range(binsize[1]):
            out[i :: binsize[0], j :: binsize[1]] += binnedArray
    if norm:
        out /= binsize[0] * binsize[1]
    return out


def shift_fft(input_img: numpy.ndarray, shift_val: tuple, method: str = "fft"):
    """Do shift using FFTs

    Shift an array like  scipy.ndimage.interpolation.shift(input, shift, mode="wrap", order="infinity") but faster
    :param input_img: 2d numpy array
    :param shift_val: 2-tuple of float
    :return: shifted image
    """
    if method == "fft":
        d0, d1 = input_img.shape
        v0, v1 = shift_val
        f0 = numpy.fft.ifftshift(numpy.arange(-d0 // 2, d0 // 2))
        f1 = numpy.fft.ifftshift(numpy.arange(-d1 // 2, d1 // 2))
        m1, m0 = numpy.meshgrid(f1, f0)
        e0 = numpy.exp(-2j * numpy.pi * v0 * m0 / float(d0))
        e1 = numpy.exp(-2j * numpy.pi * v1 * m1 / float(d1))
        e = e0 * e1
        out = abs(numpy.fft.ifft2(numpy.fft.fft2(input_img) * e))
    else:
        out = scipy.ndimage.interpolation.shift(
            input, shift, mode="wrap", order="infinity"
        )
    return out


def maximum_position(img: numpy.ndarray) -> tuple:
    """Find the position of the maximum of the values of the array.
    Same as scipy.ndimage.measurements.maximum_position

    :param img: 2-D image
    :return: 2-tuple of int with the position of the maximum
    """
    maxarg = numpy.argmax(img)
    _, s1 = img.shape
    return (maxarg // s1, maxarg % s1)


def center_of_mass(img: numpy.ndarray) -> tuple:
    """Calculate the center of mass of of the array.
    Like scipy.ndimage.measurements.center_of_mass

    :param img: 2-D array
    :return: 2-tuple of float with the center of mass
    """
    d0, d1 = img.shape
    a0, a1 = numpy.ogrid[:d0, :d1]
    img = img.astype("float64")
    img /= img.sum()
    return ((a0 * img).sum(), (a1 * img).sum())


def measure_offset(
    img1: numpy.ndarray,
    img2: numpy.ndarray,
    method: str = "numpy",
    withLog: bool = False,
    withCorr: bool = False,
) -> tuple:
    """
    Measure the actual offset between 2 images

    :param img1: ndarray, first image
    :param img2: ndarray, second image, same shape as img1
    :param withLog: shall we return logs as well ? boolean
    :return: tuple of floats with the offsets
    """
    method = str(method)
    ################################################################################
    # Start convolutions
    ################################################################################
    shape = img1.shape
    logs = []
    if img2.shape != shape:
        raise RuntimeError("images shape matches")
    t0 = time.perf_counter()
    i1f = numpy.fft.fft2(img1)
    i2f = numpy.fft.fft2(img2)
    res = numpy.fft.ifft2(i1f * i2f.conjugate()).real
    t1 = time.perf_counter()

    ################################################################################
    # END of convolutions
    ################################################################################
    offset1 = maximum_position(res)
    res = shift(res, (shape[0] // 2, shape[1] // 2))
    mean = res.mean(dtype="float64")
    maxi = res.max()
    std = res.std(dtype="float64")
    SN = (maxi - mean) / std
    new = numpy.maximum(
        numpy.zeros(shape), res - numpy.ones(shape) * (mean + std * SN * 0.9)
    )
    com2 = center_of_mass(new)
    logs.append("MeasureOffset: fine result of the centered image: %s %s " % com2)
    offset2 = (
        (com2[0] - shape[0] // 2) % shape[0],
        (com2[1] - shape[1] // 2) % shape[1],
    )
    delta0 = (offset2[0] - offset1[0]) % shape[0]
    delta1 = (offset2[1] - offset1[1]) % shape[1]
    if delta0 > shape[0] // 2:
        delta0 -= shape[0]
    if delta1 > shape[1] // 2:
        delta1 -= shape[1]
    if (abs(delta0) > 2) or (abs(delta1) > 2):
        logs.append(
            "MeasureOffset: Raw offset is %s and refined is %s. Please investigate !"
            % (offset1, offset2)
        )
    listOffset = list(offset2)
    if listOffset[0] > shape[0] // 2:
        listOffset[0] -= shape[0]
    if listOffset[1] > shape[1] // 2:
        listOffset[1] -= shape[1]
    offset = tuple(listOffset)
    t2 = time.perf_counter()
    logs.append("MeasureOffset: fine result: %s %s" % offset)
    logs.append(
        "MeasureOffset: execution time: %.3fs with %.3fs for FFTs" % (t2 - t0, t1 - t0)
    )
    if withLog:
        if withCorr:
            return offset, logs, new
        else:
            return offset, logs
    else:
        if withCorr:
            return offset, new
        else:
            return offset


def _numpy_backport_percentile(a, q, axis=None, out=None, overwrite_input=False):
    """
    Compute the qth percentile of the data along the specified axis.

    Returns the qth percentile of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    q : float in range of [0,100] (or sequence of floats)
        Percentile to compute which must be between 0 and 100 inclusive.
    axis : int, optional
        Axis along which the percentiles are computed. The default (None)
        is to compute the median along a flattened version of the array.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
    overwrite_input : bool, optional
       If True, then allow use of memory of input array `a` for
       calculations. The input array will be modified by the call to
       median. This will save memory when you do not need to preserve
       the contents of the input array. Treat the input as undefined,
       but it will probably be fully or partially sorted.
       Default is False. Note that, if `overwrite_input` is True and the
       input is not already an array, an error will be raised.

    Returns
    -------
    pcntile : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead).  If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64.  Otherwise, the output data-type is the same
        as that of the input.

    See Also
    --------
    mean, median

    Notes
    -----
    Given a vector V of length N, the qth percentile of V is the qth ranked
    value in a sorted copy of V.  A weighted average of the two nearest
    neighbors is used if the normalized ranking does not match q exactly.
    The same as the median if ``q=0.5``, the same as the minimum if ``q=0``
    and the same as the maximum if ``q=1``.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> np.percentile(a, 50)
    3.5
    >>> np.percentile(a, 0.5, axis=0)
    array([ 6.5,  4.5,  2.5])
    >>> np.percentile(a, 50, axis=1)
    array([ 7.,  2.])

    >>> m = np.percentile(a, 50, axis=0)
    >>> out = np.zeros_like(m)
    >>> np.percentile(a, 50, axis=0, out=m)
    array([ 6.5,  4.5,  2.5])
    >>> m
    array([ 6.5,  4.5,  2.5])

    >>> b = a.copy()
    >>> np.percentile(b, 50, axis=1, overwrite_input=True)
    array([ 7.,  2.])
    >>> assert not np.all(a==b)
    >>> b = a.copy()
    >>> np.percentile(b, 50, axis=None, overwrite_input=True)
    3.5

    """
    a = numpy.asarray(a)

    if q == 0:
        return a.min(axis=axis, out=out)
    elif q == 100:
        return a.max(axis=axis, out=out)

    if overwrite_input:
        if axis is None:
            sorted_list = a.ravel()
            sorted_list.sort()
        else:
            a.sort(axis=axis)
            sorted_list = a
    else:
        sorted_list = numpy.sort(a, axis=axis)
    if axis is None:
        axis = 0

    return _compute_qth_percentile(sorted_list, q, axis, out)


def _compute_qth_percentile(sorted_list, q, axis, out):
    """
    Handle sequence of q's without calling sort multiple times
    """
    if not numpy.isscalar(q):
        p = [_compute_qth_percentile(sorted_list, qi, axis, None) for qi in q]

        if out is not None:
            out.flat = p

        return p

    q = q / 100.0
    if (q < 0) or (q > 1):
        raise ValueError("percentile must be either in the range [0,100]")

    indexer = [slice(None)] * sorted_list.ndim
    Nx = sorted_list.shape[axis]
    index = q * (Nx - 1)
    i = int(index)
    if i == index:
        indexer[axis] = slice(i, i + 1)
        weights = numpy.array(1)
        sumval = 1.0
    else:
        indexer[axis] = slice(i, i + 2)
        j = i + 1
        weights = numpy.array([(j - index), (index - i)], float)
        wshape = [1] * sorted_list.ndim
        wshape[axis] = 2
        weights.shape = wshape
        sumval = weights.sum()

    # Use add.reduce in both cases to coerce data type as well as
    #   check and use out array.
    return numpy.add.reduce(sorted_list[indexer] * weights, axis=axis, out=out) / sumval


try:
    from numpy import percentile
except ImportError:
    # backport percentile from numpy 1.6.2
    logger.debug("Backtrace", exc_info=True)
    percentile = _numpy_backport_percentile


def round_fft(N: int) -> int:
    """
    This function returns the integer >=N for which size the Fourier analysis is faster (fron the FFT point of view)

    Credit: Alessandro Mirone, ESRF, 2012

    :param N: interger on which one would like to do a Fourier transform
    :return: integer with a better choice
    """
    FA, FB, FC, FD, FE, FFF = 2, 3, 5, 7, 11, 13
    DIFF = 9999999999
    RES = 1
    AA = 1
    for _ in range(int(math.log(N) / math.log(FA) + 2)):
        BB = AA
        for _ in range(int(math.log(N) / math.log(FB) + 2)):
            CC = BB

            for _ in range(int(math.log(N) / math.log(FC) + 2)):
                DD = CC

                for _ in range(int(math.log(N) / math.log(FD) + 2)):
                    EE = DD

                    for E in range(2):
                        FF = EE

                        for _ in range(2 - E):
                            if FF >= N and DIFF > abs(N - FF):
                                DIFF = abs(N - FF)
                                RES = FF
                            if FF > N:
                                break
                            FF = FF * FFF
                        if EE > N:
                            break
                        EE = EE * FE
                    if DD > N:
                        break
                    DD = DD * FD
                if CC > N:
                    break
                CC = CC * FC
            if BB > N:
                break
            BB = BB * FB
        if AA > N:
            break
        AA = AA * FA
    return RES


def is_far_from_group_python(pt: list | tuple, lst_pts: list, d2: float) -> bool:
    """
    Tells if a point is far from a group of points, distance greater than d2 (distance squared)

    :param pt: point of interest
    :param lst_pts: list of points
    :param d2: minimum distance squarred
    :return: True If the point is far from all others.

    """
    for apt in lst_pts:
        dsq = sum((i - j) * (i - j) for i, j in zip(apt, pt))
        if dsq <= d2:
            return False
    return True


try:
    from ..ext.mathutil import is_far_from_group_cython
except ImportError:
    is_far_from_group = is_far_from_group_python
else:
    is_far_from_group = is_far_from_group_cython


def rwp(obt: list | tuple, ref: list | tuple, scale: float = 1.0) -> float:
    """Compute :math:`\\sqrt{\\sum \\frac{4\\cdot(obt-ref)^2}{(obt + ref)^2}}`.

    This is done for symmetry reason between obt and ref

    :param obt: obtained data
    :type obt: 2-list of array of the same size
    :param obt: reference data
    :type obt: 2-list of array of the same size
    :param scale: scale obt intensity
    :return:  Rwp value, lineary interpolated
    """
    ref0, ref1 = ref[:2]
    obt0, obt1 = obt[:2]
    obt1 = obt1 * scale
    big0 = numpy.concatenate((obt0, ref0))
    big0.sort()
    big0 = numpy.unique(big0)
    big_ref = numpy.interp(big0, ref0, ref1, 0.0, 0.0)
    big_obt = numpy.interp(big0, obt0, obt1, 0.0, 0.0)
    big_mean = (big_ref + big_obt) / 2.0
    big_delta = big_ref - big_obt
    non_null = abs(big_mean) > 1e-10
    return numpy.sqrt(((big_delta[non_null]) ** 2 / ((big_mean[non_null]) ** 2)).sum())


def chi_square(obt: list | tuple, ref: list | tuple) -> float:
    """Compute :math:`\\sqrt{\\sum \\frac{4\\cdot(obt-ref)^2}{(obt + ref)^2}}`.

    This is done for symmetry reason between obt and ref

    :param obt: obtained data
    :type obt: 3-tuple of array of the same size containing position, intensity, variance
    :param obt: reference data
    :type obt: 3-tuple of array of the same size containing position, intensity, variance
    :return:  Chi² value, lineary interpolated
    """
    ref_pos, ref_int, ref_std = ref
    obt_pos, obt_int, obt_std = obt
    big_pos = numpy.concatenate((ref_pos, obt_pos))
    big_pos.sort()
    big_pos = numpy.unique(big_pos)
    big_ref_int = numpy.interp(big_pos, ref_pos, ref_int, 0.0, 0.0)
    big_obt_int = numpy.interp(big_pos, obt_pos, obt_int, 0.0, 0.0)
    big_delta_int = big_ref_int - big_obt_int

    big_ref_var = numpy.interp(big_pos, ref_pos, ref_std, 0.0, 0.0) ** 2
    big_obt_var = numpy.interp(big_pos, obt_pos, obt_std, 0.0, 0.0) ** 2
    big_variance = (big_ref_var + big_obt_var) / 2.0
    non_null = abs(big_variance) > 1e-10
    return (big_delta_int[non_null] ** 2 / big_variance[non_null]).mean()


class LongestRunOfHeads:
    """Implements the "longest run of heads" by Mark F. Schilling
    The College Mathematics Journal, Vol. 21, No. 3, (1990), pp. 196-207

    See: http://www.maa.org/sites/default/files/pdf/upload_library/22/Polya/07468342.di020742.02p0021g.pdf
    """

    def __init__(self):
        "We store already calculated values for (n,c)"
        self.knowledge = {}

    def A(self, n: int, c: int):
        """Calculate A(number_of_toss, length_of_longest_run)

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of
        :return: The A parameter used in the formula

        """
        if n <= c:
            return 2**n
        elif (n, c) in self.knowledge:
            return self.knowledge[(n, c)]
        else:
            s = 0
            for j in range(c, -1, -1):
                s += self.A(n - 1 - j, c)
            self.knowledge[(n, c)] = s
            return s

    def B(self, n: int, c: int) -> int:
        """Calculate B(number_of_toss, length_of_longest_run)
        to have either a run of Heads either a run of Tails

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of
        :return: The B parameter used in the formula
        """
        return 2 * self.A(n - 1, c - 1)

    def __call__(self, n: int, c: int) -> float:
        """Calculate the probability for the longest run of heads to exceed the observed length

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads, an integer
        :return: The probablility of having c subsequent heads in a n toss of fair coin
        """
        if c >= n:
            return 0
        delta = 2**n - self.A(n, c)
        if delta <= 0:
            return 0
        return 2.0 ** (math.log(delta, 2) - n)

    def probaHeadOrTail(self, n: int, c: int) -> float:
        """Calculate the probability of a longest run of head or tails to occur

        :param n: number of coin toss in the experiment, an integer
        :param c: length of the longest run of heads or tails, an integer
        :return: The probablility of having c subsequent heads or tails in a n toss of fair coin
        """
        if c > n:
            return 0
        if c == 0:
            return 0
        delta = self.B(n, c) - self.B(n, c - 1)
        if delta <= 0:
            return 0
        return min(2.0 ** (math.log(delta, 2.0) - n), 1.0)

    def probaLongerRun(self, n: int, c: int) -> float:
        """Calculate the probability for the longest run of heads or tails to exceed the observed length

        :param n: number of coin toss in the experiment, an integer
        :param c: length of thee observed run of heads or tails, an integer
        :return: The probablility of having more than c subsequent heads or tails in a n toss of fair coin
        """
        if c > n:
            return 0
        if c == 0:
            return 0
        delta = (2**n) - self.B(n, c)
        if delta <= 0:
            return 0
        return min(2.0 ** (math.log(delta, 2.0) - n), 1.0)


LROH = LongestRunOfHeads()


def _longest_true(a: numpy.ndarray) -> int:
    """measure longest section of only "true" in a binary array"""
    # Convert to array
    a = numpy.asarray(a)

    # Attach sentients on either sides w.r.t True
    b = numpy.r_[False, a, False]

    # Get indices of group shifts
    s = numpy.flatnonzero(b[:-1] != b[1:])
    if len(s):
        # Get group lengths and hence the max index group
        m = (s[1::2] - s[::2]).argmax()
        return s[2 * m + 1] - s[2 * m]
    else:
        return 0


def cormap(ref: numpy.ndarray, obt: numpy.ndarray) -> float:
    """Calculate the probabily of two array to be the same based on the CorMap algorithm
    This is a simplifed implementation
    """
    longest = max(_longest_true(ref < obt), _longest_true(ref > obt))
    return LROH.probaLongerRun(len(ref), max(1, longest - 1))


def interp_filter(
    ary: numpy.ndarray, out: numpy.ndarray | None = None
) -> numpy.ndarray:
    """Interpolate missing values (nan or infinite) in a 1D array

    :param ary: 1D array
    :param out: destination array (use ary to avoid allocation)
    :return: 1D array
    """
    x = numpy.arange(ary.shape[0])
    mask_valid = numpy.isfinite(ary)
    mask_invalid = numpy.logical_not(mask_valid)
    where = numpy.where(mask_valid)[0]
    first = ary[where[0]]
    last = ary[where[-1]]
    if out is None:
        out = ary.copy()
    elif id(out) == id(ary):
        pass
    else:
        out[mask_valid] = ary[mask_valid]
    out[mask_invalid] = numpy.interp(
        x[mask_invalid], x[mask_valid], ary[mask_valid], left=first, right=last
    )
    return out


def allclose_mod(
    a: numpy.ndarray, b: numpy.ndarray, modulo: float = 2 * numpy.pi, **kwargs
):
    """Returns True if the two arrays a & b are equal within the given
    tolerance modulo `modulo`; False otherwise.

    Thanks to "Serguei Sokol" <sokol@insa-toulouse.fr>
    """
    di = numpy.minimum((a - b) % modulo, (b - a) % modulo)
    return numpy.allclose(modulo * 0.5, (di + modulo * 0.5), **kwargs)


def quality_of_fit(
    img: numpy.ndarray,
    ai,
    calibrant,
    npt_rad: int = 1000,
    npt_azim: int = 360,
    unit="q_nm^-1",
    method: tuple = ("full", "csr", "cython"),
    empty: float = numpy.nan,
    rings: list | None = None,
):
    """Provide an indicator for the quality of fit of a given geometry for an image

    :param img: 2D image with a calibration image (containing rings)
    :param ai: azimuthal integrator object (instance of pyFAI.integrator.azimuthal.AzimuthalIntegrator)
    :param calibrant: calibration object, instance of pyFAI.calibrant.Calibrant
    :param npt_rad: int with the number of radial bins
    :param npt_azim: int with the number of azimuthal bins
    :param unit: typically "2th_deg" or "q_nm^-1", the quality of fit should be largely independant from the space.
    :param method: integration method
    :param empty: value of the empy bins, discarded values
    :param rings: list of rings to evaluate (0-based)
    :return: QoF indicator, similar to reduced χ²,  the smaller, the better
    """

    ai.empty = empty
    q_theo = calibrant.get_peaks(unit=unit)
    res = ai.integrate2d(img, npt_rad, npt_azim, method=method, unit=unit)
    if rings is None:
        rings = list(range(len(calibrant.get_2th())))
    q_theo = q_theo[rings]
    idx_theo = abs(numpy.add.outer(res.radial, -q_theo)).argmin(axis=0)
    idx_maxi = numpy.empty((res.azimuthal.size, len(rings))) + numpy.nan
    idx_fwhm = numpy.empty((res.azimuthal.size, len(rings))) + numpy.nan
    signal = res.intensity
    gradient = numpy.gradient(signal, axis=-1)
    minima = numpy.where(numpy.logical_and(gradient[:, :-1] < 0, gradient[:, 1:] >= 0))
    maxima = numpy.where(numpy.logical_and(gradient[:, :-1] > 0, gradient[:, 1:] < 0))
    for idx in range(res.azimuthal.size):
        for ring in range(len(rings)):
            q_th = q_theo[ring]
            idx_th = idx_theo[ring]
            if (q_th <= res.radial[0]) or (q_th >= res.radial[-1]):
                continue
            maxi = maxima[1][maxima[0] == idx]
            mini = minima[1][minima[0] == idx]
            idx_max = maxi[abs(maxi - idx_th).argmin()]
            idx_inf = mini[mini < idx_max]
            if idx_inf.size:
                idx_inf = idx_inf[-1]
                idx_sup = mini[mini > idx_max]
                if idx_sup.size:
                    idx_sup = idx_sup[0]
                    if idx_inf < idx_th < idx_sup:
                        sub = signal[idx, idx_inf : idx_sup + 1] - numpy.linspace(
                            signal[idx, idx_inf],
                            signal[idx, idx_sup],
                            1 + idx_sup - idx_inf,
                        )
                        com = (
                            sub
                            * numpy.linspace(idx_inf, idx_sup, 1 + idx_sup - idx_inf)
                        ).sum() / sub.sum()
                        if numpy.isfinite(com):
                            width = peak_widths(sub, [numpy.argmax(sub)])[0][0]
                            if width == 0:
                                print(
                                    f" #{idx}, {ring}: {idx_inf} < th:{idx_th} max:{idx_max} com:{com:.3f} < {idx_sup}; fwhm={width}"
                                )
                                print(signal[idx, idx_inf : idx_sup + 1])
                                # print(sub)
                            else:
                                idx_fwhm[idx, ring] = width
                                idx_maxi[idx, ring] = idx_max
    return numpy.nanmean((2.355 * (idx_maxi - idx_theo) / idx_fwhm) ** 2)


def nan_equal(a: float, b: float) -> bool:
    """return True if a==b, also if a and b are both NaNs"""
    if a == b:
        return True
    return numpy.isnan(a) and numpy.isnan(b)
