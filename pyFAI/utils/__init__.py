#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/06/2017"
__status__ = "production"

import logging
import sys
import os
import glob
import threading
import math
sem = threading.Semaphore()  # global lock for image processing initialization
import numpy
import fabio

from .._version import calc_hexversion
if ("hexversion" in dir(fabio)) and (fabio.hexversion >= calc_hexversion(0, 2, 2)):
    from fabio.nexus import exists
else:
    from os.path import exists

from math import ceil, pi
try:
    from ..third_party import six
except (ImportError, Exception):
    import six
try:
    from ..ext import relabel as _relabel
except ImportError:
    _relabel = None
try:
    from ..directories import data_dir
except ImportError:
    data_dir = None

import scipy.ndimage.filters
logger = logging.getLogger("pyFAI.utils")
import time

if sys.platform != "win32":
    WindowsError = RuntimeError

win32 = (os.name == "nt") and (tuple.__itemsize__ == 4)

EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)

StringTypes = (six.binary_type, six.text_type)

try:
    from ..ext.fastcrc import crc32
except:
    from zlib import crc32


def calc_checksum(ary, safe=True):
    """
    Calculate the checksum by default (or returns its buffer location if unsafe)
    """
    if safe:
        return crc32(ary)
    else:
        return ary.__array_interface__['data'][0]


def float_(val):
    """
    Convert anything to a float ... or None if not applicable
    """
    try:
        f = float(str(val).strip())
    except ValueError:
        f = None
    return f


def int_(val):
    """
    Convert anything to an int ... or None if not applicable
    """
    try:
        f = int(str(val).strip())
    except ValueError:
        f = None
    return f


def str_(val):
    """
    Convert anything to a string ... but None -> ""
    """
    s = ""
    if val is not None:
        try:
            s = str(val)
        except UnicodeError:
            # Python2 specific...
            s = unicode(val)
    return s


def expand2d(vect, size2, vertical=True):
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


def gaussian(M, std):
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
    return numpy.exp(-(x / std) ** 2 / 2.0) / std / numpy.sqrt(2 * numpy.pi)


def gaussian_filter(input_img, sigma, mode="reflect", cval=0.0, use_scipy=True):
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
        res = scipy.ndimage.filters.gaussian_filter(input_img, sigma, mode=(mode or "reflect"))
    else:
        if isinstance(sigma, (list, tuple)):
            sigma = (float(sigma[0]), float(sigma[1]))
        else:
            sigma = (float(sigma), float(sigma))
        k0 = int(ceil(4.0 * float(sigma[0])))
        k1 = int(ceil(4.0 * float(sigma[1])))

        if mode != "wrap":
            input_img = expand(input_img, (k0, k1), mode, cval)
        s0, s1 = input_img.shape
        g0 = gaussian(s0, sigma[0])
        g1 = gaussian(s1, sigma[1])
        g0 = numpy.concatenate((g0[s0 // 2:], g0[:s0 // 2]))  # faster than fftshift
        g1 = numpy.concatenate((g1[s1 // 2:], g1[:s1 // 2]))  # faster than fftshift
        g2 = numpy.outer(g0, g1)
        fftIn = numpy.fft.ifft2(numpy.fft.fft2(input_img) * numpy.fft.fft2(g2).conjugate())
        res = fftIn.real.astype(numpy.float32)
        if mode != "wrap":
            res = res[k0:-k0, k1:-k1]
    return res


def shift(input_img, shift_val):
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


def dog(s1, s2, shape=None):
    """
    2D difference of gaussian
    typically 1 to 10 parameters
    """
    if shape is None:
        maxi = max(s1, s2) * 5
        u, v = numpy.ogrid[-maxi:maxi + 1, -maxi:maxi + 1]
    else:
        u, v = numpy.ogrid[-shape[0] // 2:shape[0] - shape[0] // 2, -shape[1] // 2:shape[1] - shape[1] // 2]
    r2 = u * u + v * v
    centered = numpy.exp(-r2 / (2. * s1) ** 2) / 2. / numpy.pi / s1 - numpy.exp(-r2 / (2. * s2) ** 2) / 2. / numpy.pi / s2
    return centered


def dog_filter(input_img, sigma1, sigma2, mode="reflect", cval=0.0):
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
    """

    if 1:  # try:
        sigma = max(sigma1, sigma2)
        if mode != "wrap":
            input_img = expand(input_img, sigma, mode, cval)
        s0, s1 = input_img.shape
        if isinstance(sigma, (list, tuple)):
            k0 = int(ceil(4.0 * float(sigma[0])))
            k1 = int(ceil(4.0 * float(sigma[1])))
        else:
            k0 = k1 = int(ceil(4.0 * float(sigma)))

        res = numpy.fft.ifft2(numpy.fft.fft2(input_img.astype(complex)) *
                              numpy.fft.fft2(shift(dog(sigma1, sigma2, (s0, s1)),
                                                   (s0 // 2, s1 // 2)).astype(complex)).conjugate())
        if mode == "wrap":
            return res
        else:
            return res[k0:-k0, k1:-k1]


def expand(input_img, sigma, mode="constant", cval=0.0):
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
        k0 = int(ceil(float(sigma[0])))
        k1 = int(ceil(float(sigma[1])))
    else:
        k0 = k1 = int(ceil(float(sigma)))
    if k0 > s0 or k1 > s1:
        raise RuntimeError("Makes little sense to apply a kernel (%i,%i)larger than the image (%i,%i)" % (k0, k1, s0, s1))
    output = numpy.zeros((s0 + 2 * k0, s1 + 2 * k1), dtype=dtype) + float(cval)
    output[k0:k0 + s0, k1:k1 + s1] = input_img
    if (mode == "mirror"):
        # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[-2:-k0 - 2:-1, -2:-k1 - 2:-1]
        output[:k0, :k1] = input_img[k0 - 0:0:-1, k1 - 0:0:-1]
        output[:k0, s1 + k1:] = input_img[k0 - 0:0:-1, s1 - 2: s1 - k1 - 2:-1]
        output[s0 + k0:, :k1] = input_img[s0 - 2: s0 - k0 - 2:-1, k1 - 0:0:-1]
        # 4 sides
        output[k0:k0 + s0, :k1] = input_img[:s0, k1 - 0:0:-1]
        output[:k0, k1:k1 + s1] = input_img[k0 - 0:0:-1, :s1]
        output[-k0:, k1:s1 + k1] = input_img[-2:s0 - k0 - 2:-1, :]
        output[k0:s0 + k0, -k1:] = input_img[:, -2:s1 - k1 - 2:-1]
    elif mode == "reflect":
        # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[-1:-k0 - 1:-1, -1:-k1 - 1:-1]
        output[:k0, :k1] = input_img[k0 - 1::-1, k1 - 1::-1]
        output[:k0, s1 + k1:] = input_img[k0 - 1::-1, s1 - 1: s1 - k1 - 1:-1]
        output[s0 + k0:, :k1] = input_img[s0 - 1: s0 - k0 - 1:-1, k1 - 1::-1]
    # 4 sides
        output[k0:k0 + s0, :k1] = input_img[:s0, k1 - 1::-1]
        output[:k0, k1:k1 + s1] = input_img[k0 - 1::-1, :s1]
        output[-k0:, k1:s1 + k1] = input_img[:s0 - k0 - 1:-1, :]
        output[k0:s0 + k0, -k1:] = input_img[:, :s1 - k1 - 1:-1]
    elif mode == "nearest":
    # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[-1, -1]
        output[:k0, :k1] = input_img[0, 0]
        output[:k0, s1 + k1:] = input_img[0, -1]
        output[s0 + k0:, :k1] = input_img[-1, 0]
    # 4 sides
        output[k0:k0 + s0, :k1] = expand2d(input_img[:, 0], k1, False)
        output[:k0, k1:k1 + s1] = expand2d(input_img[0, :], k0)
        output[-k0:, k1:s1 + k1] = expand2d(input_img[-1, :], k0)
        output[k0:s0 + k0, -k1:] = expand2d(input_img[:, -1], k1, False)
    elif mode == "wrap":
        # 4 corners
        output[s0 + k0:, s1 + k1:] = input_img[:k0, :k1]
        output[:k0, :k1] = input_img[-k0:, -k1:]
        output[:k0, s1 + k1:] = input_img[-k0:, :k1]
        output[s0 + k0:, :k1] = input_img[:k0, -k1:]
        # 4 sides
        output[k0:k0 + s0, :k1] = input_img[:, -k1:]
        output[:k0, k1:k1 + s1] = input_img[-k0:, :]
        output[-k0:, k1:s1 + k1] = input_img[:k0, :]
        output[k0:s0 + k0, -k1:] = input_img[:, :k1]
    elif mode == "constant":
        # Nothing to do
        pass

    else:
        raise RuntimeError("Unknown expand mode: %s" % mode)
    return output


def relabel(label, data, blured, max_size=None):
    """
    Relabel limits the number of region in the label array.
    They are ranked relatively to their max(I0)-max(blur(I0)

    :param label: a label array coming out of ``scipy.ndimage.measurement.label``
    :param data: an array containing the raw data
    :param blured: an array containing the blurred data
    :param max_size: the max number of label wanted
    :return: array like label
    """
    if _relabel:
        max_label = label.max()
        a, b, c, d = _relabel.countThem(label, data, blured)
        count = d
        sortCount = count.argsort()
        invSortCount = sortCount[-1::-1]
        invCutInvSortCount = numpy.zeros(max_label + 1, dtype=int)
        for i, j in enumerate(list(invSortCount[:max_size])):
            invCutInvSortCount[j] = i
        f = lambda i:invCutInvSortCount[i]
        return f(label)
    else:
        logger.warning("relabel Cython module is not available...")
        return label


def binning(input_img, binsize, norm=True):
    """
    :param input_img: input ndarray
    :param binsize: int or 2-tuple representing the size of the binning
    :param norm: if False, do average instead of sum
    :return: binned input ndarray
    """
    inputSize = input_img.shape
    outputSize = []
    assert(len(inputSize) == 2)
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    for i, j in zip(inputSize, binsize):
        assert(i % j == 0)
        outputSize.append(i // j)

    if numpy.array(binsize).prod() < 50:
        out = numpy.zeros(tuple(outputSize))
        for i in range(binsize[0]):
            for j in range(binsize[1]):
                out += input_img[i::binsize[0], j::binsize[1]]
    else:
        temp = input_img.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    if not norm:
        out /= binsize[0] * binsize[1]
    return out


def unBinning(binnedArray, binsize, norm=True):
    """
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
            out[i::binsize[0], j::binsize[1]] += binnedArray
    if norm:
        out /= binsize[0] * binsize[1]
    return out


def shiftFFT(input_img, shift_val, method="fft"):
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
        out = scipy.ndimage.interpolation.shift(input, shift, mode="wrap", order="infinity")
    return out


def maximum_position(img):
    """
    Same as scipy.ndimage.measurements.maximum_position:
    Find the position of the maximum of the values of the array.

    :param img: 2-D image
    :return: 2-tuple of int with the position of the maximum
    """
    maxarg = numpy.argmax(img)
    _, s1 = img.shape
    return (maxarg // s1, maxarg % s1)


def center_of_mass(img):
    """
    Calculate the center of mass of of the array.
    Like scipy.ndimage.measurements.center_of_mass
    :param img: 2-D array
    :return: 2-tuple of float with the center of mass
    """
    d0, d1 = img.shape
    a0, a1 = numpy.ogrid[:d0, :d1]
    img = img.astype("float64")
    img /= img.sum()
    return ((a0 * img).sum(), (a1 * img).sum())


def measure_offset(img1, img2, method="numpy", withLog=False, withCorr=False):
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
    assert img2.shape == shape
    t0 = time.time()
    i1f = numpy.fft.fft2(img1)
    i2f = numpy.fft.fft2(img2)
    res = numpy.fft.ifft2(i1f * i2f.conjugate()).real
    t1 = time.time()

    ################################################################################
    # END of convolutions
    ################################################################################
    offset1 = maximum_position(res)
    res = shift(res, (shape[0] // 2, shape[1] // 2))
    mean = res.mean(dtype="float64")
    maxi = res.max()
    std = res.std(dtype="float64")
    SN = (maxi - mean) / std
    new = numpy.maximum(numpy.zeros(shape), res - numpy.ones(shape) * (mean + std * SN * 0.9))
    com2 = center_of_mass(new)
    logs.append("MeasureOffset: fine result of the centered image: %s %s " % com2)
    offset2 = ((com2[0] - shape[0] // 2) % shape[0], (com2[1] - shape[1] // 2) % shape[1])
    delta0 = (offset2[0] - offset1[0]) % shape[0]
    delta1 = (offset2[1] - offset1[1]) % shape[1]
    if delta0 > shape[0] // 2:
        delta0 -= shape[0]
    if delta1 > shape[1] // 2:
        delta1 -= shape[1]
    if (abs(delta0) > 2) or (abs(delta1) > 2):
        logs.append("MeasureOffset: Raw offset is %s and refined is %s. Please investigate !" % (offset1, offset2))
    listOffset = list(offset2)
    if listOffset[0] > shape[0] // 2:
        listOffset[0] -= shape[0]
    if listOffset[1] > shape[1] // 2:
        listOffset[1] -= shape[1]
    offset = tuple(listOffset)
    t2 = time.time()
    logs.append("MeasureOffset: fine result: %s %s" % offset)
    logs.append("MeasureOffset: execution time: %.3fs with %.3fs for FFTs" % (t2 - t0, t1 - t0))
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


def expand_args(args):
    """
    Takes an argv and expand it (under Windows, cmd does not convert ``*.tif``
    into a list of files.
    Keeps only valid files (thanks to glob)

    :param args: list of files or wilcards
    :return: list of actual args
    """
    new = []
    for afile in args:
        if exists(afile):
            new.append(afile)
        else:
            new += glob.glob(afile)
    return new


def _get_data_path(filename):
    """
    :param filename: the name of the requested data file.
    :type filename: str

    Can search root of data directory in:
    - Environment variable PYFAI_DATA
    - path hard coded into pyFAI.directories.data_dir
    - where this file is installed.

    In the future ....
    This method try to find the requested ui-name following the
    xfreedesktop recommendations. First the source directory then
    the system locations

    For now, just perform a recursive search
    """
    resources = [
        os.environ.get("PYFAI_DATA"),
        data_dir,
        os.path.join(os.path.dirname(__file__), "..")]
    try:
        import xdg.BaseDirectory
        resources += xdg.BaseDirectory.load_data_paths("pyFAI")
    except ImportError:
        pass

    for resource in resources:
        if not resource:
            continue
        real_filename = os.path.join(resource, "resources", filename)
        if os.path.exists(real_filename):
            return real_filename
    else:
        raise RuntimeError("Can not find the [%s] resource, "
                           "something went wrong !!!" % (real_filename,))


def get_calibration_dir():
    """get the full path of a calibration directory

    :return: the full path of the calibrant file
    """
    return _get_data_path("calibration")


def get_cl_file(filename):
    """get the full path of a openCL file

    :return: the full path of the openCL source file
    """
    if not filename.endswith(".cl"):
        filename += ".cl"
    return _get_data_path(os.path.join("openCL", filename))


def get_ui_file(filename):
    """get the full path of a user-interface file

    :return: the full path of the ui
    """
    return _get_data_path(os.path.join("gui", filename))


def deg2rad(dd):
    """
    Convert degrees to radian in the range -pi->pi

    :param dd: angle in degrees

    Nota: depending on the platform it could be 0<2pi
    A branch is cheaper than a trigo operation
    """
    while dd > 180.0:
        dd -= 360.0
    while dd < -180.0:
        dd += 360.0
    return dd * pi / 180.


class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.func_name if sys.version_info[0] < 3 else fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value

try:
    from numpy import percentile
except ImportError:  # backport percentile from numpy 1.6.2
    np = numpy
    def percentile(a, q, axis=None, out=None, overwrite_input=False):
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
        a = np.asarray(a)

        if q == 0:
            return a.min(axis=axis, out=out)
        elif q == 100:
            return a.max(axis=axis, out=out)

        if overwrite_input:
            if axis is None:
                sorted = a.ravel()
                sorted.sort()
            else:
                a.sort(axis=axis)
                sorted = a
        else:
            sorted = np.sort(a, axis=axis)
        if axis is None:
            axis = 0

        return _compute_qth_percentile(sorted, q, axis, out)

    # handle sequence of q's without calling sort multiple times
    def _compute_qth_percentile(sorted, q, axis, out):
        if not np.isscalar(q):
            p = [_compute_qth_percentile(sorted, qi, axis, None)
                 for qi in q]

            if out is not None:
                out.flat = p

            return p

        q = q / 100.0
        if (q < 0) or (q > 1):
            raise ValueError("percentile must be either in the range [0,100]")

        indexer = [slice(None)] * sorted.ndim
        Nx = sorted.shape[axis]
        index = q * (Nx - 1)
        i = int(index)
        if i == index:
            indexer[axis] = slice(i, i + 1)
            weights = np.array(1)
            sumval = 1.0
        else:
            indexer[axis] = slice(i, i + 2)
            j = i + 1
            weights = np.array([(j - index), (index - i)], float)
            wshape = [1] * sorted.ndim
            wshape[axis] = 2
            weights.shape = wshape
            sumval = weights.sum()

        # Use add.reduce in both cases to coerce data type as well as
        #   check and use out array.
        return np.add.reduce(sorted[indexer] * weights, axis=axis, out=out) / sumval


def convert_CamelCase(name):
    """
    convert a function name in CamelCase into camel_case
    """
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def readFloatFromKeyboard(text, dictVar):
    """
    Read float from the keyboard ....

    :param text: string to be displayed
    :param dictVar: dict of this type: {1: [set_dist_min],3: [set_dist_min, set_dist_guess, set_dist_max]}
    """
    fromkb = six.moves.input(text).strip()
    try:
        vals = [float(i) for i in fromkb.split()]
    except:
        logging.error("Error in parsing values")
    else:
        found = False
        for i in dictVar:
            if len(vals) == i:
                found = True
                for j in range(i):
                    dictVar[i][j](vals[j])
        if not found:
            logger.error("You should provide the good number of floats")


class FixedParameters(set):
    """
    Like a set, made for FixedParameters in geometry refinement
    """

    def add_or_discard(self, key, value=True):
        """
        Add a value to a set if value, else discard it
        :param key: element to added or discared from set
        :type value: boolean. If None do nothing !
        :return: None
        """
        if value is None:
            return
        if value:
            self.add(key)
        else:
            self.discard(key)


def roundfft(N):
    """
    This function returns the integer >=N for which size the Fourier analysis is faster (fron the FFT point of view)
    Credit: Alessandro Mirone, ESRF, 2012

    :param N: interger on which one would like to do a Fourier transform
    :return: integer with a better choice
    """
    MA, MB, MC, MD, ME, MF = 0, 0, 0, 0, 0, 0
    FA, FB, FC, FD, FE, FFF = 2, 3, 5, 7, 11, 13
    DIFF = 9999999999
    RES = 1
    R0 = 1
    AA = 1
    for A in range(int(math.log(N) / math.log(FA) + 2)):
        BB = AA
        for B in range(int(math.log(N) / math.log(FB) + 2)):
            CC = BB

            for C in range(int(math.log(N) / math.log(FC) + 2)):
                DD = CC

                for D in range(int(math.log(N) / math.log(FD) + 2)):
                    EE = DD

                    for E in range(2):
                        FF = EE

                        for F in range(2 - E):
                            if FF >= N and DIFF > abs(N - FF):
                                MA, MB, MC, MD, ME, MF = A, B, C, D, E, F
                                DIFF = abs(N - FF)
                                RES = FF
                            if FF > N: break
                            FF = FF * FFF
                        if EE > N: break
                        EE = EE * FE
                    if DD > N: break
                    DD = DD * FD
                if CC > N: break
                CC = CC * FC
            if BB > N: break
            BB = BB * FB
        if AA > N: break
        AA = AA * FA
    return RES


def is_far_from_group(pt, lst_pts, d2):
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


def fully_qualified_name(obj):
    "Return the fully qualified name of an object"
    actual_class = obj.__class__.__mro__[0]
    return actual_class.__module__ + "." + actual_class.__name__
