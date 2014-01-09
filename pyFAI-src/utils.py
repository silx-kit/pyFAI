#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""

Utilities, mainly for image treatment

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/07/2013"
__status__ = "development"

import logging, sys, types, os, glob
import threading
sem = threading.Semaphore()  # global lock for image processing initialization
import numpy
import fabio
from scipy import ndimage
from scipy.interpolate import interp1d
from math import  ceil, sin, cos, atan2, pi

try:
    from . import relabel as relabelCython
except:
    relabelCython = None
from scipy.optimize.optimize import fmin, fminbound
import scipy.ndimage.filters
logger = logging.getLogger("pyFAI.utils")
import time
timelog = logging.getLogger("pyFAI.timeit")

cu_fft = None  # No cuda here !
if sys.platform != "win32":
    WindowsError = RuntimeError
has_fftw3 = None
try:
    import fftw3
    has_fftw3 = True
except (ImportError, WindowsError) as err:
    logging.warn("Exception %s: FFTw3 not available. Falling back on Scipy", err)
    has_fftw3 = False

import traceback

def deprecated(func):
    def wrapper(*arg, **kw):
        """
        decorator that deprecates the use of a function
        """
        logger.warning("%s is Deprecated !!! %s" % (func.func_name, os.linesep.join([""] + traceback.format_stack()[:-1])))
        return func(*arg, **kw)
    return wrapper

def gaussian(M, std):
    """
    Return a Gaussian window of length M with standard-deviation std.

    This differs from the scipy.signal.gaussian implementation as:
    - The default for sym=False (needed for gaussian filtering without shift)
    - This implementation is normalized

    @param M: length of the windows (int)
    @param std: standatd deviation sigma

    The FWHM is 2*numpy.sqrt(2 * numpy.pi)*std

    """
    x = numpy.arange(M) - M / 2.0
    return numpy.exp(-(x / std) ** 2 / 2.0) / std / numpy.sqrt(2 * numpy.pi)


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
    if val != None:
        s = str(val)
    return s


def timeit(func):
    def wrapper(*arg, **kw):
        '''This is the docstring of timeit:
        a decorator that logs the execution time'''
        t1 = time.time()
        res = func(*arg, **kw)
        timelog.warning("%s took %.3fs" % (func.func_name, time.time() - t1))
        return res
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def gaussian_filter(input_img, sigma, mode="reflect", cval=0.0):
    """
    2-dimensional Gaussian filter implemented with FFTw

    @param input_img:    input array to filter
    @type input_img: array-like
    @param sigma: standard deviation for Gaussian kernel.
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
    @type sigma: scalar or sequence of scalars
    @param mode: {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    @param cval: scalar, optional
        Value to fill past edges of input if ``mode`` is 'constant'. Default is 0.0
    """
    res = None
    # TODO: understand why this is needed !
    if "has_fftw3" not in dir():
        has_fftw3 = ("fftw3" in sys.modules)
    if has_fftw3:
        try:
            if isinstance(sigma, (list, tuple)):
                sigma = (float(sigma[0]), float(sigma[1]))
            else:
                sigma = (float(sigma), float(sigma))
            k0 = int(ceil(4.0 * float(sigma[0])))
            k1 = int(ceil(4.0 * float(sigma[1])))

            if mode != "wrap":
                input_img = expand(input_img, (k0, k1), mode, cval)
            s0, s1 = input_img.shape
            sum_init = input_img.astype(numpy.float32).sum()
            fftOut = numpy.zeros((s0, s1), dtype=complex)
            fftIn = numpy.zeros((s0, s1), dtype=complex)
            with sem:
                fft = fftw3.Plan(fftIn, fftOut, direction='forward')
                ifft = fftw3.Plan(fftOut, fftIn, direction='backward')

            g0 = gaussian(s0, sigma[0])
            g1 = gaussian(s1, sigma[1])
            g0 = numpy.concatenate((g0[s0 // 2:], g0[:s0 // 2]))  # faster than fftshift
            g1 = numpy.concatenate((g1[s1 // 2:], g1[:s1 // 2]))  # faster than fftshift
            g2 = numpy.outer(g0, g1)
            g2fft = numpy.zeros((s0, s1), dtype=complex)
            fftIn[:, :] = g2.astype(complex)
            fft()
            g2fft[:, :] = fftOut.conjugate()

            fftIn[:, :] = input_img.astype(complex)
            fft()

            fftOut *= g2fft
            ifft()
            out = fftIn.real.astype(numpy.float32)
            sum_out = out.sum()
            res = out * sum_init / sum_out
            if mode != "wrap":
                res = res[k0:-k0, k1:-k1]
        except MemoryError:
            logging.error("MemoryError in FFTw3 part. Falling back on Scipy")
    if res is None:
        has_fftw3 = False
        res = scipy.ndimage.filters.gaussian_filter(input_img, sigma, mode=(mode or "reflect"))
    return res



def shift(input_img, shift_val):
    """
    Shift an array like  scipy.ndimage.interpolation.shift(input_img, shift_val, mode="wrap", order=0) but faster
    @param input_img: 2d numpy array
    @param shift_val: 2-tuple of integers
    @return: shifted image
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
    2-dimensional Difference of Gaussian filter implemented with FFTw

    @param input_img:    input_img array to filter
    @type input_img: array-like
    @param sigma: standard deviation for Gaussian kernel.
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.
    @type sigma: scalar or sequence of scalars
    @param mode: {'reflect','constant','nearest','mirror', 'wrap'}, optional
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'. Default is 'reflect'
    @param cval: scalar, optional
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

        if fftw3:
            sum_init = input_img.astype(numpy.float32).sum()
            fftOut = numpy.zeros((s0, s1), dtype=complex)
            fftIn = numpy.zeros((s0, s1), dtype=complex)

            with sem:
                fft = fftw3.Plan(fftIn, fftOut, direction='forward')
                ifft = fftw3.Plan(fftOut, fftIn, direction='backward')


            g2fft = numpy.zeros((s0, s1), dtype=complex)
            fftIn[:, :] = shift(dog(sigma1, sigma2, (s0, s1)), (s0 // 2, s1 // 2)).astype(complex)
            fft()
            g2fft[:, :] = fftOut.conjugate()

            fftIn[:, :] = input_img.astype(complex)
            fft()

            fftOut *= g2fft
            ifft()
            out = fftIn.real.astype(numpy.float32)
            sum_out = out.sum()
            res = out * sum_init / sum_out
        else:
            res = numpy.fft.ifft2(numpy.fft.fft2(input_img.astype(complex)) * \
                                  numpy.fft.fft2(shift(dog(sigma1, sigma2, (s0, s1)), (s0 // 2, s1 // 2)).astype(complex)).conjugate())
        if mode == "wrap":
            return res
        else:
            return res[k0:-k0, k1:-k1]

def expand(input_img, sigma, mode="constant", cval=0.0):

    """Expand array a with its reflection on boundaries

    @param a: 2D array
    @param sigma: float or 2-tuple of floats.
    @param mode:"constant", "nearest", "reflect" or mirror
    @param cval: filling value used for constant, 0.0 by default

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
        output[k0:k0 + s0, :k1] = numpy.outer(input_img[:, 0], numpy.ones(k1))
        output[:k0, k1:k1 + s1] = numpy.outer(numpy.ones(k0), input_img[0, :])
        output[-k0:, k1:s1 + k1] = numpy.outer(numpy.ones(k0), input_img[-1, :])
        output[k0:s0 + k0, -k1:] = numpy.outer(input_img[:, -1], numpy.ones(k1))
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

    @param label: a label array coming out of scipy.ndimage.measurement.label
    @param data: an array containing the raw data
    @param blured: an array containing the blured data
    @param max_size: the max number of label wanted
    @return array like label
    """
    if relabelCython:
        max_label = label.max()
        a, b, c, d = relabelCython.countThem(label, data, blured)
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

def averageDark(lstimg, center_method="mean", cutoff=None):
    """
    Averages a serie of dark (or flat) images.
    Centers the result on the mean or the median ...
    but averages all frames within  cutoff*std

    @param lstimg: list of 2D images or a 3D stack
    @param center_method: is the center calculated by a "mean" or a "median"
    @param cutoff: keep all data where (I-center)/std < cutoff
    @return: 2D image averaged
    """
    if "ndim" in dir(lstimg) and lstimg.ndim == 3:
        stack = lstimg.astype(numpy.float32)
        shape = stack.shape[1:]
        length = stack.shape[0]
    else:
        shape = lstimg[0].shape
        length = len(lstimg)
        if length == 1:
            return lstimg[0].astype(numpy.float32)
        stack = numpy.zeros((length, shape[0], shape[1]), dtype=numpy.float32)
        for i, img in enumerate(lstimg):
           stack[i] = img
    if center_method in dir(stack):
        center = stack.__getattribute__(center_method)(axis=0)
    elif center_method == "median":
        center = numpy.median(stack, axis=0)
    else:
        raise RuntimeError("Cannot understand method: %s in averageDark" % center_method)
    if cutoff is None or cutoff <= 0:
        output = center
    else:
        std = stack.std(axis=0)
        strides = 0, std.strides[0], std.strides[1]
        std.shape = 1, shape[0], shape[1]
        std.strides = strides
        center.shape = 1, shape[0], shape[1]
        center.strides = strides
        mask = ((abs(stack - center) / std) > cutoff)
        stack[numpy.where(mask)] = 0.0
        summed = stack.sum(axis=0)
        output = summed / numpy.maximum(1, (length - mask.sum(axis=0)))
    return output

def averageImages(listImages, output=None, threshold=0.1, minimum=None, maximum=None,
                   darks=None, flats=None, filter_="mean", correct_flat_from_dark=False,
                   cutoff=None, format="edf"):
    """
    Takes a list of filenames and create an average frame discarding all saturated pixels.

    @param listImages: list of string representing the filenames
    @param output: name of the optional output file
    @param threshold: what is the upper limit? all pixel > max*(1-threshold) are discareded.
    @param minimum: minimum valid value or True
    @param maximum: maximum valid value
    @param darks: list of dark current images for subtraction
    @param flats: list of flat field images for division
    @param filter_: can be maximum, mean or median (default=mean)
    @param correct_flat_from_dark: shall the flat be re-corrected ?
    @param cutoff: keep all data where (I-center)/std < cutoff
    @return: filename with the data or the data ndarray in case format=None
    """
    if filter_ not in ["min", "max", "median", "mean"]:
        logger.warning("Filter %s not understood. switch to mean filter")
        filter_ = "mean"
    ld = len(listImages)
    sumImg = None
    do_dark = (darks is not None)
    do_flat = (flats is not None)
    dark = None
    flat = None
    big_img = None
    for idx, fn in enumerate(listImages[:]):
        if type(fn) in types.StringTypes:
            logger.info("Reading %s" % fn)
            ds = fabio.open(fn).data
        else:
            ds = fn
            fn = "numpy_array"
            listImages[idx] = fn
        logger.debug("Intensity range for %s is %s --> %s", fn, ds.min(), ds.max())
        shape = ds.shape
        if sumImg is None:
            sumImg = numpy.zeros((shape[0], shape[1]), dtype=numpy.float32)
        if do_dark and (dark is None):
            if "ndim" in dir(darks) and darks.ndim == 3:
                dark = averageDark(darks, center_method="mean", cutoff=4)
            elif ("__len__" in dir(darks)) and (type(darks[0]) in types.StringTypes):
                dark = averageDark([fabio.open(f).data for f in darks if os.path.exists(f)], center_method="mean", cutoff=4)
            elif ("__len__" in dir(darks)) and ("ndim" in dir(darks[0])) and (darks[0].ndim == 2):
                dark = averageDark(darks, center_method="mean", cutoff=4)
        if do_flat and (flat is  None):
            if "ndim" in dir(flats) and flats.ndim == 3:
                flat = averageDark(flats, center_method="mean", cutoff=4)
            elif ("__len__" in dir(flats)) and (type(flats[0]) in types.StringTypes):
                flat = averageDark([fabio.open(f).data for f in flats if os.path.exists(f)], center_method="mean", cutoff=4)
            elif ("__len__" in dir(flats)) and ("ndim" in dir(flats[0])) and (flats[0].ndim == 2):
                flat = averageDark(flats, center_method="mean", cutoff=4)
            else:
                logger.warning("there is some wrong with flats=%s" % (flats))
            if correct_flat_from_dark:
                flat -= dark
            flat[numpy.where(flat <= 0) ] = 1.0
        correctedImg = numpy.ascontiguousarray(ds, numpy.float32)
        if threshold or minimum or maximum:
            correctedImg = removeSaturatedPixel(correctedImg, threshold, minimum, maximum)
        if do_dark:
            correctedImg -= dark
        if do_flat:
             correctedImg /= flat
        if (cutoff or (filter_ == "median")):
            if big_img is None:
                logger.info("Big array allocation for median filter or cut-off")
                big_img = numpy.zeros((ld, shape[0], shape[1]), dtype=numpy.float32)
            big_img[idx, :, :] = correctedImg
        elif filter_ == "max":
            sumImg = numpy.maximum(correctedImg, sumImg)
        elif filter_ == "min":
            sumImg = numpy.minimum(correctedImg, sumImg)
        elif filter_ == "mean":
            sumImg += correctedImg
    if cutoff or (filter_ == "median"):
        datared = averageDark(big_img, filter_, cutoff)
    else:
        if filter_ in ["max", "min"]:
            datared = numpy.ascontiguousarray(sumImg, dtype=numpy.float32)
        elif filter_ == "mean":
            datared = sumImg / numpy.float32(ld)
    logger.debug("Intensity range in merged dataset : %s --> %s", datared.min(), datared.max())
    if format is not None:
        if format.startswith("."):
            format = format.lstrip(".")
        if (output is None):
            prefix = ""
            for ch in zip(*listImages):
                c = ch[0]
                good = True
                if c in ["*", "?", "[", "{", "("]:
                    good = False
                    break
                for i in ch:
                    if i != c:
                        good = False
                        break
                if good:
                    prefix += c
                else:
                    break
            if filter_ == "max":
                output = "maxfilt%02i-%s.%s" % (ld,prefix,format)
            elif filter_ == "median":
                output = "medfilt%02i-%s.%s" % (ld,prefix,format)
            elif filter_ == "median":
                output = "meanfilt%02i-%s.%s" % (ld, prefix, format)
            else:
                output = "merged%02i-%s.%s" % (ld, prefix, format)
        if format and output:
            if "." in format:  # in case "edf.gz"
                format = format.split(".")[0]
            fabiomod = fabio.__getattribute__(format + "image")
            fabioclass = fabiomod.__getattribute__(format + "image")
            header = {"method":filter_,
                      "nframes":ld,
                      "cutoff":str(cutoff)}
            form = "merged_file_%%0%ii" % len(str(len(listImages)))
            header_list = ["method", "nframes", "cutoff"]
            for i, f in enumerate(listImages):
                name = form % i
                header[name] = f
                header_list.append(name)
            fimg = fabioclass(data=datared,
                              header=header)
#            if "header_keys" in dir(fimg):
            fimg.header_keys = header_list
                                      
            fimg.write(output)
            logger.info("Wrote %s" % output)
        return output
    else:
        return datared

def boundingBox(img):
    """
    Tries to guess the bounding box around a valid massif

    @param img: 2D array like
    @return: 4-typle (d0_min, d1_min, d0_max, d1_max)
    """
    img = img.astype(numpy.int)
    img0 = (img.sum(axis=1) > 0).astype(numpy.int)
    img1 = (img.sum(axis=0) > 0).astype(numpy.int)
    dimg0 = img0[1:] - img0[:-1]
    min0 = dimg0.argmax()
    max0 = dimg0.argmin() + 1
    dimg1 = img1[1:] - img1[:-1]
    min1 = dimg1.argmax()
    max1 = dimg1.argmin() + 1
    if max0 == 1:
        max0 = img0.size
    if max1 == 1:
        max1 = img1.size
    return (min0, min1, max0, max1)


def removeSaturatedPixel(ds, threshold=0.1, minimum=None, maximum=None):
    """
    @param ds: a dataset as  ndarray

    @param threshold: what is the upper limit? all pixel > max*(1-threshold) are discareded.
    @param minimum: minumum valid value (or True for auto-guess)
    @param maximum: maximum valid value
    @return: another dataset
    """
    shape = ds.shape
    if ds.dtype == numpy.uint16:
        maxt = (1.0 - threshold) * 65535.0
    elif ds.dtype == numpy.int16:
        maxt = (1.0 - threshold) * 32767.0
    elif ds.dtype == numpy.uint8:
        maxt = (1.0 - threshold) * 255.0
    elif ds.dtype == numpy.int8:
        maxt = (1.0 - threshold) * 127.0
    else:
        if maximum is  None:
            maxt = (1.0 - threshold) * ds.max()
        else:
            maxt = maximum
    if maximum is not None:
        maxt = min(maxt, maximum)
    invalid = (ds > maxt)
    if minimum:
        if  minimum is True:  # automatic guess of the best minimum TODO: use the HWHM to guess the minumum...
            data_min = ds.min()
            x, y = numpy.histogram(numpy.log(ds - data_min + 1.0), bins=100)
            f = interp1d((y[1:] + y[:-1]) / 2.0, -x, bounds_error=False, fill_value= -x.min())
            max_low = fmin(f, y[1], disp=0)
            max_hi = fmin(f, y[-1], disp=0)
            if max_hi > max_low:
                f = interp1d((y[1:] + y[:-1]) / 2.0, x, bounds_error=False)
                min_center = fminbound(f, max_low, max_hi)
            else:
                min_center = max_hi
            minimum = float(numpy.exp(y[((min_center / y) > 1).sum() - 1])) - 1.0 + data_min
            logger.debug("removeSaturatedPixel: best minimum guessed is %s", minimum)
        ds[ds < minimum] = minimum
        ds -= minimum  # - 1.0

    if invalid.sum(dtype=int) == 0:
        logger.debug("No saturated area where found")
        return ds
    gi = ndimage.morphology.binary_dilation(invalid)
    lgi, nc = ndimage.label(gi)
    if nc > 100:
        logger.warning("More than 100 saturated zones were found on this image !!!!")
    for zone in range(nc + 1):
        dzone = (lgi == zone)
        if dzone.sum(dtype=int) > ds.size // 2:
            continue
        min0, min1, max0, max1 = boundingBox(dzone)
        ksize = min(max0 - min0, max1 - min1)
        subset = ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)]
        while subset.max() > maxt:
            subset = ndimage.median_filter(subset, ksize)
        ds[max(0, min0 - 4 * ksize):min(shape[0], max0 + 4 * ksize), max(0, min1 - 4 * ksize):min(shape[1], max1 + 4 * ksize)] = subset
    fabio.edfimage.edfimage(data=ds).write("removeSaturatedPixel.edf")
    return ds


def binning(input_img, binsize):
    """
    @param input_img: input ndarray
    @param binsize: int or 2-tuple representing the size of the binning
    @return: binned input ndarray
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
        for i in xrange(binsize[0]):
            for j in xrange(binsize[1]):
                out += input_img[i::binsize[0], j::binsize[1]]
    else:
        temp = input_img.copy()
        temp.shape = (outputSize[0], binsize[0], outputSize[1], binsize[1])
        out = temp.sum(axis=3).sum(axis=1)
    return out


def unBinning(binnedArray, binsize, norm=True):
    """
    @param binnedArray: input ndarray
    @param binsize: 2-tuple representing the size of the binning
    @return: unBinned input ndarray
    """
    if isinstance(binsize, int):
        binsize = (binsize, binsize)
    outputShape = []
    for i, j in zip(binnedArray.shape, binsize):
        outputShape.append(i * j)
    out = numpy.zeros(tuple(outputShape), dtype=binnedArray.dtype)
    for i in xrange(binsize[0]):
        for j in xrange(binsize[1]):
            out[i::binsize[0], j::binsize[1]] += binnedArray
    if norm:
        out /= binsize[0] * binsize[1]
    return out



def shiftFFT(input_img, shift_val, method="fftw"):
    """
    Do shift using FFTs
    Shift an array like  scipy.ndimage.interpolation.shift(input, shift, mode="wrap", order="infinity") but faster
    @param input_img: 2d numpy array
    @param shift_val: 2-tuple of float
    @return: shifted image

    """
    # TODO: understand why this is needed !
    if "has_fftw3" not in dir():
        has_fftw3 = ("fftw3" in sys.modules)
    if "has_fftw3" and ("fftw3" not in dir()):
        fftw3 = sys.modules.get("fftw3")
    else:
        fftw3 = None
#    print fftw3
    d0, d1 = input_img.shape
    v0, v1 = shift_val
    f0 = numpy.fft.ifftshift(numpy.arange(-d0 // 2, d0 // 2))
    f1 = numpy.fft.ifftshift(numpy.arange(-d1 // 2, d1 // 2))
    m1, m0 = numpy.meshgrid(f1, f0)
    e0 = numpy.exp(-2j * numpy.pi * v0 * m0 / float(d0))
    e1 = numpy.exp(-2j * numpy.pi * v1 * m1 / float(d1))
    e = e0 * e1
    if method.startswith("fftw") and (fftw3 is not None):
        input_ = numpy.zeros((d0, d1), dtype=complex)
        output = numpy.zeros((d0, d1), dtype=complex)
        with sem:
            fft = fftw3.Plan(input_, output, direction='forward', flags=['estimate'])
            ifft = fftw3.Plan(output, input_, direction='backward', flags=['estimate'])
        input_[:, :] = input_img.astype(complex)
        fft()
        output *= e
        ifft()
        out = input_ / input_.size
    else:
        out = numpy.fft.ifft2(numpy.fft.fft2(input_img) * e)
    return abs(out)

def maximum_position(img):
    """
    Same as scipy.ndimage.measurements.maximum_position:
    Find the position of the maximum of the values of the array.

    @param img: 2-D image
    @return: 2-tuple of int with the position of the maximum
    """
    maxarg = numpy.argmax(img)
    _, s1 = img.shape
    return (maxarg // s1, maxarg % s1)

def center_of_mass(img):
    """
    Calculate the center of mass of of the array.
    Like scipy.ndimage.measurements.center_of_mass
    @param img: 2-D array
    @return: 2-tuple of float with the center of mass
    """
    d0, d1 = img.shape
    a0, a1 = numpy.ogrid[:d0, :d1]
    img = img.astype("float64")
    img /= img.sum()
    return ((a0 * img).sum(), (a1 * img).sum())

def measure_offset(img1, img2, method="numpy", withLog=False, withCorr=False):
    """
    Measure the actual offset between 2 images
    @param img1: ndarray, first image
    @param img2: ndarray, second image, same shape as img1
    @param withLog: shall we return logs as well ? boolean
    @return: tuple of floats with the offsets
    """
    method = str(method)
    ################################################################################
    # Start convolutions
    ################################################################################
    shape = img1.shape
    logs = []
    assert img2.shape == shape
    t0 = time.time()
    if method[:4] == "fftw" and (fftw3 is not None):
        input_ = numpy.zeros(shape, dtype=complex)
        output = numpy.zeros(shape, dtype=complex)
        with sem:
            fft = fftw3.Plan(input_, output, direction='forward', flags=['measure'])
            ifft = fftw3.Plan(output, input_, direction='backward', flags=['measure'])
        input_[:, :] = img2.astype(complex)
        fft()
        temp = output.conjugate()
        input_[:, :] = img1.astype(complex)
        fft()
        output *= temp
        ifft()
        res = input_.real / input_.size
#    elif method[:4] == "cuda" and (cu_fft is not None):
#        with sem:
#            cuda_correlate = CudaCorrelate(shape)
#            res = cuda_correlate.correlate(img1, img2)
    else:  # use numpy fftpack
        i1f = numpy.fft.fft2(img1)
        i2f = numpy.fft.fft2(img2)
        res = numpy.fft.ifft2(i1f * i2f.conjugate()).real
    t1 = time.time()

    ################################################################################
    # END of convolutions
    ################################################################################
    offset1 = maximum_position(res)
    res = shift(res, (shape[0] // 2 , shape[1] // 2))
    mean = res.mean(dtype="float64")
    maxi = res.max()
    std = res.std(dtype="float64")
    SN = (maxi - mean) / std
    new = numpy.maximum(numpy.zeros(shape), res - numpy.ones(shape) * (mean + std * SN * 0.9))
    com2 = center_of_mass(new)
    logs.append("MeasureOffset: fine result of the centered image: %s %s " % com2)
    offset2 = ((com2[0] - shape[0] // 2) % shape[0] , (com2[1] - shape[1] // 2) % shape[1])
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
    Takes an argv and expand it (under Windows, cmd does not convert *.tif into a list of files.
    Keeps only valid files (thanks to glob)

    @param args: list of files or wilcards
    @return: list of actual args
    """
    new = []
    for afile in  args:
        if os.path.exists(afile):
            new.append(afile)
        else:
            new += glob.glob(afile)
    return new


def _get_data_path(filename):
    """
    @param filename: the name of the requested data file.
    @type filename: str

    In the future ....
    This method try to find the requested ui-name following the
    xfreedesktop recommendations. First the source directory then
    the system locations

    For now, just perform a recursive search
    """
    # when using bootstrap the file is located under the build directory
#    real_filename = os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                                 os.path.pardir,
#                                                 os.path.pardir,
#                                                 os.path.pardir,
#                                                 'data',
#                                                 filename))
#    if not os.path.exists(real_filename):
    resources = [os.path.dirname(__file__)]
    try:
        import xdg.BaseDirectory
        resources += xdg.BaseDirectory.load_data_paths("pyFAI")
    except ImportError:
        pass

    for resource in resources:
        real_filename = os.path.join(resource, filename)
        if os.path.exists(real_filename):
            return real_filename
    else:
        raise Exception("Can not find the [%s] resource, "
                        " something went wrong !!!" % (real_filename,))
#    else:
#        return real_filename


def get_ui_file(filename):
    return _get_data_path(os.path.join("gui", filename))
#    return _get_data_path(filename)


def get_cl_file(filename):
#    return _get_data_path(os.path.join("openCL", filename))
    return _get_data_path(filename)

def deg2rad(dd):
    """
    Convert degrees to radian in the range -pi->pi

    @param dd: angle in degrees

    Nota: depending on the platform it could be 0<2pi
    A branch is cheaper than a trigo operation
    """
    while dd > 180.0:
        dd -= 360.0
    while dd <= -180.0:
        dd += 360.0
    return dd * pi / 180.

class lazy_property(object):
    '''
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    '''

    def __init__(self,fget):
        self.fget = fget
        self.func_name = fget.__name__


    def __get__(self,obj,cls):
        if obj is None:
            return None
        value = self.fget(obj)
        setattr(obj,self.func_name,value)
        return value

