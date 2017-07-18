#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
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

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/07/2017"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import os
import logging
logger = logging.getLogger(__name__)
import warnings
import tempfile
import threading
import gc
import numpy
from math import pi, log, ceil
from numpy import rad2deg
from .geometry import Geometry
from . import units
from .utils import EPS32, deg2rad, crc32
from .decorators import deprecated
from .containers import Integrate1dResult, Integrate2dResult
from .io import DefaultAiWriter
error = None


try:
    from .ext import splitBBoxLUT
except ImportError as error:
    logger.warning("Unable to import pyFAI.ext.splitBBoxLUT for"
                   " Look-up table based azimuthal integration")
    logger.debug("Backtrace", exc_info=True)
    splitBBoxLUT = None

try:
    # Used for 1D integration
    from .ext import splitPixel
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitPixel"
                 " full pixel splitting: %s" % error)
    logger.debug("Backtrace", exc_info=True)
    splitPixel = None

# try:
#    # Used fro 2D integration
#    from .ext import splitPixelFull  # IGNORE:F0401
# except ImportError as error:
#    logger.error("Unable to import pyFAI.splitPixelFull"
#                  " full pixel splitting: %s" % error)
#    splitPixelFull = None

try:
    from .ext import splitBBox  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitBBox"
                 " Bounding Box pixel splitting: %s" % error)
    splitBBox = None

try:
    from .ext import histogram
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.histogram"
                 " Cython histogram implementation: %s" % error)
    histogram = None

try:
    from .ext import splitBBoxCSR  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitBBoxCSR"
                 " CSR based azimuthal integration: %s" % error)
    splitBBoxCSR = None

try:
    from .ext import splitPixelFullCSR  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.ext.splitPixelFullCSR"
                 " CSR based azimuthal integration: %s" % error)
    splitPixelFullCSR = None


from .opencl import ocl
if ocl:
    try:
        from .opencl import azim_hist as ocl_azim  # IGNORE:F0401
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.ocl_azim: %s",
                     error)
        ocl_azim = None
    try:
        from .opencl import azim_csr as ocl_azim_csr  # IGNORE:F0401
    except ImportError as error:
        logger.error("Unable to import pyFAI.ocl_azim_csr: %s",
                     error)
        ocl_azim_csr = None
    try:
        from .opencl import azim_lut as ocl_azim_lut  # IGNORE:F0401
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.ocl_azim_lut for: %s",
                     error)
        ocl_azim_lut = None
    try:
        from .opencl import sort as ocl_sort
    except ImportError as error:  # IGNORE:W0703
        logger.error("Unable to import pyFAI.ocl_sort for: %s",
                     error)
        ocl_sort = None
else:
    ocl_azim = ocl_azim_csr = ocl_azim_lut = None


class AzimuthalIntegrator(Geometry):
    """
    This class is an azimuthal integrator based on P. Boesecke's
    geometry and histogram algorithm by Manolo S. del Rio and V.A Sole

    All geometry calculation are done in the Geometry class

    main methods are:

        >>> tth, I = ai.integrate1d(data, npt, unit="2th_deg")
        >>> q, I, sigma = ai.integrate1d(data, npt, unit="q_nm^-1", error_model="poisson")
        >>> regrouped = ai.integrate2d(data, npt_rad, npt_azim, unit="q_nm^-1")[0]
    """
    DEFAULT_METHOD = "splitbbox"

    def __init__(self, dist=1, poni1=0, poni2=0,
                 rot1=0, rot2=0, rot3=0,
                 pixel1=None, pixel2=None,
                 splineFile=None, detector=None, wavelength=None):
        """
        :param dist: distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        :type dist: float
        :param poni1: coordinate of the point of normal incidence along the detector's first dimension, in meter
        :type poni1: float
        :param poni2: coordinate of the point of normal incidence along the detector's second dimension, in meter
        :type poni2: float
        :param rot1: first rotation from sample ref to detector's ref, in radians
        :type rot1: float
        :param rot2: second rotation from sample ref to detector's ref, in radians
        :type rot2: float
        :param rot3: third rotation from sample ref to detector's ref, in radians
        :type rot3: float
        :param pixel1: Deprecated. Pixel size of the fist dimension of the detector,  in meter.
            If both pixel1 and pixel2 are not None, detector pixel size is overwritten.
            Prefer defining the detector pixel size on the provided detector object.
            Prefer defining the detector pixel size on the provided detector
            object (``detector.pixel1 = 5e-6``).
        :type pixel1: float
        :param pixel2: Deprecated. Pixel size of the second dimension of the detector,  in meter.
            If both pixel1 and pixel2 are not None, detector pixel size is overwritten.
            Prefer defining the detector pixel size on the provided detector
            object (``detector.pixel2 = 5e-6``).
        :type pixel2: float
        :param splineFile: Deprecated. File containing the geometric distortion of the detector.
            If not None, pixel1 and pixel2 are ignored and detector spline is overwritten.
            Prefer defining the detector spline manually
            (``detector.splineFile = "file.spline"``).
        :type splineFile: str
        :param detector: name of the detector or Detector instance. String
            description is deprecated. Prefer using the result of the detector
            factory: ``pyFAI.detector_factory("eiger4m")``
        :type detector: str or pyFAI.Detector
        :param wavelength: Wave length used in meter
        :type wavelength: float
        """
        Geometry.__init__(self, dist, poni1, poni2,
                          rot1, rot2, rot3,
                          pixel1, pixel2, splineFile, detector, wavelength)
        self._nbPixCache = {}  # key=shape, value: array

        # mask, maskfile, darkcurrent and flatfield are properties pointing to
        # self.detector now (16/06/2017

        self._ocl_integrator = None
        self._ocl_lut_integr = None
        self._ocl_csr_integr = None
        self._lut_integrator = None
        self._csr_integrator = None
        self._ocl_sorter = None
        self._ocl_sem = threading.Semaphore()
        self._lut_sem = threading.Semaphore()
        self._csr_sem = threading.Semaphore()
        self._ocl_csr_sem = threading.Semaphore()
        self._ocl_lut_sem = threading.Semaphore()
        self._empty = 0.0

    def reset(self):
        """
        Reset azimuthal integrator in addition to other arrays.
        """
        Geometry.reset(self)
        with self._ocl_sem:
            self._ocl_integrator = None
            self._ocl_csr_integr = None
        with self._lut_sem:
            self._lut_integrator = None
            self._csr_integrator = None

    def create_mask(self, data, mask=None,
                    dummy=None, delta_dummy=None, mode="normal"):
        """
        Combines various masks into another one.

        :param data: input array of data
        :type data: ndarray
        :param mask: input mask (if none, self.mask is used)
        :type mask: ndarray
        :param dummy: value of dead pixels
        :type dummy: float
        :param delta_dumy: precision of dummy pixels
        :type delta_dummy: float
        :param mode: can be "normal" or "numpy" (inverted) or "where" applied to the mask
        :type mode: str

        :return: the new mask
        :rtype: ndarray of bool

        This method combine two masks (dynamic mask from *data &
        dummy* and *mask*) to generate a new one with the 'or' binary
        operation.  One can adjust the level, with the *dummy* and
        the *delta_dummy* parameter, when you consider the *data*
        values needs to be masked out.

        This method can work in two different *mode*:

            * "normal": False for valid pixels, True for bad pixels
            * "numpy": True for valid pixels, false for others

        This method tries to accomodate various types of masks (like
        valid=0 & masked=-1, ...) and guesses if an input mask needs
        to be inverted.
        """
        shape = data.shape
        #       ^^^^   this is why data is mandatory !
        if mask is None:
            mask = self.mask
        if mask is None:
            mask = numpy.zeros(shape, dtype=bool)
        elif mask.min() < 0 and mask.max() == 0:  # 0 is valid, <0 is invalid
            mask = (mask < 0)
        else:
            mask = mask.astype(bool)
        if mask.sum(dtype=int) > mask.size // 2:
            logger.warning("Mask likely to be inverted as more"
                           " than half pixel are masked !!!")
            numpy.logical_not(mask, mask)
        if (mask.shape != shape):
            try:
                mask = mask[:shape[0], :shape[1]]
            except Exception as error:  # IGNORE:W0703
                logger.error("Mask provided has wrong shape:"
                             " expected: %s, got %s, error: %s" %
                             (shape, mask.shape, error))
                mask = numpy.zeros(shape, dtype=bool)
        if dummy is not None:
            if delta_dummy is None:
                numpy.logical_or(mask, (data == dummy), mask)
            else:
                numpy.logical_or(mask,
                                 abs(data - dummy) <= delta_dummy,
                                 mask)
        if mode == "numpy":
            numpy.logical_not(mask, mask)
        elif mode == "where":
            mask = numpy.where(numpy.logical_not(mask))
        return mask

    def dark_correction(self, data, dark=None):
        """
        Correct for Dark-current effects.
        If dark is not defined, correct for a dark set by "set_darkfiles"

        :param data: input ndarray with the image
        :param dark: ndarray with dark noise or None
        :return: 2tuple: corrected_data, dark_actually used (or None)
        """
        dark = dark if dark is not None else self.detector.darkcurrent
        if dark is not None:
            return data - dark, dark
        else:
            return data, None

    def flat_correction(self, data, flat=None):
        """
        Correct for flat field.
        If flat is not defined, correct for a flat set by "set_flatfiles"

        :param data: input ndarray with the image
        :param flat: ndarray with flatfield or None for no correction
        :return: 2tuple: corrected_data, flat_actually used (or None)
        """
        flat = flat if flat is not None else self.detector.flatfield
        if flat is not None:
            return data / flat, flat
        else:
            return data, None

    @deprecated
    def xrpd_numpy(self, data, npt, filename=None, correctSolidAngle=True,
                   tthRange=None, mask=None, dummy=None, delta_dummy=None,
                   polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Numpy implementation: slow and without pixels splitting.
        This method should not be used in production, it remains
        to explain how other more sophisticated algorithms works.
        Use xrpd_splitBBox instead

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of the 2theta
        :type tthRange: (float, float), optional
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray

        :return: (2theta, I) in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Bad pixels can be masked out by setting them to an impossible
        value (-1) and calling this value the "dummy value".  Some
        Pilatus detectors are setting non existing pixel to -1 and
        dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so that
        any value between -3.5 and -0.5 are considered as bad.

        The polarisation correction can be taken into account with the
        *polarization_factor* parameter. Set it between [-1, 1], to
        correct your data. If set to 0 there is correction for circular 
        polarization, When set to None, there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        mask = self.create_mask(data, mask, dummy, delta_dummy, mode="where")
        tth = self.twoThetaArray(data.shape)[mask]
        data = numpy.ascontiguousarray(data, dtype=numpy.float32)

        data, dark = self.dark_correction(data, dark)
        data, flat = self.flat_correction(data, flat)

        if correctSolidAngle:
            data /= self.solidAngleArray(data.shape)

        if polarization_factor is not None:
            data /= self.polarization(data.shape, factor=polarization_factor)

        data = data[mask]

        if tthRange is not None:
            tthRange = (deg2rad(tthRange[0]),
                        deg2rad(tthRange[-1]) * EPS32)
        else:
            tthRange = (tth.min(), tth.max() * EPS32)
        if npt not in self._nbPixCache:
            ref, _ = numpy.histogram(tth, npt, range=tthRange)
            self._nbPixCache[npt] = numpy.maximum(1, ref)

        val, b = numpy.histogram(tth,
                                 bins=npt,
                                 weights=data,
                                 range=tthRange)
        tthAxis = 90.0 * (b[1:] + b[:-1]) / pi
        I = val / self._nbPixCache[npt]
        self.save1D(filename, tthAxis, I, None, "2th_deg",
                    dark is not None, flat is not None, polarization_factor)
        return tthAxis, I

    @deprecated
    def xrpd_cython(self, data, npt, filename=None, correctSolidAngle=True,
                    tthRange=None, mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    pixelSize=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Cython multithreaded implementation: fast but still lacks
        pixels splitting as numpy implementation. This method should
        not be used in production, it remains to explain why
        histograms are hard to implement in parallel. Use
        xrpd_splitBBox instead
        """
        if histogram is None:
            logger.warning("pyFAI.histogram is not available,"
                           " falling back on old numpy method !")
            return self.xrpd_numpy(data=data,
                                   npt=npt,
                                   filename=filename,
                                   correctSolidAngle=correctSolidAngle,
                                   tthRange=tthRange,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   polarization_factor=polarization_factor)

        mask = self.create_mask(data, mask, dummy, delta_dummy, mode="where")
        tth = self.twoThetaArray(data.shape)[mask]
        data = numpy.ascontiguousarray(data, dtype=numpy.float32)

        data, dark = self.dark_correction(data, dark)
        data, flat = self.flat_correction(data, flat)

        if correctSolidAngle:
            data /= self.solidAngleArray(data.shape)

        if polarization_factor is not None:
            data /= self.polarization(data.shape, factor=polarization_factor)

        data = data[mask]

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, -1))
        tthAxis, I, _, _ = histogram.histogram(pos=tth,
                                               weights=data,
                                               bins=npt,
                                               bin_range=tthRange,
                                               pixelSize_in_Pos=pixelSize,
                                               empty=dummy if dummy is not None else self._empty)
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg",
                    dark is not None, flat is not None, polarization_factor)
        return tthAxis, I

    @deprecated
    def xrpd_splitBBox(self, data, npt, filename=None, correctSolidAngle=True,
                       tthRange=None, chiRange=None, mask=None,
                       dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Cython implementation

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of the 2theta
        :type tthRange: (float, float), optional
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), optional, disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor correction
        :type polarization_factor: float or None
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray

        :return: (2theta, I) in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image as also a chi coordinate. So it
        is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern. you just need to
        set the range with the *chiRange* parameter. like the
        *tthRange* parameter, value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same). Pixels can also be maseked by seting them to an

        Bad pixels can be masked out by setting them to an impossible
        value (-1) and calling this value the "dummy value".  Some
        Pilatus detectors are setting non existing pixel to -1 and
        dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so that
        any value between -3.5 and -0.5 are considered as bad.

        Some Pilatus detectors are setting non existing pixel to -1
        and dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so
        that any value between -3.5 and -0.5 are considered as bad.

        The polarisation correction can be taken into account with the
        *polarization_factor* parameter. Set it between [-1, 1], to
        correct your data. If set to 0, the circular polarization is used. 
        When None, there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitBBox is None:
            logger.warning("Unable to use splitBBox,"
                           " falling back on numpy histogram !")
            return self.xrpd_numpy(data=data,
                                   npt=npt,
                                   filename=filename,
                                   correctSolidAngle=correctSolidAngle,
                                   tthRange=tthRange,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   polarization_factor=polarization_factor,
                                   dark=dark,
                                   flat=flat)
        shape = data.shape
        if chiRange is not None:
            chi = self.chiArray(shape)
            dchi = self.deltaChi(shape)
        else:
            chi = None
            dchi = None

        tth = self.twoThetaArray(data.shape)
        dtth = self.delta2Theta(data.shape)

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, -1))

        if chiRange is not None:
            chiRange = tuple(deg2rad(chiRange[i]) for i in (0, -1))

        if flat is None:
            flat = self.flatfield

        if dark is None:
            dark = self.darkcurrent

        if correctSolidAngle:
            solidangle = self.solidAngleArray(data.shape)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(data.shape, polarization_factor)

        # ??? what about create_mask like with other methods
        if mask is None:
            mask = self.mask

        # outPos, outMerge, outData, outCount
        tthAxis, I, _, _ = splitBBox.histoBBox1d(weights=data,
                                                 pos0=tth,
                                                 delta_pos0=dtth,
                                                 pos1=chi,
                                                 delta_pos1=dchi,
                                                 bins=npt,
                                                 pos0Range=tthRange,
                                                 pos1Range=chiRange,
                                                 dummy=dummy,
                                                 delta_dummy=delta_dummy,
                                                 mask=mask,
                                                 dark=dark,
                                                 flat=flat,
                                                 solidangle=solidangle,
                                                 polarization=polarization,
                                                 )
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg", dark is not None, flat is not None, polarization_factor)
        return tthAxis, I

    @deprecated
    def xrpd_splitPixel(self, data, npt,
                        filename=None, correctSolidAngle=True,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Cython implementation (single threaded)

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of the 2theta
        :type tthRange: (float, float), optional
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), optional, disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray

        :return: (2theta, I) in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image as also a chi coordinate. So it
        is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern. you just need to
        set the range with the *chiRange* parameter. like the
        *tthRange* parameter, value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Bad pixels can be masked out by setting them to an impossible
        value (-1) and calling this value the "dummy value".  Some
        Pilatus detectors are setting non existing pixel to -1 and
        dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so that
        any value between -3.5 and -0.5 are considered as bad.

        Some Pilatus detectors are setting non existing pixel to -1
        and dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so
        that any value between -3.5 and -0.5 are considered as bad.

        The polarisation correction can be taken into account with the
        *polarization_factor* parameter. Set it between [-1, 1], to
        correct your data. If set to 0: circular polarization. 
        None for no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitPixel is None:
            logger.warning("splitPixel is not available,"
                           " falling back on numpy histogram !")
            return self.xrpd_numpy(data=data,
                                   npt=npt,
                                   filename=filename,
                                   correctSolidAngle=correctSolidAngle,
                                   tthRange=tthRange,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   polarization_factor=polarization_factor,
                                   dark=dark,
                                   flat=flat)

        pos = self.cornerArray(data.shape)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(data.shape)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(data.shape, polarization_factor)

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, -1))

        if chiRange is not None:
            chiRange = tuple(deg2rad(chiRange[i]) for i in (0, -1))

        # ??? what about dark and flat computation like with other methods ?

        tthAxis, I, _, _ = splitPixel.fullSplit1D(pos=pos,
                                                  weights=data,
                                                  bins=npt,
                                                  pos0Range=tthRange,
                                                  pos1Range=chiRange,
                                                  dummy=dummy,
                                                  delta_dummy=delta_dummy,
                                                  mask=mask,
                                                  dark=dark,
                                                  flat=flat,
                                                  solidangle=solidangle,
                                                  polarization=polarization,
                                                  )
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg",
                    dark is not None, flat is not None, polarization_factor)
        return tthAxis, I

    # Default implementation:
    xrpd = xrpd_splitBBox

    @deprecated
    def xrpd_OpenCL(self, data, npt, filename=None, correctSolidAngle=True,
                    dark=None, flat=None,
                    tthRange=None, mask=None, dummy=None, delta_dummy=None,
                    devicetype="gpu", useFp64=True,
                    platformid=None, deviceid=None, safe=True):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        This is (now) a pure pyopencl implementation so it just needs
        pyopencl which requires a clean OpenCL installation. This
        implementation is not slower than the previous Cython and is
        less problematic for compilation/installation.

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of the 2theta
        :type tthRange: (float, float), optional
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        OpenCL specific parameters:

        :param devicetype: possible values "cpu", "gpu", "all" or "def"
        :type devicetype: str
        :param useFp64: shall histogram be done in double precision (strongly adviced)
        :type useFp64: bool
        :param platformid: platform number
        :type platformid: int
        :param deviceid: device number
        :type deviceid: int
        :param safe: set to False if your GPU is already set-up correctly
        :type safe: bool

        :return: (2theta, I) angle being in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.
        The powder diffraction is computed internally using an
        histogram which by default use should be done in 64bits. One
        can switch to 32 bits with the *useFp64* parameter set to
        False. In 32bit mode; do not expect better than 1% error and
        one can even observe overflows ! 32 bits is only left for
        testing hardware capabilities and should NEVER be used in any
        real experiment analysis.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image has also a chi coordinate. So
        it is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern. You just need to
        set the range with the *chiRange* parameter; like the
        *tthRange* parameter, value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Bad pixels can also be masked by setting them to an impossible
        value (-1) and calling this value the "dummy value".  Some
        Pilatus detectors are setting non existing pixel to -1 and
        dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so that
        any value between -3.5 and -0.5 are considered as bad.

        *devicetype*, *platformid* and *deviceid*, parameters are
        specific to the OpenCL implementation. If you set *devicetype*
        to 'all', 'cpu', 'gpu', 'def' you can force the device used to
        perform the computation; the program will select the device
        accordinly. By setting *platformid* and *deviceid*, you can
        directly address a specific device (which is computer
        specific).

        The *safe* parameter is specific to the integrator object,
        located on the OpenCL device. You can set it to False if you
        think the integrator is already setup correcty (device,
        geometric arrays, mask, 2theta/chi range). Unless many tests
        will be done at each integration.
        """
        if not ocl_azim:
            logger.warning("OpenCL implementation not available"
                           " falling back on old method !")
            return self.xrpd_splitBBox(data=data,
                                       npt=npt,
                                       filename=filename,
                                       correctSolidAngle=correctSolidAngle,
                                       tthRange=tthRange,
                                       mask=mask,
                                       dummy=dummy,
                                       delta_dummy=delta_dummy,
                                       dark=dark,
                                       flat=flat
                                       )
        shape = data.shape
        if flat is None:
            flat = self.flatfield
        if flat is None:
            flat = 1

        if dark is None:
            dark = self.darkcurrent
        if dark is not None:
            data = data.astype(numpy.float32) - dark

        if self._ocl_integrator is None:
            with self._ocl_sem:
                if self._ocl_integrator is None:
                    size = data.size
                    fd, tmpfile = tempfile.mkstemp(".log", "pyfai-opencl-")
                    os.close(fd)
                    integr = ocl_azim.Integrator1d(tmpfile)
                    if (platformid is not None) and (deviceid is not None):
                        rc = integr.init(devicetype=devicetype,
                                         platformid=platformid,
                                         deviceid=deviceid,
                                         useFp64=useFp64)
                    else:
                        rc = integr.init(devicetype=devicetype,
                                         useFp64=useFp64)
                    if rc:
                        raise RuntimeError("Failed to initialize OpenCL"
                                           " deviceType %s (%s,%s) 64bits: %s"
                                           % (devicetype, platformid,
                                              deviceid, useFp64))

                    if integr.getConfiguration(size, npt):
                        raise RuntimeError("Failed to configure 1D integrator"
                                           " with Ndata=%s and Nbins=%s"
                                           % (size, npt))

                    if integr.configure():
                        raise RuntimeError('Failed to compile kernel')
                    pos0 = self.twoThetaArray(shape)
                    delta_pos0 = self.delta2Theta(shape)
                    if tthRange is not None and len(tthRange) > 1:
                        pos0_min = deg2rad(min(tthRange))
                        pos0_maxin = deg2rad(max(tthRange))
                    else:
                        pos0_min = pos0.min()
                        pos0_maxin = pos0.max()
                    if pos0_min < 0.0:
                        pos0_min = 0.0
                    pos0_max = pos0_maxin * EPS32
                    if integr.loadTth(pos0, delta_pos0, pos0_min, pos0_max):
                        raise RuntimeError("Failed to upload 2th arrays")
                    self._ocl_integrator = integr
        with self._ocl_sem:
            if safe:
                param = self._ocl_integrator.get_status()
                if (dummy is None) and param["dummy"]:
                    self._ocl_integrator.unsetDummyValue()
                elif (dummy is not None) and not param["dummy"]:
                    if delta_dummy is None:
                        delta_dummy = 1e-6
                    self._ocl_integrator.setDummyValue(dummy, delta_dummy)
                if (correctSolidAngle and not param["solid_angle"]):
                    self._ocl_integrator.setSolidAngle(flat * self.solidAngleArray(shape, correctSolidAngle))
                elif (not correctSolidAngle) and param["solid_angle"] and (flat is 1):
                    self._ocl_integrator.unsetSolidAngle()
                elif not correctSolidAngle and not param["solid_angle"] and (flat is not 1):
                    self._ocl_integrator.setSolidAngle(flat)
                if (mask is not None) and not param["mask"]:
                    self._ocl_integrator.setMask(mask)
                elif (mask is None) and param["mask"]:
                    self._ocl_integrator.unsetMask()
            tthAxis, I, _, = self._ocl_integrator.execute(data)
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg")
        return tthAxis, I

    def setup_LUT(self, shape, npt, mask=None,
                  pos0_range=None, pos1_range=None, mask_checksum=None,
                  unit=units.TTH):
        """
        Prepare a look-up-table

        :param shape: shape of the dataset
        :type shape: (int, int)
        :param npt: number of points in the the output pattern
        :type npt: int or (int, int)
        :param mask: array with masked pixel (1=masked)
        :type mask: ndarray
        :param pos0_range: range in radial dimension
        :type pos0_range: (float, float)
        :param pos1_range: range in azimuthal dimension
        :type pos1_range: (float, float)
        :param mask_checksum: checksum of the mask buffer
        :type mask_checksum: int (or anything else ...)
        :param unit: use to propagate the LUT object for further checkings
        :type unit: pyFAI.units.Unit

        This method is called when a look-up table needs to be set-up.
        The *shape* parameter, correspond to the shape of the original
        datatset. It is possible to customize the number of point of
        the output histogram with the *npt* parameter which can be
        either an integer for an 1D integration or a 2-tuple of
        integer in case of a 2D integration. The LUT will have a
        different shape: (npt, lut_max_size), the later parameter
        being calculated during the instanciation of the splitBBoxLUT
        class.

        It is possible to prepare the LUT with a predefine
        *mask*. This operation can speedup the computation of the
        later integrations. Instead of applying the patch on the
        dataset, it is taken into account during the histogram
        computation. If provided the *mask_checksum* prevent the
        re-calculation of the mask. When the mask changes, its
        checksum is used to reset (or not) the LUT (which is a very
        time consuming operation !)

        It is also possible to restrain the range of the 1D or 2D
        pattern with the *pos1_range* and *pos2_range*.

        The *unit* parameter is just propagated to the LUT integrator
        for further checkings: The aim is to prevent an integration to
        be performed in 2th-space when the LUT was setup in q space.
        """

        if "__len__" in dir(npt) and len(npt) == 2:
            int2d = True
        else:
            int2d = False
        pos0 = self.array_from_unit(shape, "center", unit, scale=False)
        dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
        if (pos1_range is None) and (not int2d):
            pos1 = None
            dpos1 = None
        else:
            pos1 = self.chiArray(shape)
            dpos1 = self.deltaChi(shape)
        if ("__len__" in dir(pos0_range)) and (len(pos0_range) > 1):
            pos0_min = min(pos0_range)
            pos0_maxin = max(pos0_range)
            pos0Range = (pos0_min, pos0_maxin * EPS32)
        else:
            pos0Range = None
        if ("__len__" in dir(pos1_range)) and (len(pos1_range) > 1):
            pos1_min = min(pos1_range)
            pos1_maxin = max(pos1_range)
            pos1Range = (pos1_min, pos1_maxin * EPS32)
        else:
            pos1Range = None
        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape

        if int2d:
            return splitBBoxLUT.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                            bins=npt,
                                            pos0Range=pos0Range,
                                            pos1Range=pos1Range,
                                            mask=mask,
                                            mask_checksum=mask_checksum,
                                            allow_pos0_neg=False,
                                            unit=unit)
        else:
            return splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                            bins=npt,
                                            pos0Range=pos0Range,
                                            pos1Range=pos1Range,
                                            mask=mask,
                                            mask_checksum=mask_checksum,
                                            allow_pos0_neg=False,
                                            unit=unit)

    def setup_CSR(self, shape, npt, mask=None, pos0_range=None, pos1_range=None, mask_checksum=None, unit=units.TTH, split="bbox"):
        """
        Prepare a look-up-table

        :param shape: shape of the dataset
        :type shape: (int, int)
        :param npt: number of points in the the output pattern
        :type npt: int or (int, int)
        :param mask: array with masked pixel (1=masked)
        :type mask: ndarray
        :param pos0_range: range in radial dimension
        :type pos0_range: (float, float)
        :param pos1_range: range in azimuthal dimension
        :type pos1_range: (float, float)
        :param mask_checksum: checksum of the mask buffer
        :type mask_checksum: int (or anything else ...)
        :param unit: use to propagate the LUT object for further checkings
        :type unit: pyFAI.units.Unit
        :param split: Splitting scheme: valid options are "no", "bbox", "full"

        This method is called when a look-up table needs to be set-up.
        The *shape* parameter, correspond to the shape of the original
        datatset. It is possible to customize the number of point of
        the output histogram with the *npt* parameter which can be
        either an integer for an 1D integration or a 2-tuple of
        integer in case of a 2D integration. The LUT will have a
        different shape: (npt, lut_max_size), the later parameter
        being calculated during the instanciation of the splitBBoxLUT
        class.

        It is possible to prepare the LUT with a predefine
        *mask*. This operation can speedup the computation of the
        later integrations. Instead of applying the patch on the
        dataset, it is taken into account during the histogram
        computation. If provided the *mask_checksum* prevent the
        re-calculation of the mask. When the mask changes, its
        checksum is used to reset (or not) the LUT (which is a very
        time consuming operation !)

        It is also possible to restrain the range of the 1D or 2D
        pattern with the *pos1_range* and *pos2_range*.

        The *unit* parameter is just propagated to the LUT integrator
        for further checkings: The aim is to prevent an integration to
        be performed in 2th-space when the LUT was setup in q space.
        """

        if "__len__" in dir(npt) and len(npt) == 2:
            int2d = True
        else:
            int2d = False
        if split == "full":
            pos = self.array_from_unit(shape, "corner", unit, scale=False)
        else:
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            if split == "no":
                dpos0 = None
            else:
                dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
            if (pos1_range is None) and (not int2d):
                pos1 = None
                dpos1 = None
            else:
                pos1 = self.chiArray(shape)
                if split == "no":
                    dpos1 = None
                else:
                    dpos1 = self.deltaChi(shape)
        if ("__len__" in dir(pos0_range)) and (len(pos0_range) > 1):
            pos0_min = min(pos0_range)
            pos0_maxin = max(pos0_range)
            pos0Range = (pos0_min, pos0_maxin * EPS32)
        else:
            pos0Range = None
        if ("__len__" in dir(pos1_range)) and (len(pos1_range) > 1):
            pos1_min = min(pos1_range)
            pos1_maxin = max(pos1_range)
            pos1Range = (pos1_min, pos1_maxin * EPS32)
        else:
            pos1Range = None
        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape
        if split == "full":

            if int2d:
                raise NotImplementedError("Full pixel splitting using CSR is not yet available in 2D")
#                return splitBBoxCSR.HistoBBox2d(pos0, dpos0, pos1, dpos1,
#                                                bins=npt,
#                                                pos0Range=pos0Range,
#                                                pos1Range=pos1Range,
#                                                mask=mask,
#                                                mask_checksum=mask_checksum,
#                                                allow_pos0_neg=False,
#                                                unit=unit)
            else:
                return splitPixelFullCSR.FullSplitCSR_1d(pos,
                                                         bins=npt,
                                                         pos0Range=pos0Range,
                                                         pos1Range=pos1Range,
                                                         mask=mask,
                                                         mask_checksum=mask_checksum,
                                                         allow_pos0_neg=False,
                                                         unit=unit)
        else:
            if int2d:
                return splitBBoxCSR.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0Range=pos0Range,
                                                pos1Range=pos1Range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit)
            else:
                return splitBBoxCSR.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0Range=pos0Range,
                                                pos1Range=pos1Range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                )

    @deprecated
    def xrpd_LUT(self, data, npt, filename=None, correctSolidAngle=True,
                 tthRange=None, chiRange=None, mask=None,
                 dummy=None, delta_dummy=None,
                 safe=True, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from an image.

        Parallel Cython implementation using a Look-Up Table (OpenMP).

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of the 2theta angle
        :type tthRange: (float, float), optional
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), optional
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        LUT specific parameters:

        :param safe: set to False if your LUT is already set-up correctly (mask, ranges, ...).
        :type safe: bool

        :return: (2theta, I) with 2theta angle in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image as also a chi coordinate. So it
        is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern by setting the
        range with the *chiRange* parameter. Like the *tthRange*
        parameter, value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Dynamic masking (i.e recalculated for each image) can be
        achieved by setting masked pixels to an impossible value (-1)
        and calling this value the "dummy value". Dynamic masking is
        computed at integration whereas static masking is done at
        LUT-generation, hence faster.

        Some Pilatus detectors are setting non existing pixel to -1
        and dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so
        that any value between -3.5 and -0.5 are considered as bad.

        The *safe* parameter is specific to the LUT implementation,
        you can set it to false if you think the LUT calculated is
        already the correct one (setup, mask, 2theta/chi range).

        TODO: replace with integrate1D

        """

        if not splitBBoxLUT:
            logger.warning("Look-up table implementation not available:"
                           " falling back on old method !")
            return self.xrpd_splitBBox(data=data,
                                       npt=npt,
                                       filename=filename,
                                       correctSolidAngle=correctSolidAngle,
                                       tthRange=tthRange,
                                       mask=mask,
                                       dummy=dummy,
                                       delta_dummy=delta_dummy,
                                       flat=flat,
                                       dark=dark)
        return self.integrate1d(data,
                                npt,
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=None,
                                error_model=None,
                                radial_range=tthRange,
                                azimuth_range=chiRange,
                                mask=mask,
                                dummy=dummy,
                                delta_dummy=delta_dummy,
                                polarization_factor=None,
                                dark=dark,
                                flat=flat,
                                method="lut",
                                unit="2th_deg",
                                safe=safe)

    @deprecated
    def xrpd_LUT_OCL(self, data, npt, filename=None, correctSolidAngle=True,
                     tthRange=None, chiRange=None, mask=None,
                     dummy=None, delta_dummy=None,
                     safe=True, devicetype="all",
                     platformid=None, deviceid=None, dark=None, flat=None):

        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        PyOpenCL implementation using a Look-Up Table (OpenCL). The
        look-up table is a Cython module.

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle in degrees.
        :type chiRange: (float, float)
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        LUT specific parameters:

        :param safe: set to False if your LUT & GPU is already set-up correctly
        :type safe: bool

        OpenCL specific parameters:

        :param devicetype: can be "all", "cpu", "gpu", "acc" or "def"
        :type devicetype: str
        :param platformid: platform number
        :type platformid: int
        :param deviceid: device number
        :type deviceid: int

        :return: (2theta, I) in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image has also a chi coordinate. So
        it is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern by setting the
        *chiRange* parameter. Like the *tthRange* parameter, value
        outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Dynamic masking (i.e recalculated for each image) can be
        achieved by setting masked pixels to an impossible value (-1)
        and calling this value the "dummy value". Dynamic masking is
        computed at integration whereas static masking is done at
        LUT-generation, hence faster.

        Some Pilatus detectors are setting non existing pixel to -1
        and dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so
        that any value between -3.5 and -0.5 are considered as bad.

        The *safe* parameter is specific to the OpenCL/LUT
        implementation, you can set it to false if you think the LUT
        calculated is already the correct one (setup, mask, 2theta/chi
        range) and the device set-up is the expected one.

        *devicetype*, *platformid* and *deviceid*, parameters are
        specific to the OpenCL implementation. If you set *devicetype*
        to 'all', 'cpu', or 'gpu' you can force the device used to
        perform the computation. By providing the *platformid* and
        *deviceid* you can chose a specific device (computer
        specific).
        """
        if not (splitBBoxLUT and ocl_azim_lut):
            logger.warning("Look-up table implementation not available:"
                           " falling back on old method !")
            return self.xrpd_splitBBox(data=data,
                                       npt=npt,
                                       filename=filename,
                                       correctSolidAngle=correctSolidAngle,
                                       tthRange=tthRange,
                                       mask=mask,
                                       dummy=dummy,
                                       delta_dummy=delta_dummy)
        meth = "lut_ocl"
        if platformid and deviceid:
            meth += "_%i,%i" % (platformid, deviceid)
        elif devicetype != "all":
            meth += "_" + devicetype

        return self.integrate1d(data,
                                npt,
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=None,
                                error_model=None,
                                radial_range=tthRange,
                                azimuth_range=chiRange,
                                mask=mask,
                                dummy=dummy,
                                delta_dummy=delta_dummy,
                                polarization_factor=None,
                                dark=dark,
                                flat=flat,
                                method=meth,
                                unit="2th_deg",
                                safe=safe)

    @deprecated
    def xrpd_CSR_OCL(self, data, npt, filename=None, correctSolidAngle=True,
                     tthRange=None, mask=None, dummy=None, delta_dummy=None,
                     dark=None, flat=None, chiRange=None, safe=True,
                     devicetype="all", platformid=None, deviceid=None, block_size=32):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        PyOpenCL implementation using a CSR version of the Look-Up Table (OpenCL). The
        look-up table is a Cython module.

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: integer
        :param filename: file to save data in ascii format 2 column
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle in degrees.
        :type chiRange: (float, float)
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        LUT specific parameters:

        :param safe: set to False if your LUT & GPU is already set-up correctly
        :type safe: bool

        OpenCL specific parameters:

        :param devicetype: can be "all", "cpu", "gpu", "acc" or "def"
        :type devicetype: str
        :param platformid: platform number
        :type platformid: int
        :param deviceid: device number
        :type deviceid: int
        :param block_size: OpenCL grid size
        :type block_size: int
        Unused/deprecated arguments:
        :param padded: deprecated

        :return: (2theta, I) in degrees
        :rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *npt* parameter. If you give a *filename*, the
        powder diffraction is also saved as a two column text file.

        It is possible to correct or not the powder diffraction
        pattern using the *correctSolidAngle* parameter. The weight of
        a pixel is ponderate by its solid angle.

        The 2theta range of the powder diffraction pattern can be set
        using the *tthRange* parameter. If not given the maximum
        available range is used. Indeed pixel outside this range are
        ignored.

        Each pixel of the *data* image has also a chi coordinate. So
        it is possible to restrain the chi range of the pixels to
        consider in the powder diffraction pattern by setting the
        *chiRange* parameter. Like the *tthRange* parameter, value
        outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Dynamic masking (i.e recalculated for each image) can be
        achieved by setting masked pixels to an impossible value (-1)
        and calling this value the "dummy value". Dynamic masking is
        computed at integration whereas static masking is done at
        LUT-generation, hence faster.

        Some Pilatus detectors are setting non existing pixel to -1
        and dead pixels to -2. Then use dummy=-2 & delta_dummy=1.5 so
        that any value between -3.5 and -0.5 are considered as bad.

        The *safe* parameter is specific to the OpenCL/LUT
        implementation, you can set it to false if you think the LUT
        calculated is already the correct one (setup, mask, 2theta/chi
        range) and the device set-up is the expected one.

        *devicetype*, *platformid* and *deviceid*, parameters are
        specific to the OpenCL implementation. If you set *devicetype*
        to 'all', 'cpu', or 'gpu' you can force the device used to
        perform the computation. By providing the *platformid* and
        *deviceid* you can chose a specific device (computer
        specific).
        """
        if not (splitBBoxCSR and ocl_azim_csr):
            logger.warning("CSR implementation not available:"
                           " falling back on look-up table implementation!")
            return self.xrpd_LUT_OCL(data=data,
                                     npt=npt,
                                     filename=filename,
                                     correctSolidAngle=correctSolidAngle,
                                     tthRange=tthRange,
                                     mask=mask,
                                     dummy=dummy,
                                     delta_dummy=delta_dummy,
                                     dark=dark,
                                     flat=flat,
                                     chiRange=chiRange,
                                     safe=safe,
                                     devicetype=devicetype,
                                     platformid=platformid,
                                     deviceid=deviceid)
        meth = "csr_ocl"
        if platformid and deviceid:
            meth += "_%i,%i" % (platformid, deviceid)
        elif devicetype != "all":
            meth += "_" + devicetype

        return self.integrate1d(data,
                                npt,
                                filename=filename,
                                correctSolidAngle=correctSolidAngle,
                                variance=None,
                                error_model=None,
                                radial_range=tthRange,
                                azimuth_range=chiRange,
                                mask=mask,
                                dummy=dummy,
                                delta_dummy=delta_dummy,
                                polarization_factor=None,
                                dark=dark,
                                flat=flat,
                                method=meth,
                                unit="2th_deg",
                                safe=safe,
                                block_size=block_size)

    @deprecated
    def xrpd2_numpy(self, data, npt_rad, npt_azim=360,
                    filename=None, correctSolidAngle=True,
                    dark=None, flat=None,
                    tthRange=None, chiRange=None,
                    mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta, Chi) from
        a set of data, an image

        Pure numpy implementation (VERY SLOW !!!)

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt_rad: number of bin of the Radial (horizontal) axis (2Theta)
        :type npt: int
        :param npt_azim: number of bin of the Azimuthal (vertical) axis (chi)
        :type npt_azim: int
        :param filename: file to save data in
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        :return: azimuthaly regrouped data, 2theta pos and chipos
        :rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is simular to
        a rectangular to polar conversion. The number of point of the
        new image is given by *npt_rad* and *npt_azim*. If you give a
        *filename*, the new image is also saved as an edf file.

        It is possible to correct the 2theta/chi pattern using the
        *correctSolidAngle* parameter. The weight of a pixel is
        ponderate by its solid angle.

        The 2theta and range of the new image can be set using the
        *tthRange* parameter. If not given the maximum available range
        is used. Indeed pixel outside this range are ignored.

        Each pixel of the *data* image has a 2theta and a chi
        coordinate. So it is possible to restrain on any of those
        ranges ; you just need to set the range with the *tthRange* or
        thee *chiRange* parameter. like the *tthRange* parameter,
        value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Masking can also be achieved by setting masked pixels to an
        impossible value (-1) and calling this value the "dummy
        value".  Some Pilatus detectors are setting non existing pixel
        to -1 and dead pixels to -2. Then use dummy=-2 &
        delta_dummy=1.5 so that any value between -3.5 and -0.5 are
        considered as bad.
        """
        mask = self.create_mask(data, mask, dummy, delta_dummy, mode="numpy")
        shape = data.shape
        tth = self.twoThetaArray(shape)[mask]
        chi = self.chiArray(shape)[mask]
        data, dark = self.dark_correction(data, dark)
        data, flat = self.flat_correction(data, flat)
        data = data.astype(numpy.float32)[mask]

        if correctSolidAngle is not None:
            data /= self.solidAngleArray(shape, correctSolidAngle)[mask]

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, -1))
        else:
            tthRange = [tth.min(), tth.max() * EPS32]

        if chiRange is not None:
            chiRange = tuple(deg2rad(chiRange[i]) for i in (0, -1))
        else:
            chiRange = [chi.min(), chi.max() * EPS32]

        bins = (npt_azim, npt_rad)
        if bins not in self._nbPixCache:
            ref, _, _ = numpy.histogram2d(chi, tth,
                                          bins=list(bins),
                                          range=[chiRange, tthRange])
            self._nbPixCache[bins] = numpy.maximum(1.0, ref)

        val, binsChi, bins2Th = numpy.histogram2d(chi, tth,
                                                  bins=list(bins),
                                                  weights=data,
                                                  range=[chiRange, tthRange])
        I = val / self._nbPixCache[bins]
        self.save2D(filename, I, bins2Th, binsChi)

        return I, bins2Th, binsChi

    @deprecated
    def xrpd2_histogram(self, data, npt_rad, npt_azim=360,
                        filename=None, correctSolidAngle=True,
                        dark=None, flat=None,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Cython implementation: fast but incaccurate

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt_rad: number of bin of the Radial (horizontal) axis (2Theta)
        :type npt: int
        :param npt_azim: number of bin of the Azimuthal (vertical) axis (chi)
        :type npt_azim: int
        :param filename: file to save data in
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float

        :return: azimuthaly regrouped data, 2theta pos and chipos
        :rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is simular to
        a rectangular to polar conversion. The number of point of the
        new image is given by *npt_rad* and *npt_azim*. If you give a
        *filename*, the new image is also saved as an edf file.

        It is possible to correct the 2theta/chi pattern using the
        *correctSolidAngle* parameter. The weight of a pixel is
        ponderate by its solid angle.

        The 2theta and range of the new image can be set using the
        *tthRange* parameter. If not given the maximum available range
        is used. Indeed pixel outside this range are ignored.

        Each pixel of the *data* image has a 2theta and a chi
        coordinate. So it is possible to restrain on any of those
        ranges ; you just need to set the range with the *tthRange* or
        thee *chiRange* parameter. like the *tthRange* parameter,
        value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Masking can also be achieved by setting masked pixels to an
        impossible value (-1) and calling this value the "dummy
        value". Some Pilatus detectors are setting non existing pixel
        to -1 and dead pixels to -2. Then use dummy=-2 &
        delta_dummy=1.5 so that any value between -3.5 and -0.5 are
        considered as bad.
        """

        if histogram is None:
            logger.warning("pyFAI.histogram is not available,"
                           " falling back on numpy")
            return self.xrpd2_numpy(data=data,
                                    npt_rad=npt_rad,
                                    npt_azim=npt_azim,
                                    filename=filename,
                                    correctSolidAngle=correctSolidAngle,
                                    tthRange=tthRange,
                                    chiRange=chiRange,
                                    mask=mask,
                                    dummy=dummy,
                                    delta_dummy=delta_dummy)
        shape = data.shape
        mask = self.create_mask(data, mask, dummy, delta_dummy, mode="numpy")
        tth = self.twoThetaArray(data.shape)[mask]
        chi = self.chiArray(data.shape)[mask]
        data = data.astype(numpy.float32)[mask]

        if dark is None:
            dark = self.darkcurrent
        if dark is not None:
            data -= dark[mask]

        if flat is None:
            flat = self.flatfield
        if flat is not None:
            data /= flat[mask]

        if correctSolidAngle:
            data /= self.solidAngleArray(shape)[mask]

        if dummy is None:
            I, binsChi, bins2Th, _, _ = histogram.histogram2d(pos0=chi, pos1=tth,
                                                              bins=(npt_azim, npt_rad),
                                                              weights=data,
                                                              split=1,
                                                              empty=dummy if dummy is not None else self._empty)
        bins2Th = rad2deg(bins2Th)
        binsChi = rad2deg(binsChi)
        self.save2D(filename, I, bins2Th, binsChi)
        return I, bins2Th, binsChi

    @deprecated
    def xrpd2_splitBBox(self, data, npt_rad, npt_azim=360,
                        filename=None, correctSolidAngle=True,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Split pixels according to their coordinate and a bounding box

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt_rad: number of bin of the Radial (horizontal) axis (2Theta)
        :type npt: int
        :param npt_azim: number of bin of the Azimuthal (vertical) axis (chi)
        :type npt_azim: int
        :param filename: file to save data in
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray

        :return: azimuthaly regrouped data, 2theta pos. and chi pos.
        :rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is similar to
        a rectangular to polar conversion. The number of point of the
        new image is given by *npt_rad* and *npt_azim*. If you give a
        *filename*, the new image is also saved as an edf file.

        It is possible to correct the 2theta/chi pattern using the
        *correctSolidAngle* parameter. The weight of a pixel is
        ponderate by its solid angle.

        The 2theta and range of the new image can be set using the
        *tthRange* parameter. If not given the maximum available range
        is used. Indeed pixel outside this range are ignored.

        Each pixel of the *data* image has a 2theta and a chi
        coordinate. So it is possible to restrain on any of those
        ranges ; you just need to set the range with the *tthRange* or
        thee *chiRange* parameter. like the *tthRange* parameter,
        value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Masking can also be achieved by setting masked pixels to an
        impossible value (-1) and calling this value the "dummy
        value". Some Pilatus detectors are setting non existing pixel
        to -1 and dead pixels to -2. Then use dummy=-2 &
        delta_dummy=1.5 so that any value between -3.5 and -0.5 are
        considered as bad.

        the polarisation correction can be taken into account with the
        *polarization_factor* parameter. Set it between [-1, 1], to
        correct your data. If set to 0: circular polarization. When None there 
        is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitBBox is None:
            logger.warning("Unable to use splitBBox,"
                           " falling back on numpy histogram !")
            return self.xrpd2_histogram(data=data,
                                        npt_rad=npt_rad,
                                        npt_azim=npt_azim,
                                        filename=filename,
                                        correctSolidAngle=correctSolidAngle,
                                        tthRange=tthRange,
                                        chiRange=chiRange,
                                        mask=mask,
                                        dummy=dummy,
                                        delta_dummy=delta_dummy)
        tth = self.twoThetaArray(data.shape)
        chi = self.chiArray(data.shape)
        dtth = self.delta2Theta(data.shape)
        dchi = self.deltaChi(data.shape)

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, 1))

        if chiRange is not None:
            chiRange = tuple(deg2rad(chiRange[i]) for i in (0, -1))

        if dark is None:
            dark = self.darkcurrent

        if flat is None:
            flat = self.flatfield

        if correctSolidAngle:
            solidangle = self.solidAngleArray(data.shape)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(data.shape, polarization_factor)
        I, bins2Th, binsChi, _, _ = splitBBox.histoBBox2d(weights=data,
                                                          pos0=tth,
                                                          delta_pos0=dtth,
                                                          pos1=chi,
                                                          delta_pos1=dchi,
                                                          bins=(npt_rad, npt_azim),
                                                          pos0Range=tthRange,
                                                          pos1Range=chiRange,
                                                          dummy=dummy,
                                                          delta_dummy=delta_dummy,
                                                          mask=mask,
                                                          dark=dark,
                                                          flat=flat,
                                                          solidangle=solidangle,
                                                          polarization=polarization)
        bins2Th = rad2deg(bins2Th)
        binsChi = rad2deg(binsChi)
        self.save2D(filename, I, bins2Th, binsChi, has_dark=dark is not None, has_flat=flat is not None,
                    polarization_factor=polarization_factor)
        return I, bins2Th, binsChi

    @deprecated
    def xrpd2_splitPixel(self, data, npt_rad, npt_azim=360,
                         filename=None, correctSolidAngle=True,
                         tthRange=None, chiRange=None, mask=None,
                         dummy=None, delta_dummy=None,
                         polarization_factor=None, dark=None, flat=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Split pixels according to their corner positions

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt_rad: number of bin of the Radial (horizontal) axis (2Theta)
        :type npt: int
        :param npt_azim: number of bin of the Azimuthal (vertical) axis (chi)
        :type npt_azim: int
        :param filename: file to save data in
        :type filename: str
        :param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        :type correctSolidAngle: bool or int
        :param tthRange: The lower and upper range of 2theta
        :type tthRange: (float, float)
        :param chiRange: The lower and upper range of the chi angle.
        :type chiRange: (float, float), disabled for now
        :param mask: array with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels (dynamic mask)
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray

        :return: azimuthaly regrouped data, 2theta pos. and chi pos.
        :rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is similar to
        a rectangular to polar conversion. The number of point of the
        new image is given by *npt_rad* and *npt_azim*. If you give a
        *filename*, the new image is also saved as an edf file.

        It is possible to correct the 2theta/chi pattern using the
        *correctSolidAngle* parameter. The weight of a pixel is
        ponderate by its solid angle.

        The 2theta and range of the new image can be set using the
        *tthRange* parameter. If not given the maximum available range
        is used. Indeed pixel outside this range are ignored.

        Each pixel of the *data* image has a 2theta and a chi
        coordinate. So it is possible to restrain on any of those
        ranges ; you just need to set the range with the *tthRange* or
        thee *chiRange* parameter. like the *tthRange* parameter,
        value outside this range are ignored.

        Sometimes one needs to mask a few pixels (beamstop, hot
        pixels, ...), to ignore a few of them you just need to provide
        a *mask* array with a value of 1 for those pixels. To take a
        pixel into account you just need to set a value of 0 in the
        mask array. Indeed the shape of the mask array should be
        idential to the data shape (size of the array _must_ be the
        same).

        Masking can also be achieved by setting masked pixels to an
        impossible value (-1) and calling this value the "dummy
        value". Some Pilatus detectors are setting non existing pixel
        to -1 and dead pixels to -2. Then use dummy=-2 &
        delta_dummy=1.5 so that any value between -3.5 and -0.5 are
        considered as bad.

        the polarisation correction can be taken into account with the
        *polarization_factor* parameter. Set it between [-1, 1], to
        correct your data. If set to 0: circular polarization. 
        When None, there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitPixel is None:
            logger.warning("splitPixel is not available,"
                           " falling back on SplitBBox !")
            return self.xrpd2_splitBBox(
                                    data=data,
                                    npt_rad=npt_rad,
                                    npt_azim=npt_azim,
                                    filename=filename,
                                    correctSolidAngle=correctSolidAngle,
                                    tthRange=tthRange,
                                    chiRange=chiRange,
                                    mask=mask,
                                    dummy=dummy,
                                    delta_dummy=delta_dummy,
                                    polarization_factor=polarization_factor,
                                    dark=dark,
                                    flat=flat)

        pos = self.cornerArray(data.shape)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(data.shape)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(data.shape, polarization_factor)

        if dark is None:
            dark = self.darkcurrent

        if flat is None:
            flat = self.flatfield

        if tthRange is not None:
            tthRange = tuple(deg2rad(tthRange[i]) for i in (0, -1))

        if chiRange is not None:
            chiRange = tuple(deg2rad(chiRange[i]) for i in (0, -1))

        I, bins2Th, binsChi, _, _ = splitPixel.fullSplit2D(pos=pos,
                                                           weights=data,
                                                           bins=(npt_rad, npt_azim),
                                                           pos0Range=tthRange,
                                                           pos1Range=chiRange,
                                                           dummy=dummy,
                                                           delta_dummy=delta_dummy,
                                                           mask=mask,
                                                           dark=dark,
                                                           flat=flat,
                                                           solidangle=solidangle,
                                                           polarization=polarization)
        bins2Th = rad2deg(bins2Th)
        binsChi = rad2deg(binsChi)
        self.save2D(filename, I, bins2Th, binsChi, has_dark=dark is not None,
                    has_flat=flat is not None,
                    polarization_factor=polarization_factor)
        return I, bins2Th, binsChi

    xrpd2 = xrpd2_splitBBox

    def integrate1d(self, data, npt, filename=None,
                    correctSolidAngle=True,
                    variance=None, error_model=None,
                    radial_range=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    method="csr", unit=units.Q, safe=True,
                    normalization_factor=1.0,
                    block_size=32, profile=False, all=False, metadata=None):
        """Calculate the azimuthal integrated Saxs curve in q(nm^-1) by default

        Multi algorithm implementation (tries to be bullet proof), suitable for SAXS, WAXS, ... and much more



        :param data: 2D array from the Detector/CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: int
        :param filename: output filename in 2/3 column ascii format
        :type filename: str
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool
        :param variance: array containing the variance of the data. If not available, no error propagation is done
        :type variance: ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
               0 for circular polarization or random,
               None for no correction,
               True for using the former correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "csr", "nosplit_csr", "full_csr", "lut_ocl" and "csr_ocl" if you want to go on GPU. To Specify the device: "csr_ocl_1,2"
        :type method: str
        :param unit: Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
        :type unit: pyFAI.units.Unit
        :param safe: Do some extra checks to ensure LUT/CSR is still valid. False is faster.
        :type safe: bool
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param block_size: size of the block for OpenCL integration (unused?)
        :param profile: set to True to enable profiling in OpenCL
        :param all: if true return a dictionary with many more parameters (deprecated, please refer to the documentation of Integrate1dResult).
        :type all: bool
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: q/2th/r bins center positions and regrouped intensity (and error array if variance or variance model provided), uneless all==True.
        :rtype: Integrate1dResult, dict
        """
        if all:
            logger.warning("Deprecation: please use the object returned by ai.integrate1d, not the option `all`")

        method = method.lower()
        unit = units.to_unit(unit)

        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "provided"
            mask_crc = crc32(mask)

        shape = data.shape
        pos0_scale = unit.scale

        if radial_range:
            radial_range = tuple([i / pos0_scale for i in radial_range])

        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if azimuth_range is not None:
            azimuth_range = tuple(deg2rad(azimuth_range[i]) for i in (0, -1))
            if azimuth_range[1] <= azimuth_range[0]:
                azimuth_range = (azimuth_range[0], azimuth_range[1] + 2 * pi)
            chi = self.chiArray(shape)
        else:
            chi = None

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_checksum = None
        else:
            polarization, polarization_checksum = self.polarization(shape, polarization_factor, with_checksum=True)

        if dark is None:
            dark = self.detector.darkcurrent
            if dark is None:
                has_dark = False
            else:
                has_dark = "from detector"
        else:
            has_dark = "provided"

        if flat is None:
            flat = self.detector.flatfield
            if dark is None:
                has_flat = False
            else:
                has_flat = "from detector"
        else:
            has_flat = "provided"

        I = None
        sigma = None
        count = None
        sum_ = None

        if (I is None) and ("lut" in method):
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._lut_integrator is None:
                    reset = "init"
                if (not reset) and safe:
                    if self._lut_integrator.unit != unit:
                        reset = "unit changed"
                    if self._lut_integrator.bins != npt:
                        reset = "number of points changed"
                    if self._lut_integrator.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and\
                            (not self._lut_integrator.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and (self._lut_integrator.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and\
                            (self._lut_integrator.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and\
                            (self._lut_integrator.pos0Range is not None):
                        reset = "radial_range was defined in LUT"
                    elif (radial_range is not None) and\
                            (self._lut_integrator.pos0Range !=
                             (min(radial_range), max(radial_range) * EPS32)):
                        reset = ("radial_range is defined"
                                 " but not the same as in LUT")
                    if (azimuth_range is None) and\
                            (self._lut_integrator.pos1Range is not None):
                        reset = ("azimuth_range not defined and"
                                 " LUT had azimuth_range defined")
                    elif (azimuth_range is not None) and\
                            (self._lut_integrator.pos1Range !=
                             (min(azimuth_range), max(azimuth_range) * EPS32)):
                        reset = ("azimuth_range requested and"
                                 " LUT's azimuth_range don't match")
                if reset:
                    logger.info("AI.integrate1d: Resetting integrator because %s", reset)
                    try:
                        self._lut_integrator = self.setup_LUT(shape, npt, mask,
                                                              radial_range, azimuth_range,
                                                              mask_checksum=mask_crc, unit=unit)
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on default forward implementation")
                        self._lut_integrator = None
                        self._ocl_lut_integr = None
                        gc.collect()
                        method = self.DEFAULT_METHOD

                if self._lut_integrator:
                    if ("ocl" in method) and ocl_azim_lut:
                        with self._ocl_lut_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_lut_integr is None) or\
                                    (self._ocl_lut_integr.on_device["lut"] != self._lut_integrator.lut_checksum):
                                self._ocl_lut_integr = ocl_azim_lut.OCL_LUT_Integrator(self._lut_integrator.lut,
                                                                                       self._lut_integrator.size,
                                                                                       devicetype=devicetype,
                                                                                       platformid=platformid,
                                                                                       deviceid=deviceid,
                                                                                       checksum=self._lut_integrator.lut_checksum)
                            if self._ocl_lut_integr is not None:
                                I, sum_, count = self._ocl_lut_integr.integrate(data, dark=dark, flat=flat,
                                                                                solidangle=solidangle,
                                                                                solidangle_checksum=self._dssa_crc,
                                                                                dummy=dummy,
                                                                                delta_dummy=delta_dummy,
                                                                                polarization=polarization,
                                                                                polarization_checksum=polarization_checksum,
                                                                                normalization_factor=normalization_factor)
                                qAxis = self._lut_integrator.outPos  # this will be copied later
                                if error_model == "azimuthal":
                                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                                if variance is not None:
                                    var1d, a, b = self._ocl_lut_integr.integrate(variance,
                                                                                 solidangle=None,
                                                                                 dummy=dummy,
                                                                                 delta_dummy=delta_dummy,
                                                                                 normalization_factor=1.0)
                                    with numpy.errstate(divide='ignore'):
                                        sigma = numpy.sqrt(a) / (b * normalization_factor)
                                    sigma[b == 0] = dummy if dummy is not None else self._empty
                    else:
                        qAxis, I, sum_, count = self._lut_integrator.integrate(data, dark=dark, flat=flat,
                                                                               solidAngle=solidangle,
                                                                               dummy=dummy,
                                                                               delta_dummy=delta_dummy,
                                                                               polarization=polarization,
                                                                               normalization_factor=normalization_factor)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                        if variance is not None:
                            _, var1d, a, b = self._lut_integrator.integrate(variance,
                                                                            solidAngle=None,
                                                                            dummy=dummy,
                                                                            delta_dummy=delta_dummy,
                                                                            normalization_factor=1.0)
                            with numpy.errstate(divide='ignore'):
                                sigma = numpy.sqrt(a) / (b * normalization_factor)
                            sigma[b == 0] = dummy if dummy is not None else self._empty

        if (I is None) and ("csr" in method):
            with self._csr_sem:
                reset = None
                if self._csr_integrator is None:
                    reset = "init"
                if (not reset) and safe:
                    if self._csr_integrator.unit != unit:
                        reset = "unit changed"
                    if self._csr_integrator.bins != npt:
                        reset = "number of points changed"
                    if self._csr_integrator.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and\
                            (not self._csr_integrator.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (self._csr_integrator.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and\
                            (self._csr_integrator.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and\
                            (self._csr_integrator.pos0Range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and\
                            (self._csr_integrator.pos0Range !=
                             (min(radial_range), max(radial_range) * EPS32)):
                        reset = ("radial_range is defined"
                                 " but not the same as in CSR")
                    if (azimuth_range is None) and\
                            (self._csr_integrator.pos1Range is not None):
                        reset = ("azimuth_range not defined and"
                                 " CSR had azimuth_range defined")
                    elif (azimuth_range is not None) and\
                            (self._csr_integrator.pos1Range !=
                             (min(azimuth_range), max(azimuth_range) * EPS32)):
                        reset = ("azimuth_range requested and"
                                 " CSR's azimuth_range don't match")
                if reset:
                    logger.info("AI.integrate1d: Resetting integrator because %s", reset)
                    if "no" in method:
                        split = "no"
                    elif "full" in method:
                        split = "full"
                    else:
                        split = "bbox"
                    try:
                        self._csr_integrator = self.setup_CSR(shape, npt, mask,
                                                              radial_range, azimuth_range,
                                                              mask_checksum=mask_crc,
                                                              unit=unit, split=split)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        self._ocl_csr_integr = None
                        self._csr_integrator = None
                        gc.collect()
                        method = self.DEFAULT_METHOD
                if self._csr_integrator:
                    if ("ocl" in method) and ocl_azim_csr:
                        with self._ocl_csr_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_csr_integr is None) or\
                                    (self._ocl_csr_integr.on_device["data"] != self._csr_integrator.lut_checksum):
                                self._ocl_csr_integr = ocl_azim_csr.OCL_CSR_Integrator(self._csr_integrator.lut,
                                                                                       self._csr_integrator.size,
                                                                                       devicetype=devicetype,
                                                                                       platformid=platformid,
                                                                                       deviceid=deviceid,
                                                                                       checksum=self._csr_integrator.lut_checksum,
                                                                                       block_size=block_size,
                                                                                       profile=profile)
                            I, sum_, count = self._ocl_csr_integr.integrate(data, dark=dark, flat=flat,
                                                                            solidangle=solidangle,
                                                                            solidangle_checksum=self._dssa_crc,
                                                                            dummy=dummy,
                                                                            delta_dummy=delta_dummy,
                                                                            polarization=polarization,
                                                                            polarization_checksum=polarization_checksum,
                                                                            normalization_factor=normalization_factor)
                            qAxis = self._csr_integrator.outPos  # this will be copied later
                            if error_model == "azimuthal":
                                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                            if variance is not None:
                                var1d, a, b = self._ocl_csr_integr.integrate(variance,
                                                                             solidangle=None,
                                                                             dummy=dummy,
                                                                             delta_dummy=delta_dummy)
                                with numpy.errstate(divide='ignore'):
                                    sigma = numpy.sqrt(a) / (b * normalization_factor)
                                sigma[b == 0] = dummy if dummy is not None else self._empty
                    else:
                        qAxis, I, sum_, count = self._csr_integrator.integrate(data, dark=dark, flat=flat,
                                                                               solidAngle=solidangle,
                                                                               dummy=dummy,
                                                                               delta_dummy=delta_dummy,
                                                                               polarization=polarization,
                                                                               normalization_factor=normalization_factor)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                        if variance is not None:
                            _, var1d, a, b = self._csr_integrator.integrate(variance,
                                                                            solidAngle=None,
                                                                            dummy=dummy,
                                                                            delta_dummy=delta_dummy,
                                                                            normalization_factor=1.0)
                            with numpy.errstate(divide='ignore'):
                                sigma = numpy.sqrt(a) / (b * normalization_factor)
                            sigma[b == 0] = dummy if dummy is not None else self._empty

        if (I is None) and ("splitpix" in method):
#            if "full" in method:
                if splitPixel is None:
                    logger.warning("SplitPixelFull is not available,"
                                " falling back on splitbbox histogram !")
                    method = self.DEFAULT_METHOD
                else:
                    logger.debug("integrate1d uses SplitPixel implementation")
                    pos = self.array_from_unit(shape, "corner", unit, scale=False)
                    qAxis, I, sum_, count = splitPixel.fullSplit1D(pos=pos,
                                                                   weights=data,
                                                                   bins=npt,
                                                                   pos0Range=radial_range,
                                                                   pos1Range=azimuth_range,
                                                                   dummy=dummy,
                                                                   delta_dummy=delta_dummy,
                                                                   mask=mask,
                                                                   dark=dark,
                                                                   flat=flat,
                                                                   solidangle=solidangle,
                                                                   polarization=polarization,
                                                                   normalization_factor=normalization_factor
                                                                   )
                    if error_model == "azimuthal":
                        variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                    if variance is not None:
                        _, var1d, a, b = splitPixel.fullSplit1D(pos=pos,
                                                                weights=variance,
                                                                bins=npt,
                                                                pos0Range=radial_range,
                                                                pos1Range=azimuth_range,
                                                                dummy=dummy,
                                                                delta_dummy=delta_dummy,
                                                                mask=mask,
                                                                normalization_factor=1.0
                                                                )
                        with numpy.errstate(divide='ignore'):
                            sigma = numpy.sqrt(a) / (b * normalization_factor)
                        sigma[b == 0] = dummy if dummy is not None else self._empty

        if (I is None) and ("bbox" in method):
            if splitBBox is None:
                logger.warning("pyFAI.splitBBox is not available,"
                               " falling back on cython histograms")
                method = "cython"
            else:
                logger.debug("integrate1d uses BBox implementation")
                if chi is not None:
                    chi = chi
                    dchi = self.deltaChi(shape)
                else:
                    dchi = None
                pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
                qAxis, I, sum_, count = splitBBox.histoBBox1d(weights=data,
                                                              pos0=pos0,
                                                              delta_pos0=dpos0,
                                                              pos1=chi,
                                                              delta_pos1=dchi,
                                                              bins=npt,
                                                              pos0Range=radial_range,
                                                              pos1Range=azimuth_range,
                                                              dummy=dummy,
                                                              delta_dummy=delta_dummy,
                                                              mask=mask,
                                                              dark=dark,
                                                              flat=flat,
                                                              solidangle=solidangle,
                                                              polarization=polarization,
                                                              normalization_factor=normalization_factor)
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                if variance is not None:
                    _, var1d, a, b = splitBBox.histoBBox1d(weights=variance,
                                                           pos0=pos0,
                                                           delta_pos0=dpos0,
                                                           pos1=chi,
                                                           delta_pos1=dchi,
                                                           bins=npt,
                                                           pos0Range=radial_range,
                                                           pos1Range=azimuth_range,
                                                           dummy=dummy,
                                                           delta_dummy=delta_dummy,
                                                           mask=mask,
                                                           )
                    with numpy.errstate(divide='ignore'):
                        sigma = numpy.sqrt(a) / (b * normalization_factor)
                    sigma[b == 0] = dummy if dummy is not None else self._empty

        if I is None:
            # Common part for  Numpy and Cython
            data = data.astype(numpy.float32)
            mask = self.create_mask(data, mask, dummy, delta_dummy, mode="numpy")
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            if radial_range is not None:
                mask *= (pos0 >= min(radial_range))
                mask *= (pos0 <= max(radial_range))
            if azimuth_range is not None:
                chiMin, chiMax = azimuth_range
                chi = self.chiArray(shape)
                mask *= (chi >= chiMin) * (chi <= chiMax)
            mask = numpy.where(mask)
            pos0 = pos0[mask]
            if dark is not None:
                data -= dark
            if flat is not None:
                data /= flat
            if polarization is not None:
                data /= polarization
            if solidangle is not None:
                data /= solidangle
            data = data[mask]
            if variance is not None:
                variance = variance[mask]
            if radial_range is None:
                radial_range = (pos0.min(), pos0.max() * EPS32)

            if ("cython" in method):
                if histogram is not None:
                    logger.debug("integrate1d uses cython implementation")
                    qAxis, I, sum_, count = histogram.histogram(pos=pos0,
                                                                weights=data,
                                                                bins=npt,
                                                                pixelSize_in_Pos=0,
                                                                empty=dummy if dummy is not None else self._empty,
                                                                normalization_factor=normalization_factor)
                    if error_model == "azimuthal":
                        variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, correctSolidAngle=False)[mask]) ** 2
                    if variance is not None:
                        _, var1d, a, b = histogram.histogram(pos=pos0,
                                                             weights=variance,
                                                             bins=npt,
                                                             pixelSize_in_Pos=1,
                                                             empty=dummy if dummy is not None else self._empty)
                        with numpy.errstate(divide='ignore'):
                            sigma = numpy.sqrt(a) / (b * normalization_factor)
                        sigma[b == 0] = dummy if dummy is not None else self._empty
                else:
                    logger.warning("pyFAI.histogram is not available,"
                                   " falling back on numpy")
                    method = "numpy"

        if I is None:
            logger.debug("integrate1d uses Numpy implementation")
            method = "numpy"
            count, b = numpy.histogram(pos0, npt, range=radial_range)
            qAxis = (b[1:] + b[:-1]) / 2.0
            sum_, b = numpy.histogram(pos0, npt, weights=data, range=radial_range)
            with numpy.errstate(divide='ignore'):
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, correctSolidAngle=False)[mask]) ** 2
                if variance is not None:
                    var1d, b = numpy.histogram(pos0, npt, weights=variance, range=radial_range)
                    sigma = numpy.sqrt(var1d) / (count * normalization_factor)
                    sigma[count == 0] = dummy if dummy is not None else self._empty
                with numpy.errstate(divide='ignore'):
                    I = sum_ / count / normalization_factor
                I[count == 0] = dummy if dummy is not None else self._empty

        if pos0_scale:
            # not in place to make a copy
            qAxis = qAxis * pos0_scale

        result = Integrate1dResult(qAxis, I, sigma)
        result._set_method_called("integrate1d")
        result._set_compute_engine(method)
        result._set_unit(unit)
        result._set_sum(sum_)
        result._set_count(count)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_metadata(metadata)

        if filename is not None:
            writer = DefaultAiWriter(filename, self)
            writer.write(result)

        if all:
            logger.warning("integrate1d(all=True) is deprecated. "
                           "Please refer to the documentation of Integrate1dResult")

            res = {"radial": result.radial,
                   "unit": result.unit,
                   "I": result.intensity,
                   "sum": result.sum,
                   "count": result.count
                   }
            if result.sigma is not None:
                res["sigma"] = result.sigma
            return res

        return result
    
    def integrate_radial(self, data, npt, npt_rad=100,
                         correctSolidAngle=True,
                         radial_range=None, azimuth_range=None,
                         mask=None, dummy=None, delta_dummy=None,
                         polarization_factor=None, dark=None, flat=None,
                         method="csr", unit=units.CHI_DEG, radial_unit=units.Q,
                         normalization_factor=1.0,):
        """Calculate the radial integrated profile curve as I = f(chi)

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt: number of points in the output pattern
        :param int npt_rad: number of points in the radial space. Too few points may lead to huge rounding errors.
        :param str filename: output filename in 2/3 column ascii format
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :type radial_range: Tuple(float, float)
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :type azimuth_range: Tuple(float, float)
        :param ndarray mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :param float dummy: value for dead/masked pixels
        :param float delta_dummy: precision for dummy value
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
                * 0 for circular polarization or random,
                * None for no correction,
                * True for using the former correction
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param str method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "csr", "nosplit_csr", "full_csr", "lut_ocl" and "csr_ocl" if you want to go on GPU. To Specify the device: "csr_ocl_1,2"
        :param pyFAI.units.Unit unit: Output units, can be "chi_deg" or "chi_rad"
        :param pyFAI.units.Unit radial_unit: unit used for radial representation, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
        :param float normalization_factor: Value of a normalization monitor
        :return: chi bins center positions and regrouped intensity
        :rtype: Integrate1dResult
        """
        unit = units.to_unit(unit, type_=units.AZIMUTHAL_UNITS)
        res = self.integrate2d(data, npt_rad, npt,
                               correctSolidAngle=correctSolidAngle,
                               mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                               polarization_factor=polarization_factor,
                               dark=dark, flat=flat, method=method,
                               normalization_factor=normalization_factor,
                               radial_range=radial_range,
                               azimuth_range=azimuth_range,
                               unit=radial_unit)

        azim_scale = unit.scale / units.CHI_DEG.scale
        sum_ = res.sum.sum(axis=-1)
        count = res.count.sum(axis=-1)
        intensity = sum_ / count
        empty = dummy if dummy is not None else self.empty
        intensity[count == 0] = empty
        result = Integrate1dResult(res.azimuthal * azim_scale, intensity, None)
        result._set_method_called("integrate_radial")
        result._set_unit(unit)
        result._set_sum(sum_)
        result._set_count(count)
        result._set_has_dark_correction(dark is not None)
        result._set_has_flat_correction(flat is not None)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        return result

    def integrate2d(self, data, npt_rad, npt_azim=360,
                    filename=None, correctSolidAngle=True, variance=None,
                    error_model=None, radial_range=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    method="bbox", unit=units.Q, safe=True,
                    normalization_factor=1.0, all=False, metadata=None):
        """
        Calculate the azimuthal regrouped 2d image in q(nm^-1)/chi(deg) by default

        Multi algorithm implementation (tries to be bullet proof)

        :param data: 2D array from the Detector/CCD camera
        :type data: ndarray
        :param npt_rad: number of points in the radial direction
        :type npt_rad: int
        :param npt_azim: number of points in the azimuthal direction
        :type npt_azim: int
        :param filename: output image (as edf format)
        :type filename: str
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool
        :param variance: array containing the variance of the data. If not available, no error propagation is done
        :type variance: ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor between -1 (vertical) 
                and +1 (horizontal). 0 for circular polarization or random, 
                None for no correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "csr; "lut_ocl" and "csr_ocl" if you want to go on GPU. To Specify the device: "csr_ocl_1,2"
        :type method: str
        :param unit: Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
        :type unit: pyFAI.units.Unit
        :param safe: Do some extra checks to ensure LUT is still valid. False is faster.
        :type safe: bool
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param all: if true, return many more intermediate results as a dict (deprecated, please refer to the documentation of Integrate2dResult).
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :type all: bool
        :return: azimuthaly regrouped intensity, q/2theta/r pos. and chi pos.
        :rtype: Integrate2dResult, dict
        """
        if all:
            logger.warning("Deprecation: please use the object returned by ai.integrate2d, not the option `all`")
        method = method.lower()
        npt = (npt_rad, npt_azim)
        unit = units.to_unit(unit)
        pos0_scale = unit.scale
        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "provided"
            mask_crc = crc32(mask)

        shape = data.shape

        if radial_range:
            radial_range = tuple([i / pos0_scale for i in radial_range])

        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if azimuth_range is not None:
            azimuth_range = tuple(deg2rad(azimuth_range[i]) for i in (0, -1))
            if azimuth_range[1] <= azimuth_range[0]:
                azimuth_range = (azimuth_range[0], azimuth_range[1] + 2 * pi)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_checksum = None
        else:
            polarization, polarization_checksum = self.polarization(shape, polarization_factor, with_checksum=True)

        if dark is None:
            dark = self.detector.darkcurrent
            if dark is None:
                has_dark = False
            else:
                has_dark = "from detector"
        else:
            has_dark = "provided"

        if flat is None:
            flat = self.detector.flatfield
            if dark is None:
                has_flat = False
            else:
                has_flat = "from detector"
        else:
            has_flat = "provided"

        I = None
        sigma = None
        sum_ = None
        count = None

        if (I is None) and ("lut" in method):
            logger.debug("in lut")
            with self._lut_sem:
                reset = None
                if self._lut_integrator is None:
                    reset = "init"
                if (not reset) and safe:
                    if self._lut_integrator.unit != unit:
                        reset = "unit changed"
                    if self._lut_integrator.bins != npt:
                        reset = "number of points changed"
                    if self._lut_integrator.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and (not self._lut_integrator.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and (self._lut_integrator.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and (self._lut_integrator.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and (self._lut_integrator.pos0Range is not None):
                        reset = "radial_range was defined in LUT"
                    elif (radial_range is not None) and self._lut_integrator.pos0Range != (min(radial_range), max(radial_range) * EPS32):
                        reset = "radial_range is defined but not the same as in LUT"
                    if (azimuth_range is None) and (self._lut_integrator.pos1Range is not None):
                        reset = "azimuth_range not defined and LUT had azimuth_range defined"
                    elif (azimuth_range is not None) and self._lut_integrator.pos1Range != (min(azimuth_range), max(azimuth_range) * EPS32):
                        reset = "azimuth_range requested and LUT's azimuth_range don't match"
                error = False
                if reset:
                    logger.info("AI.integrate2d: Resetting integrator because %s", reset)
                    try:
                        self._lut_integrator = self.setup_LUT(shape, npt, mask, radial_range, azimuth_range, mask_checksum=mask_crc, unit=unit)
                        error = False
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        self._ocl_lut_integr = None
                        gc.collect()
                        method = self.DEFAULT_METHOD
                        error = True
                if not error:
                    if ("ocl" in method) and ocl_azim_lut:
                        with self._ocl_lut_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_lut_integr is None) or (self._ocl_lut_integr.on_device["lut"] != self._lut_integrator.lut_checksum):
                                self._ocl_lut_integr = ocl_azim_lut.OCL_LUT_Integrator(self._lut_integrator.lut,
                                                                                       self._lut_integrator.size,
                                                                                       devicetype=devicetype,
                                                                                       platformid=platformid,
                                                                                       deviceid=deviceid,
                                                                                       checksum=self._lut_integrator.lut_checksum)
                            if not error:
                                I, sum_, count = self._ocl_lut_integr.integrate(data, dark=dark, flat=flat,
                                                                                solidangle=solidangle,
                                                                                solidangle_checksum=self._dssa_crc,
                                                                                dummy=dummy,
                                                                                delta_dummy=delta_dummy,
                                                                                polarization=polarization,
                                                                                polarization_checksum=polarization_checksum,
                                                                                normalization_factor=normalization_factor,
                                                                                safe=safe)
                                I.shape = npt
                                I = I.T
                                bins_rad = self._lut_integrator.outPos0  # this will be copied later
                                bins_azim = self._lut_integrator.outPos1
                    else:
                        I, bins_rad, bins_azim, sum_, count = self._lut_integrator.integrate(data, dark=dark, flat=flat,
                                                                                             solidAngle=solidangle,
                                                                                             dummy=dummy,
                                                                                             delta_dummy=delta_dummy,
                                                                                             polarization=polarization,
                                                                                             normalization_factor=normalization_factor
                                                                                             )

        if (I is None) and ("csr" in method):
            logger.debug("in csr")
            with self._lut_sem:
                reset = None
                if self._csr_integrator is None:
                    reset = "init"
                if (not reset) and safe:
                    if self._csr_integrator.unit != unit:
                        reset = "unit changed"
                    if self._csr_integrator.bins != npt:
                        reset = "number of points changed"
                    if self._csr_integrator.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and (not self._csr_integrator.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (self._csr_integrator.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and (self._csr_integrator.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and (self._csr_integrator.pos0Range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and self._csr_integrator.pos0Range != (min(radial_range), max(radial_range) * EPS32):
                        reset = "radial_range is defined but not the same as in CSR"
                    if (azimuth_range is None) and (self._csr_integrator.pos1Range is not None):
                        reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and self._csr_integrator.pos1Range != (min(azimuth_range), max(azimuth_range) * EPS32):
                        reset = "azimuth_range requested and CSR's azimuth_range don't match"
                error = False
                if reset:
                    logger.info("AI.integrate2d: Resetting integrator because %s", reset)
                    if "no" in method:
                        split = "no"
                    elif "full" in method:
                        split = "full"
                    else:
                        split = "bbox"
                    try:
                        self._csr_integrator = self.setup_CSR(shape, npt, mask,
                                                              radial_range, azimuth_range,
                                                              mask_checksum=mask_crc,
                                                              unit=unit, split=split)
                        error = False
                    except MemoryError:
                        logger.warning("MemoryError: falling back on default forward implementation")
                        self._ocl_csr_integr = None
                        gc.collect()
                        method = self.DEFAULT_METHOD
                        error = True
                if not error:  # not yet implemented...
                    if ("ocl" in method) and ocl_azim_lut:
                        with self._ocl_lut_sem:
                            if "," in method:
                                c = method.index(",")
                                platformid = int(method[c - 1])
                                deviceid = int(method[c + 1])
                                devicetype = "all"
                            elif "gpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "gpu"
                            elif "cpu" in method:
                                platformid = None
                                deviceid = None
                                devicetype = "cpu"
                            else:
                                platformid = None
                                deviceid = None
                                devicetype = "all"
                            if (self._ocl_csr_integr is None) or (self._ocl_csr_integr.on_device["data"] != self._csr_integrator.lut_checksum):
                                self._ocl_csr_integr = ocl_azim_csr.OCL_CSR_Integrator(self._csr_integrator.lut,
                                                                                       self._csr_integrator.size,
                                                                                       devicetype=devicetype,
                                                                                       platformid=platformid,
                                                                                       deviceid=deviceid,
                                                                                       checksum=self._csr_integrator.lut_checksum)
                        if not error:
                                I, sum_, count = self._ocl_csr_integr.integrate(data, dark=dark, flat=flat,
                                                                                solidangle=solidangle,
                                                                                solidangle_checksum=self._dssa_crc,
                                                                                dummy=dummy,
                                                                                delta_dummy=delta_dummy,
                                                                                polarization=polarization,
                                                                                polarization_checksum=polarization_checksum,
                                                                                safe=safe,
                                                                                normalization_factor=normalization_factor)
                                I.shape = npt
                                I = I.T
                                bins_rad = self._csr_integrator.outPos0  # this will be copied later
                                bins_azim = self._csr_integrator.outPos1
                    else:
                        I, bins_rad, bins_azim, sum_, count = self._csr_integrator.integrate(data, dark=dark, flat=flat,
                                                                                             solidAngle=solidangle,
                                                                                             dummy=dummy,
                                                                                             delta_dummy=delta_dummy,
                                                                                             polarization=polarization,
                                                                                             normalization_factor=normalization_factor)

        if (I is None) and ("splitpix" in method):
            if splitPixel is None:
                logger.warning("splitPixel is not available;"
                               " falling back on default method")
                method = self.DEFAULT_METHOD
            else:
                logger.debug("integrate2d uses SplitPixel implementation")
                pos = self.array_from_unit(shape, "corner", unit, scale=False)
                I, bins_rad, bins_azim, sum_, count = splitPixel.fullSplit2D(pos=pos,
                                                                             weights=data,
                                                                             bins=(npt_rad, npt_azim),
                                                                             pos0Range=radial_range,
                                                                             pos1Range=azimuth_range,
                                                                             dummy=dummy,
                                                                             delta_dummy=delta_dummy,
                                                                             mask=mask,
                                                                             dark=dark,
                                                                             flat=flat,
                                                                             solidangle=solidangle,
                                                                             polarization=polarization,
                                                                             normalization_factor=normalization_factor)
        if (I is None) and ("bbox" in method):
            if splitBBox is None:
                logger.warning("splitBBox is not available;"
                               " falling back on cython histogram method")
                method = "cython"
            else:
                logger.debug("integrate2d uses BBox implementation")
                chi = self.chiArray(shape)
                dchi = self.deltaChi(shape)
                pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
                I, bins_rad, bins_azim, sum_, count = splitBBox.histoBBox2d(weights=data,
                                                                            pos0=pos0,
                                                                            delta_pos0=dpos0,
                                                                            pos1=chi,
                                                                            delta_pos1=dchi,
                                                                            bins=(npt_rad, npt_azim),
                                                                            pos0Range=radial_range,
                                                                            pos1Range=azimuth_range,
                                                                            dummy=dummy,
                                                                            delta_dummy=delta_dummy,
                                                                            mask=mask,
                                                                            dark=dark,
                                                                            flat=flat,
                                                                            solidangle=solidangle,
                                                                            polarization=polarization,
                                                                            normalization_factor=normalization_factor)

        if (I is None):
            logger.debug("integrate2d uses cython implementation")
            data = data.astype(numpy.float32)  # it is important to make a copy see issue #88
            mask = self.create_mask(data, mask, dummy, delta_dummy,
                                    mode="numpy")
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            pos1 = self.chiArray(shape)

            if radial_range is not None:
                mask *= (pos0 >= min(radial_range)) * (pos0 <= max(radial_range))
            else:
                radial_range = [pos0.min(), pos0.max() * EPS32]

            if azimuth_range is not None:
                mask *= (pos1 >= min(azimuth_range)) * (pos1 <= max(azimuth_range))
            else:
                azimuth_range = [pos1.min(), pos1.max() * EPS32]

            if variance is not None:
                variance = variance[mask]

            if dark is not None:
                data -= dark

            if flat is not None:
                data /= flat

            if polarization is not None:
                data /= polarization

            if solidangle is not None:
                data /= solidangle

            data = data[mask]
            pos0 = pos0[mask]
            pos1 = pos1[mask]
            if ("cython" in method):
                if histogram is None:
                    logger.warning("Cython histogram is not available;"
                                   " falling back on numpy histogram")
                    method = "numpy"
                else:
                    I, bins_azim, bins_rad, sum_, count = histogram.histogram2d(pos0=pos1,
                                                                                pos1=pos0,
                                                                                weights=data,
                                                                                bins=(npt_azim, npt_rad),
                                                                                split=False,
                                                                                empty=dummy if dummy is not None else self._empty,
                                                                                normalization_factor=normalization_factor)

        if I is None:
            logger.debug("integrate2d uses Numpy implementation")
            count, b, c = numpy.histogram2d(pos1, pos0, (npt_azim, npt_rad), range=[azimuth_range, radial_range])
            bins_azim = (b[1:] + b[:-1]) / 2.0
            bins_rad = (c[1:] + c[:-1]) / 2.0
            count1 = numpy.maximum(1, count)
            sum_, b, c = numpy.histogram2d(pos1, pos0, (npt_azim, npt_rad),
                                           weights=data, range=[azimuth_range, radial_range])
            I = sum_ / count1 / normalization_factor
            I[count == 0] = dummy if dummy is not None else self._empty
        # I know I make copies ....
        bins_rad = bins_rad * pos0_scale
        bins_azim = bins_azim * 180.0 / pi

        result = Integrate2dResult(I, bins_rad, bins_azim, sigma)
        result._set_method_called("integrate2d")
        result._set_compute_engine(method)
        result._set_unit(unit)
        result._set_count(count)
        result._set_sum(sum_)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_metadata(metadata)

        if filename is not None:
            writer = DefaultAiWriter(filename, self)
            writer.write(result)

        if all:
            logger.warning("integrate2d(all=True) is deprecated. Please refer to the documentation of Integrate2dResult")

            res = {"I": result.intensity,
                   "radial": result.radial,
                   "azimuthal": result.azimuthal,
                   "count": result.count,
                   "sum": result.sum
                   }
            if result.sigma is not None:
                res["sigma"] = result.sigma
            return res

        return result

    @deprecated
    def saxs(self, data, npt, filename=None,
             correctSolidAngle=True, variance=None,
             error_model=None, qRange=None, chiRange=None,
             mask=None, dummy=None, delta_dummy=None,
             polarization_factor=None, dark=None, flat=None,
             method="bbox", unit=units.Q):
        """
        Calculate the azimuthal integrated Saxs curve in q in nm^-1.

        Wrapper for integrate1d emulating behavour of old saxs method

        :param data: 2D array from the CCD camera
        :type data: ndarray
        :param npt: number of points in the output pattern
        :type npt: int
        :param filename: file to save data to
        :type filename: str
        :param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        :type correctSolidAngle: bool
        :param variance: array containing the variance of the data, if you know it
        :type variance: ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param qRange: The lower and upper range of the sctter vector q. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type qRange: (float, float), optional
        :param chiRange: The lower and upper range of the chi angle. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type chiRange: (float, float), optional
        :param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        :type mask: ndarray
        :param dummy: value for dead/masked pixels
        :type dummy: float
        :param delta_dummy: precision for dummy value
        :type delta_dummy: float
        :param polarization_factor: polarization factor between -1 and +1. 
                               0 for circular correction, None for no correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param method: can be "numpy", "cython", "BBox" or "splitpixel"
        :type method: str

        :return: azimuthaly regrouped data, 2theta pos. and chi pos.
        :rtype: 3-tuple of ndarrays
        """
        out = self.integrate1d(data, npt,
                               filename=filename,
                               correctSolidAngle=correctSolidAngle,
                               variance=variance,
                               error_model=error_model,
                               radial_range=qRange,
                               azimuth_range=chiRange,
                               mask=mask,
                               dummy=dummy,
                               delta_dummy=delta_dummy,
                               polarization_factor=polarization_factor,
                               dark=dark,
                               flat=flat,
                               method=method,
                               unit=unit)
        if len(out) == 2:
            return out[0], out[1], None
        else:
            return out

    @deprecated
    def save1D(self, filename, dim1, I, error=None, dim1_unit=units.TTH,
               has_dark=False, has_flat=False, polarization_factor=None, normalization_factor=None):
        """This method save the result of a 1D integration.
        
        Deprecated on 13/06/2017
        
        :param filename: the filename used to save the 1D integration
        :type filename: str
        :param dim1: the x coordinates of the integrated curve
        :type dim1: numpy.ndarray
        :param I: The integrated intensity
        :type I: numpy.mdarray
        :param error: the error bar for each intensity
        :type error: numpy.ndarray or None
        :param dim1_unit: the unit of the dim1 array
        :type dim1_unit: pyFAI.units.Unit
        :param has_dark: save the darks filenames (default: no)
        :type has_dark: bool
        :param has_flat: save the flat filenames (default: no)
        :type has_flat: bool
        :param polarization_factor: the polarization factor
        :type polarization_factor: float
        :param normalization_factor: the monitor value
        :type normalization_factor: float

        
        """
        if not filename:
            return
        writer = DefaultAiWriter(None, self)
        writer.save1D(filename, dim1, I, error, dim1_unit, has_dark, has_flat,
                      polarization_factor, normalization_factor)

    @deprecated
    def save2D(self, filename, I, dim1, dim2, error=None, dim1_unit=units.TTH,
               has_dark=False, has_flat=False,
               polarization_factor=None, normalization_factor=None):
        """This method save the result of a 2D integration.
        
        Deprecated on 13/06/2017
        
        :param filename: the filename used to save the 2D histogram
        :type filename: str
        :param dim1: the 1st coordinates of the histogram
        :type dim1: numpy.ndarray
        :param dim1: the 2nd coordinates of the histogram
        :type dim1: numpy.ndarray
        :param I: The integrated intensity
        :type I: numpy.mdarray
        :param error: the error bar for each intensity
        :type error: numpy.ndarray or None
        :param dim1_unit: the unit of the dim1 array
        :type dim1_unit: pyFAI.units.Unit
        :param has_dark: save the darks filenames (default: no)
        :type has_dark: bool
        :param has_flat: save the flat filenames (default: no)
        :type has_flat: bool
        :param polarization_factor: the polarization factor
        :type polarization_factor: float
        :param normalization_factor: the monitor value
        :type normalization_factor: float

        
        """
        if not filename:
            return
        writer = DefaultAiWriter(None, self)
        writer.save2D(filename, I, dim1, dim2, error, dim1_unit, has_dark, has_flat,
                      polarization_factor, normalization_factor)

    def medfilt1d(self, data, npt_rad=1024, npt_azim=512,
                  correctSolidAngle=True,
                  polarization_factor=None, dark=None, flat=None,
                  method="splitpixel", unit=units.Q,
                  percentile=50, mask=None, normalization_factor=1.0, metadata=None):
        """Perform the 2D integration and filter along each row using a median
        filter
        
        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool    
        :param polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
               0 for circular polarization or random,
               None for no correction,
               True for using the former correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param unit: unit to be used for integration
        :param method: pathway for integration and sort
        :param percentile: which percentile use for cutting out
                           percentil can be a 2-tuple to specify a region to 
                           average out
        :param mask: masked out pixels array
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param metadata: any other metadata, 
        :type metadata: JSON serializable dict
        :return: Integrate1D like result like
        """

        dummy = numpy.finfo(numpy.float32).min

        if "ocl" in method and npt_azim and (npt_azim - 1):
            old = npt_azim
            npt_azim = 1 << int(round(log(npt_azim, 2)))  # power of two above
            if npt_azim != old:
                logger.warning("Change number of azimuthal bins to nearest power of two: %s->%s",
                               old, npt_azim)
            # self._ocl_sem.acquire()
        res2d = self.integrate2d(data, npt_rad, npt_azim, mask=mask,
                                 flat=flat, dark=dark,
                                 unit=unit, method=method,
                                 dummy=dummy,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 normalization_factor=normalization_factor)
        integ2d = res2d.intensity
        if ("ocl" in method) and (ocl is not None):
            if "csr" in method and self._ocl_csr_integr:
                ctx = self._ocl_csr_integr.ctx
            elif "lut" in method and self._ocl_lut_integr:
                ctx = self._ocl_lut_integr.ctx
            else:
                ctx = None

            if numpy.isfortran(integ2d) and integ2d.dtype == numpy.float32:
                rdata = integ2d.T
                horizontal = True
            else:
                rdata = numpy.ascontiguousarray(integ2d, dtype=numpy.float32)
                horizontal = False

            if self._ocl_sorter:
                if self._ocl_sorter.npt_width != rdata.shape[1] or self._ocl_sorter.npt_height != rdata.shape[0]:
                    self._ocl_sorter = None
            if not self._ocl_sorter:
                logger.info("reset opencl sorter")
                self._ocl_sorter = ocl_sort.Separator(npt_height=rdata.shape[0], npt_width=rdata.shape[1], ctx=ctx)
            if "__len__" in dir(percentile):
                if horizontal:
                    spectrum = self._ocl_sorter.trimmed_mean_horizontal(rdata, dummy, [(i / 100.0) for i in percentile]).get()
                else:
                    spectrum = self._ocl_sorter.trimmed_mean_vertical(rdata, dummy, [(i / 100.0) for i in percentile]).get()
            else:
                if horizontal:
                    spectrum = self._ocl_sorter.filter_horizontal(rdata, dummy, percentile / 100.0).get()
                else:
                    spectrum = self._ocl_sorter.filter_vertical(rdata, dummy, percentile / 100.0).get()
        else:
            dummies = (integ2d == dummy).sum(axis=0)
            # add a line of zeros at the end (along npt_azim) so that the value for no valid pixel is 0
            sorted_ = numpy.zeros((npt_azim + 1, npt_rad))
            sorted_[:npt_azim, :] = numpy.sort(integ2d, axis=0)

            if "__len__" in dir(percentile):
                # mean over the valid value
                lower = dummies + (numpy.floor(min(percentile) * (npt_azim - dummies) / 100.)).astype(int)
                upper = dummies + (numpy.ceil(max(percentile) * (npt_azim - dummies) / 100.)).astype(int)
                bounds = numpy.zeros(sorted_.shape, dtype=int)
                assert (lower >= 0).all()
                assert (upper <= npt_azim).all()

                rng = numpy.arange(npt_rad)
                bounds[lower, rng] = 1
                bounds[upper, rng] = 1
                valid = (numpy.cumsum(bounds, axis=0) % 2)
                invalid = numpy.logical_not(valid)
                sorted_[invalid] = numpy.nan
                spectrum = numpy.nanmean(sorted_, axis=0)
            else:
                # read only the valid value
                dummies = (integ2d == dummy).sum(axis=0)
                pos = dummies + (numpy.round(percentile * (npt_azim - dummies) / 100.)).astype(int)
                assert (pos >= 0).all()
                assert (pos <= npt_azim).all()
                spectrum = sorted_[(pos, numpy.arange(npt_rad))]

        result = Integrate1dResult(res2d.radial, spectrum)
        result._set_method_called("medfilt1d")
        result._set_compute_engine(method)
        result._set_percentile(percentile)
        result._set_npt_azim(npt_azim)
        result._set_unit(unit)
        result._set_has_mask_applied(res2d.has_mask_applied)
        result._set_metadata(metadata)
        result._set_has_dark_correction(res2d.has_dark_correction)
        result._set_has_flat_correction(res2d.has_flat_correction)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        return result

    def sigma_clip(self, data, npt_rad=1024, npt_azim=512,
                   correctSolidAngle=True,
                   polarization_factor=None, dark=None, flat=None,
                   method="splitpixel", unit=units.Q,
                   thres=3, max_iter=5,
                   mask=None, normalization_factor=1.0, metadata=None):
        """Perform the 2D integration and perform a sigm-clipping iterative filter 
        along each row. see the doc of scipy.stats.sigmaclip for the options.
        
        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool    
        :param polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
               0 for circular polarization or random,
               None for no correction,
               True for using the former correction
        :type polarization_factor: float
        :param dark: dark noise image
        :type dark: ndarray
        :param flat: flat field image
        :type flat: ndarray
        :param unit: unit to be used for integration
        :param method: pathway for integration and sort
        :param thres: cut-off for n*sigma: discard any values with (I-<I>)/sigma > thres. 
                The threshold can be a 2-tuple with sigma_low and sigma_high.
        :param max_iter: maximum number of iterations        :param mask: masked out pixels array
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param metadata: any other metadata, 
        :type metadata: JSON serializable dict
        :return: Integrate1D like result like
        """
        # We use NaN as dummies
        dummy = numpy.NaN

        if "__len__" in dir(thres) and len(thres) > 0:
            sigma_lo = thres[0]
            sigma_hi = thres[-1]
        else:
            sigma_lo = sigma_hi = thres

        if "ocl" in method and npt_azim and (npt_azim - 1):
            old = npt_azim
            npt_azim = 1 << int(round(log(npt_azim, 2)))  # power of two above
            if npt_azim != old:
                logger.warning("Change number of azimuthal bins to nearest power of two: %s->%s",
                               old, npt_azim)

        res2d = self.integrate2d(data, npt_rad, npt_azim, mask=mask,
                                 flat=flat, dark=dark,
                                 unit=unit, method=method,
                                 dummy=dummy,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 normalization_factor=normalization_factor)
        image = res2d.intensity
        if ("ocl" in method) and (ocl is not None):
            if "csr" in method and self._ocl_csr_integr:
                ctx = self._ocl_csr_integr.ctx
            elif "lut" in method and self._ocl_lut_integr:
                ctx = self._ocl_lut_integr.ctx
            else:
                ctx = None

            if numpy.isfortran(image) and image.dtype == numpy.float32:
                rdata = image.T
                horizontal = True
            else:
                rdata = numpy.ascontiguousarray(image, dtype=numpy.float32)
                horizontal = False

            if self._ocl_sorter:
                if self._ocl_sorter.npt_width != rdata.shape[1] or self._ocl_sorter.npt_height != rdata.shape[0]:
                    self._ocl_sorter = None
            if not self._ocl_sorter:
                logger.info("reset opencl sorter")
                self._ocl_sorter = ocl_sort.Separator(npt_height=rdata.shape[0], npt_width=rdata.shape[1], ctx=ctx)
            if horizontal:
                res = self._ocl_sorter.sigma_clip_horizontal(rdata, dummy=dummy,
                                                             sigma_lo=sigma_lo,
                                                             sigma_hi=sigma_hi,
                                                             max_iter=max_iter)
            else:
                res = self._ocl_sorter.sigma_clip_vertical(rdata, dummy=dummy,
                                                           sigma_lo=sigma_lo,
                                                           sigma_hi=sigma_hi,
                                                           max_iter=max_iter)
            mean = res[0].get()
            std = res[1].get()
        else:
            as_strided = numpy.lib.stride_tricks.as_strided
            mask = numpy.logical_not(numpy.isfinite(image))
            dummies = mask.sum()
            image[mask] = numpy.NaN
            mean = numpy.nanmean(image, axis=0)
            std = numpy.nanstd(image, axis=0)
            for i in range(max_iter):
                mean2d = as_strided(mean, image.shape, (0, mean.strides[0]))
                std2d = as_strided(std, image.shape, (0, std.strides[0]))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    delta = (image - mean2d) / std2d
                    mask = numpy.logical_or(delta > sigma_hi,
                                            delta < -sigma_lo)
                dummies = mask.sum()
                if dummies == 0:
                    break
                image[mask] = numpy.NaN
                mean = numpy.nanmean(image, axis=0)
                std = numpy.nanstd(image, axis=0)

        result = Integrate1dResult(res2d.radial, mean, std)
        result._set_method_called("sigma_clip")
        result._set_compute_engine(method)
        result._set_percentile(thres)
        result._set_npt_azim(npt_azim)
        result._set_unit(unit)
        result._set_has_mask_applied(res2d.has_mask_applied)
        result._set_metadata(metadata)
        result._set_has_dark_correction(res2d.has_dark_correction)
        result._set_has_flat_correction(res2d.has_flat_correction)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        return result

    def separate(self, data, npt_rad=1024, npt_azim=512, unit="2th_deg", method="splitpixel",
                 percentile=50, mask=None, restore_mask=True):
        """
        Separate bragg signal from powder/amorphous signal using azimuthal integration,
        median filering and projected back before subtraction.

        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param unit: unit to be used for integration
        :param method: pathway for integration and sort
        :param percentile: which percentile use for cutting out
        :param mask: masked out pixels array
        :param restore_mask: masked pixels have the same value as input data provided
        :return: bragg, amorphous
        """

        radial, spectrum = self.medfilt1d(data, npt_rad=npt_rad, npt_azim=npt_azim,
                                          unit=unit, method=method,
                                          percentile=percentile, mask=mask)
        # This takes 100ms and is the next to be optimized.
        amorphous = self.calcfrom1d(radial, spectrum, data.shape, mask=None,
                                    dim1_unit=unit, correctSolidAngle=True)
        bragg = data - amorphous
        if restore_mask:
            wmask = numpy.where(mask)
            maskdata = data[wmask]
            bragg[wmask] = maskdata
            amorphous[wmask] = maskdata
        return bragg, amorphous

    def inpainting(self, data, mask, npt_rad=1024, npt_azim=512,
                   unit="r_m", method="splitpixel", poissonian=False,
                   grow_mask=3):
        """Re-invent the values of masked pixels

        :param data: input image as 2d numpy array
        :param mask: masked out pixels array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param unit: unit to be used for integration
        :param method: pathway for integration
        :param poissonian: If True, add some poisonian noise to the data to make
                           then more realistic
        :param grow_mask: grow mask in polar coordinated to accomodate pixel splitting algoritm
        
        :return: inpainting object which contains the restored image as .data 
        """
        from .ext import inpainting
        dummy = -1
        delta_dummy = 0.9

        assert mask.shape == self.detector.shape
        mask = numpy.ascontiguousarray(mask, numpy.int8)
        blank_data = numpy.zeros(mask.shape, dtype=numpy.float32)

        to_mask = numpy.where(mask)

        blank_mask = numpy.zeros_like(mask)
        masked = numpy.zeros(mask.shape, dtype=numpy.float32)
        masked[to_mask] = dummy

        masked_data = data.astype(numpy.float32)  # explicit copy
        masked_data[to_mask] = dummy

        if self.chiDiscAtPi:
            azimuth_range = (-180, 180)
        else:
            azimuth_range = (0, 360)
        r = self.array_from_unit(typ="corner", unit=unit, scale=True)
        rmax = (1.0 + numpy.finfo(numpy.float32).eps) * r[..., 0].max()
        kwargs = {"npt_rad": npt_rad,
                  "npt_azim": npt_azim,
                  "unit": unit,
                  "dummy": dummy,
                  "delta_dummy": delta_dummy,
                  "method": method,
                  "correctSolidAngle": False,
                  "mask": blank_mask,
                  "azimuth_range": azimuth_range,
                  "radial_range": (0, rmax),
                  "polarization_factor": None}
        imgb = self.integrate2d(blank_data, **kwargs)
        imgp = self.integrate2d(masked, **kwargs)
        imgd = self.integrate2d(masked_data, **kwargs)
        omask = numpy.ascontiguousarray(numpy.round(imgb.intensity / dummy), numpy.int8)
        imask = numpy.ascontiguousarray(numpy.round(imgp.intensity / dummy), numpy.int8)
        to_paint = (imask - omask)

        if grow_mask:
            # inpaint a bit more than needed to avoid "side" effects.
            from scipy.ndimage import binary_dilation
            structure = [[1], [1], [1]]
            to_paint = binary_dilation(to_paint, structure=structure, iterations=grow_mask)
            to_paint = to_paint.astype(numpy.int8)

        polar_inpainted = inpainting.polar_inpaint(imgd.intensity,
                                                   to_paint, omask, 0)
        r = self.array_from_unit(typ="center", unit=unit, scale=True)
        chi = numpy.rad2deg(self.chiArray())
        cart_inpatined = inpainting.polar_interpolate(data, mask,
                                                      r,
                                                      chi,
                                                      polar_inpainted,
                                                      imgd.radial, imgd.azimuthal)

        if poissonian:
            res = data.copy()
            res[to_mask] = numpy.random.poisson(cart_inpatined[to_mask])
        else:
            res = cart_inpatined
        return res

################################################################################
# Some properties
################################################################################

    def set_darkcurrent(self, dark):
        self.detector.set_darkcurrent(dark)

    def get_darkcurrent(self):
        return self.detector.get_darkcurrent()

    darkcurrent = property(get_darkcurrent, set_darkcurrent)

    def set_flatfield(self, flat):
        self.detector.set_flatfield(flat)

    def get_flatfield(self):
        return self.detector.get_flatfield()

    flatfield = property(get_flatfield, set_flatfield)

    def set_darkfiles(self, files=None, method="mean"):
        """Moved to Detector
        
        :param files: file(s) used to compute the dark.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str

        Set the dark current from one or mutliple files, avaraged
        according to the method provided
        """
        self.detector.set_darkfiles(files, method)

    @property
    def darkfiles(self):
        return self.detector.darkfiles

    def set_flatfiles(self, files, method="mean"):
        """Moved to Detector
        
        :param files: file(s) used to compute the flat-field.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str

        Set the flat field from one or mutliple files, averaged
        according to the method provided
        """
        self.detector.set_flatfiles(files, method)

    @property
    def flatfiles(self):
        return self.detector.flatfiles

    def get_empty(self):
        return self._empty

    def set_empty(self, value):
        self._empty = float(value)
        # propagate empty values to integrators
        for integrator in (self._ocl_integrator, self._ocl_lut_integr,
                           self._ocl_csr_integr, self._lut_integrator, self._csr_integrator):
            if integrator:
                integrator.empty = self._empty
    empty = property(get_empty, set_empty)
