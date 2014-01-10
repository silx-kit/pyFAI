#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/09/2013"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import os
import logging
logger = logging.getLogger("pyFAI.azimuthalIntegrator")
import tempfile
import types
import threading
import gc
import numpy
from math import pi
from numpy import rad2deg
EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)
from . import geometry
Geometry = geometry.Geometry
from . import units
from . import utils
import fabio
error = None
try:
    from . import ocl_azim  # IGNORE:F0401
    from . import opencl
except ImportError as error:  # IGNORE:W0703
    logger.warning("Unable to import pyFAI.ocl_azim")
    ocl_azim = None
    ocl = None
else:
    ocl = opencl.OpenCL()

try:
    from . import splitBBoxLUT
except ImportError as error:  # IGNORE:W0703
    logger.warning("Unable to import pyFAI.splitBBoxLUT for"
                 " Look-up table based azimuthal integration")
    splitBBoxLUT = None

try:
    from . import ocl_azim_lut
except ImportError as error:  # IGNORE:W0703
    logger.error("Unable to import pyFAI.ocl_azim_lut for"
                 " Look-up table based azimuthal integration on GPU")
    ocl_azim_lut = None

try:
    from .fastcrc import crc32
except ImportError:
    from zlib import crc32

try:
    from . import splitPixel  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.splitPixel"
                  " full pixel splitting: %s" % error)
    splitPixel = None

try:
    from . import splitBBox  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.splitBBox"
                 " Bounding Box pixel splitting: %s" % error)
    splitBBox = None

try:
    from . import histogram  # IGNORE:F0401
except ImportError as error:
    logger.error("Unable to import pyFAI.histogram"
                 " Cython OpenMP histogram implementation: %s" % error)
    histogram = None
del error  # just to see how clever pylint is !




class AzimuthalIntegrator(Geometry):
    """
    This class is an azimuthal integrator based on P. Boesecke's
    geometry and histogram algorithm by Manolo S. del Rio and V.A Sole

    All geometry calculation are done in the Geometry class

    main methods are:

        >>> tth, I = ai.integrate1d(data, nbPt, unit="2th_deg")
        >>> q, I, sigma = ai.integrate1d(data, nbPt, unit="q_nm^-1", error_model="poisson")
        >>> regrouped = ai.integrate2d(data, nbPt_rad, nbPt_azim, unit="q_nm^-1")[0]
    """

    def __init__(self, dist=1, poni1=0, poni2=0,
                 rot1=0, rot2=0, rot3=0,
                 pixel1=None, pixel2=None,
                 splineFile=None, detector=None, wavelength=None):
        """
        @param dist: distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        @type dist: float
        @param poni1: coordinate of the point of normal incidence along the detector's first dimension, in meter
        @type poni1: float
        @param poni2: coordinate of the point of normal incidence along the detector's second dimension, in meter
        @type poni2: float
        @param rot1: first rotation from sample ref to detector's ref, in radians
        @type rot1: float
        @param rot2: second rotation from sample ref to detector's ref, in radians
        @type rot2: float
        @param rot3: third rotation from sample ref to detector's ref, in radians
        @type rot3: float
        @param pixel1: pixel size of the fist dimension of the detector,  in meter
        @type pixel1: float
        @param pixel2: pixel size of the second dimension of the detector,  in meter
        @type pixel2: float
        @param splineFile: file containing the geometric distortion of the detector. Overrides the pixel size.
        @type splineFile: str
        @param detector: name of the detector or Detector instance.
        @type detector: str or pyFAI.Detector
        """
        Geometry.__init__(self, dist, poni1, poni2,
                          rot1, rot2, rot3,
                          pixel1, pixel2, splineFile, detector, wavelength)
        self._nbPixCache = {}  # key=shape, value: array

        #
        # mask and maskfile are properties pointing to self.detector

        self._flatfield = None
        self._darkcurrent = None
        self._flatfield_crc = None
        self._darkcurrent_crc = None
        self.flatfiles = None
        self.darkfiles = None

        self.header = None

        self._ocl_integrator = None
        self._ocl_lut_integr = None
        self._lut_integrator = None
        self._ocl_sem = threading.Semaphore()
        self._lut_sem = threading.Semaphore()
        self._ocl_lut_sem = threading.Semaphore()

    def reset(self):
        """
        Reset azimuthal integrator in addition to other arrays.
        """
        Geometry.reset(self)
        with self._ocl_sem:
            self._ocl_integrator = None
        with self._lut_sem:
            self._lut_integrator = None

    def makeMask(self, data, mask=None,
                 dummy=None, delta_dummy=None, mode="normal"):
        """
        Combines various masks into another one.

        @param data: input array of data
        @type data: ndarray
        @param mask: input mask (if none, self.mask is used)
        @type mask: ndarray
        @param dummy: value of dead pixels
        @type dummy: float
        @param delta_dumy: precision of dummy pixels
        @type delta_dummy: float
        @param mode: can be "normal" or "numpy" (inverted) or "where" applied to the mask
        @type mode: str

        @return: the new mask
        @rtype: ndarray of bool

        This method combine two masks (dynamic mask from *data &
        dummy* and *mask*) to generate a new one with the 'or' binary
        operation.  One can adjuste the level, with the *dummy* and
        the *delta_dummy* parameter, when you considere the *data*
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
            logger.debug("Mask likely to be inverted as more"
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

        @param data: input ndarray with the image
        @param dark: ndarray with dark noise or None
        @return: 2tuple: corrected_data, dark_actually used (or None)
        """
        if dark is not None:
            return data - dark, dark
        elif self._darkcurrent is not None:
            return data - self._darkcurrent, self._darkcurrent
        return data, None


    def flat_correction(self, data, flat=None):
        """
        Correct for flat field.
        If flat is not defined, correct for a flat set by "set_flatfiles"

        @param data: input ndarray with the image
        @param dark: ndarray with dark noise or None
        @return: 2tuple: corrected_data, flat_actually used (or None)
        """
        if flat is not None:
            return data / flat, flat
        if self._flatfield is not None:
            return data / self._flatfield, self._flatfield
        else:
            return data, None


    def xrpd_numpy(self, data, nbPt, filename=None, correctSolidAngle=True,
                   tthRange=None, mask=None, dummy=None, delta_dummy=None,
                   polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Numpy implementation: slow and without pixels splitting.
        This method should not be used in production, it remains
        to explain how other more sophisticated algorithms works.
        Use xrpd_splitBBox instead

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of the 2theta
        @type tthRange: (float, float), optional
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray

        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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
        correct your data. If set to 0 there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        mask = self.makeMask(data, mask, dummy, delta_dummy, mode="where")
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
            tthRange = (utils.deg2rad(tthRange[0]),
                        utils.deg2rad(tthRange[-1]) * EPS32)
        else:
            tthRange = (tth.min(), tth.max() * EPS32)
        if nbPt not in self._nbPixCache:
            ref, b = numpy.histogram(tth, nbPt, range=tthRange)
            self._nbPixCache[nbPt] = numpy.maximum(1, ref)

        val, b = numpy.histogram(tth,
                                 bins=nbPt,
                                 weights=data,
                                 range=tthRange)
        tthAxis = 90.0 * (b[1:] + b[:-1]) / pi
        I = val / self._nbPixCache[nbPt]
        self.save1D(filename, tthAxis, I, None, "2th_deg",
                    dark, flat, polarization_factor)
        return tthAxis, I

    def xrpd_cython(self, data, nbPt, filename=None, correctSolidAngle=True,
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
                                   nbPt=nbPt,
                                   filename=filename,
                                   correctSolidAngle=correctSolidAngle,
                                   tthRange=tthRange,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   polarization_factor=polarization_factor)

        mask = self.makeMask(data, mask, dummy, delta_dummy, mode="where")
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
            tthRange = (utils.deg2rad(tthRange[0]),utils.deg2rad(tthRange[-1]))
        if dummy is None:
            dummy = 0.0
        tthAxis, I, _, _ = histogram.histogram(pos=tth,
                                               weights=data,
                                               bins=nbPt,
                                               bin_range=tthRange,
                                               pixelSize_in_Pos=pixelSize,
                                               dummy=dummy)
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg",
                    dark, flat, polarization_factor)
        return tthAxis, I

    def xrpd_splitBBox(self, data, nbPt, filename=None, correctSolidAngle=True,
                       tthRange=None, chiRange=None, mask=None,
                       dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Cython implementation

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of the 2theta
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), optional, disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray

        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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
        correct your data. If set to 0 there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitBBox is None:
            logger.warning("Unable to use splitBBox,"
                           " falling back on numpy histogram !")
            return self.xrpd_numpy(data=data,
                                   nbPt=nbPt,
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
            tthRange = (utils.deg2rad(tthRange[0]),utils.deg2rad(tthRange[-1]))

        if chiRange is not None:
             chiRange = [utils.deg2rad(chiRange[0]), utils.deg2rad(chiRange[-1])]

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

        # ??? what about makeMask like with other methods
        if mask is None:
            mask = self.mask

        # outPos, outMerge, outData, outCount
        tthAxis, I, _, _ = splitBBox.histoBBox1d(weights=data,
                                                 pos0=tth,
                                                 delta_pos0=dtth,
                                                 pos1=chi,
                                                 delta_pos1=dchi,
                                                 bins=nbPt,
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
        self.save1D(filename, tthAxis, I, None, "2th_deg", dark, flat, polarization_factor)
        return tthAxis, I

    def xrpd_splitPixel(self, data, nbPt,
                        filename=None, correctSolidAngle=True,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        Cython implementation (single threaded)

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of the 2theta
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), optional, disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray

        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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
        correct your data. If set to 0 there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitPixel is None:
            logger.warning("splitPixel is not available,"
                           " falling back on numpy histogram !")
            return self.xrpd_numpy(data=data,
                                   nbPt=nbPt,
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
            tthRange = (utils.deg2rad(tthRange[0]),utils.deg2rad(tthRange[-1]))

        if chiRange is not None:
            chiRange = [utils.deg2rad(chiRange[0]), utils.deg2rad(chiRange[-1])]

        # ??? what about dark and flat computation like with other methods ?

        tthAxis, I, _, _ = splitPixel.fullSplit1D(pos=pos,
                                                  weights=data,
                                                  bins=nbPt,
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
                    dark, flat, polarization_factor)
        return tthAxis, I
    # Default implementation:
    xrpd = xrpd_splitBBox

    def xrpd_OpenCL(self, data, nbPt, filename=None, correctSolidAngle=True,
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

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of the 2theta
        @type tthRange: (float, float), optional
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float

        OpenCL specific parameters:

        @param devicetype: possible values "cpu", "gpu", "all" or "def"
        @type devicetype: str
        @param useFp64: shall histogram be done in double precision (strongly adviced)
        @type useFp64: bool
        @param platformid: platform number
        @type platformid: int
        @param deviceid: device number
        @type deviceid: int
        @param safe: set to False if your GPU is already set-up correctly
        @type safe: bool

        @return: (2theta, I) angle being in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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
                                       nbPt=nbPt,
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

                    if integr.getConfiguration(size, nbPt):
                        raise RuntimeError("Failed to configure 1D integrator"
                                           " with Ndata=%s and Nbins=%s"
                                           % (size, nbPt))

                    if integr.configure():
                        raise RuntimeError('Failed to compile kernel')
                    pos0 = self.twoThetaArray(shape)
                    delta_pos0 = self.delta2Theta(shape)
                    if tthRange is not None and len(tthRange) > 1:
                        pos0_min = utils.deg2rad(min(tthRange))
                        pos0_maxin = utils.deg2rad(max(tthRange))
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
                elif not correctSolidAngle and not param["solid_angle"] and (flat is not 1) :
                    self._ocl_integrator.setSolidAngle(flat)
                if (mask is not None) and not param["mask"]:
                    self._ocl_integrator.setMask(mask)
                elif (mask is None) and param["mask"]:
                    self._ocl_integrator.unsetMask()
            tthAxis, I, _, = self._ocl_integrator.execute(data)
        tthAxis = rad2deg(tthAxis)
        self.save1D(filename, tthAxis, I, None, "2th_deg")  # , dark, flat, polarization_factor)
        return tthAxis, I

    def setup_LUT(self, shape, nbPt, mask=None,
                  pos0_range=None, pos1_range=None, mask_checksum=None,
                  unit=units.TTH):
        """
        Prepare a look-up-table

        @param shape: shape of the dataset
        @type shape: (int, int)
        @param nbPt: number of points in the the output pattern
        @type nbPt: int or (int, int)
        @param mask: array with masked pixel (1=masked)
        @type mask: ndarray
        @param pos0_range: range in radial dimension
        @type pos0_range: (float, float)
        @param pos1_range: range in azimuthal dimension
        @type pos1_range: (float, float)
        @param mask_checksum: checksum of the mask buffer
        @type mask_checksum: int (or anything else ...)
        @param unit: use to propagate the LUT object for further checkings
        @type unit: pyFAI.units.Enum

        This method is called when a look-up table needs to be set-up.
        The *shape* parameter, correspond to the shape of the original
        datatset. It is possible to customize the number of point of
        the output histogram with the *nbPt* parameter which can be
        either an integer for an 1D integration or a 2-tuple of
        integer in case of a 2D integration. The LUT will have a
        different shape: (nbPt, lut_max_size), the later parameter
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

        if "__len__" in dir(nbPt) and len(nbPt) == 2:
            int2d = True
        else:
            int2d = False
        pos0 = self.array_from_unit(shape, "center", unit)
        dpos0 = self.array_from_unit(shape, "delta", unit)
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
                                            bins=nbPt,
                                            pos0Range=pos0Range,
                                            pos1Range=pos1Range,
                                            mask=mask,
                                            mask_checksum=mask_checksum,
                                            allow_pos0_neg=False,
                                            unit=unit)
        else:
            return splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                            bins=nbPt,
                                            pos0Range=pos0Range,
                                            pos1Range=pos1Range,
                                            mask=mask,
                                            mask_checksum=mask_checksum,
                                            allow_pos0_neg=False,
                                            unit=unit)

    def xrpd_LUT(self, data, nbPt, filename=None, correctSolidAngle=True,
                 tthRange=None, chiRange=None, mask=None,
                 dummy=None, delta_dummy=None,
                 safe=True, dark=None, flat=None):
        """
        Calculate the powder diffraction pattern from an image.

        Parallel Cython implementation using a Look-Up Table (OpenMP).

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of the 2theta angle
        @type tthRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), optional
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float

        LUT specific parameters:

        @param safe: set to False if your LUT is already set-up correctly (mask, ranges, ...).
        @type safe: bool

        @return: (2theta, I) with 2theta angle in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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

        TODO: replace with inegrate1D

        """

        if not splitBBoxLUT:
            logger.warning("Look-up table implementation not available:"
                           " falling back on old method !")
            return self.xrpd_splitBBox(data=data,
                                       nbPt=nbPt,
                                       filename=filename,
                                       correctSolidAngle=correctSolidAngle,
                                       tthRange=tthRange,
                                       mask=mask,
                                       dummy=dummy,
                                       delta_dummy=delta_dummy,
                                       flat=flat,
                                       dark=dark)
        return self.integrate1d(data,
                                nbPt,
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

    def xrpd_LUT_OCL(self, data, nbPt, filename=None, correctSolidAngle=True,
                     tthRange=None, chiRange=None, mask=None,
                     dummy=None, delta_dummy=None,
                     safe=True, devicetype="all",
                     platformid=None, deviceid=None, dark=None, flat=None):

        """
        Calculate the powder diffraction pattern from a set of data,
        an image.

        PyOpenCL implementation using a Look-Up Table (OpenCL). The
        look-up table is a Cython module.

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: integer
        @param filename: file to save data in ascii format 2 column
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of 2theta
        @type tthRange: (float, float)
        @param chiRange: The lower and upper range of the chi angle in degrees.
        @type chiRange: (float, float)
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float

        LUT specific parameters:

        @param safe: set to False if your LUT & GPU is already set-up correctly
        @type safe: bool

        OpenCL specific parameters:

        @param devicetype: can be "all", "cpu", "gpu", "acc" or "def"
        @type devicetype: str
        @param platformid: platform number
        @type platformid: int
        @param deviceid: device number
        @type deviceid: int

        @return: (2theta, I) in degrees
        @rtype: 2-tuple of 1D arrays

        This method compute the powder diffraction pattern, from a
        given *data* image. The number of point of the pattern is
        given by the *nbPt* parameter. If you give a *filename*, the
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
                                       nbPt=nbPt,
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
                                nbPt,
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

    def xrpd2_numpy(self, data, nbPt2Th, nbPtChi=360,
                    filename=None, correctSolidAngle=True,
                    dark=None, flat=None,
                    tthRange=None, chiRange=None,
                    mask=None, dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta, Chi) from
        a set of data, an image

        Pure numpy implementation (VERY SLOW !!!)

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of bin of the Radial (horizontal) axis (2Theta)
        @type nbPt: int
        @param nbPtChi: number of bin of the Azimuthal (vertical) axis (chi)
        @type nbPtChi: int
        @param filename: file to save data in
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of 2theta
        @type tthRange: (float, float)
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float

        @return: azimuthaly regrouped data, 2theta pos and chipos
        @rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is simular to
        a rectangular to polar conversion. The number of point of the
        new image is given by *nbPt2Th* and *nbPtChi*. If you give a
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
        mask = self.makeMask(data, mask, dummy, delta_dummy, mode="numpy")
        shape = data.shape
        tth = self.twoThetaArray(shape)[mask]
        chi = self.chiArray(shape)[mask]
        data, dark = self.dark_correction(data, dark)
        data, flat = self.flat_correction(data, flat)
        data = data.astype(numpy.float32)[mask]

        if correctSolidAngle is not None:
            data /= self.solidAngleArray(shape, correctSolidAngle)[mask]

        if tthRange is not None:
            tthRange = [deg2rad(tthRange[0]), deg2rad(tthRange[-1])]
        else:
            tthRange = [tth.min(), tth.max() * EPS32]

        if chiRange is not None:
            chiRange = [utils.deg2rad(chiRange[0]), utils.deg2rad(chiRange[-1])]
        else:
            chiRange = [chi.min(), chi.max() * EPS32]

        bins = (nbPtChi, nbPt2Th)
        if bins not in self._nbPixCache:
            ref, binsChi, bins2Th = numpy.histogram2d(chi, tth,
                                                      bins=list(bins),
                                                      range=[chiRange, tthRange])
            self._nbPixCache[bins] = numpy.maximum(1.0, ref)

        val, binsChi, bins2Th = numpy.histogram2d(chi, tth,
                                                  bins=list(bins),
                                                  weights=data,
                                                  range=[chiRange, tthRange])
        I = val / self._nbPixCache[bins]
        self.save2D(filename, I, bins2Th, binsChi)  # , dark, flat, polarization_factor)

        return I, bins2Th, binsChi

    def xrpd2_histogram(self, data, nbPt2Th, nbPtChi=360,
                        filename=None, correctSolidAngle=True,
                        dark=None, flat=None,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Cython implementation: fast but incaccurate

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of bin of the Radial (horizontal) axis (2Theta)
        @type nbPt: int
        @param nbPtChi: number of bin of the Azimuthal (vertical) axis (chi)
        @type nbPtChi: int
        @param filename: file to save data in
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of 2theta
        @type tthRange: (float, float)
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float

        @return: azimuthaly regrouped data, 2theta pos and chipos
        @rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is simular to
        a rectangular to polar conversion. The number of point of the
        new image is given by *nbPt2Th* and *nbPtChi*. If you give a
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
                                    nbPt2Th=nbPt2Th,
                                    nbPtChi=nbPtChi,
                                    filename=filename,
                                    correctSolidAngle=correctSolidAngle,
                                    tthRange=tthRange,
                                    chiRange=chiRange,
                                    mask=mask,
                                    dummy=dummy,
                                    delta_dummy=delta_dummy)

        mask = self.makeMask(data, mask, dummy, delta_dummy, mode="numpy")
        tth = self.twoThetaArray(data.shape)[mask]
        chi = self.chiArray(data.shape)[mask]
        data = data.astype(numpy.float32)[mask]

        # ??? idem here
        if dark is None:
            dark = self.darkcurrent
        if dark is not None:
            data -= dark[mask]

        if flat is None:
            flat = self.flatfield
        if flat is not None:
            data /= flat[mask]

        if correctSolidAngle is not None:
            data /= self.solidAngleArray(data.shape)[mask]

        if dummy is None:
            dummy = 0.0
            I, binsChi, bins2Th, _, _ = histogram.histogram2d(pos0=chi, pos1=tth,
                                      bins=(nbPtChi, nbPt2Th),
                                      weights=data,
                                      split=1,
                                      dummy=dummy)
        bins2Th = rad2deg(bins2Th)
        binsChi = rad2deg(binsChi)
        self.save2D(filename, I, bins2Th, binsChi)  # , dark, flat, polarization_factor)
        return I, bins2Th, binsChi

    def xrpd2_splitBBox(self, data, nbPt2Th, nbPtChi=360,
                        filename=None, correctSolidAngle=True,
                        tthRange=None, chiRange=None, mask=None,
                        dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Split pixels according to their coordinate and a bounding box

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of bin of the Radial (horizontal) axis (2Theta)
        @type nbPt: int
        @param nbPtChi: number of bin of the Azimuthal (vertical) axis (chi)
        @type nbPtChi: int
        @param filename: file to save data in
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of 2theta
        @type tthRange: (float, float)
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray

        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is similar to
        a rectangular to polar conversion. The number of point of the
        new image is given by *nbPt2Th* and *nbPtChi*. If you give a
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
        correct your data. If set to 0 there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitBBox is None:
            logger.warning("Unable to use splitBBox,"
                           " falling back on numpy histogram !")
            return self.xrpd2_histogram(data=data,
                                        nbPt2Th=nbPt2Th,
                                        nbPtChi=nbPtChi,
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
            tthRange = (utils.deg2rad(tthRange[0]), utils.deg2rad(tthRange[-1]))

        if chiRange is not None:
            chiRange = [utils.deg2rad(chiRange[0]), utils.deg2rad(chiRange[-1])]

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
                                  bins=(nbPt2Th, nbPtChi),
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
        self.save2D(filename, I, bins2Th, binsChi, dark=dark, flat=flat,
                    polarization_factor=polarization_factor)
        return I, bins2Th, binsChi

    def xrpd2_splitPixel(self, data, nbPt2Th, nbPtChi=360,
                         filename=None, correctSolidAngle=True,
                         tthRange=None, chiRange=None, mask=None,
                         dummy=None, delta_dummy=None,
                         polarization_factor=None, dark=None, flat=None):
        """
        Calculate the 2D powder diffraction pattern (2Theta,Chi) from
        a set of data, an image

        Split pixels according to their corner positions

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt2Th: number of bin of the Radial (horizontal) axis (2Theta)
        @type nbPt: int
        @param nbPtChi: number of bin of the Azimuthal (vertical) axis (chi)
        @type nbPtChi: int
        @param filename: file to save data in
        @type filename: str
        @param correctSolidAngle: solid angle correction, order 1 or 3 (like fit2d)
        @type correctSolidAngle: bool or int
        @param tthRange: The lower and upper range of 2theta
        @type tthRange: (float, float)
        @param chiRange: The lower and upper range of the chi angle.
        @type chiRange: (float, float), disabled for now
        @param mask: array with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels (dynamic mask)
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray

        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays

        This method convert the *data* image from the pixel
        coordinates to the 2theta, chi coordinates. This is similar to
        a rectangular to polar conversion. The number of point of the
        new image is given by *nbPt2Th* and *nbPtChi*. If you give a
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
        correct your data. If set to 0 there is no correction at all.

        The *dark* and the *flat* can be provided to correct the data
        before computing the radial integration.
        """
        if splitPixel is None:
            logger.warning("splitPixel is not available,"
                           " falling back on SplitBBox !")
            return self.xrpd2_splitBBox(
                data=data,
                nbPt2Th=nbPt2Th,
                nbPtChi=nbPtChi,
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
            tthRange = (utils.deg2rad(tthRange[0]), utils.deg2rad(tthRange[-1]))

        if chiRange is not None:
            chiRange = [utils.deg2rad(chiRange[0]), utils.deg2rad(chiRange[-1])]

        I, bins2Th, binsChi, _, _ = splitPixel.fullSplit2D(pos=pos,
                                   weights=data,
                                   bins=(nbPt2Th, nbPtChi),
                                   pos0Range=tthRange,
                                   pos1Range=chiRange,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   mask=mask,
                                   dark=dark,
                                   flat=flat,
                                   solidangle=solidangle,
                                   polarization=polarization,)
        bins2Th = rad2deg(bins2Th)
        binsChi = rad2deg(binsChi)
        self.save2D(filename, I, bins2Th, binsChi, dark=dark, flat=flat,
                    polarization_factor=polarization_factor)
        return I, bins2Th, binsChi
    xrpd2 = xrpd2_splitBBox

    def array_from_unit(self, shape, typ="center", unit=units.TTH):
        """
        Generate an array of position in different dimentions (R, Q,
        2Theta)

        @param shape: shape of the expected array
        @type shape: ndarray.shape
        @param typ: "center", "corner" or "delta"
        @type typ: str
        @param unit: can be Q, TTH, R for now
        @type unit: pyFAI.units.Enum

        @return: R, Q or 2Theta array depending on unit
        @rtype: ndarray
        """
        if not typ in ("center", "corner", "delta"):
            logger.warning("Unknown type of array %s,"
                           " defaulting to 'center'" % typ)
            typ = "center"
        unit = units.to_unit(unit)
        out = Geometry.__dict__[unit[typ]](self, shape)
        return out

    def integrate1d(self, data, nbPt, filename=None,
                    correctSolidAngle=True,
                    variance=None, error_model=None,
                    radial_range=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    method="lut", unit=units.Q, safe=True, normalization_factor=None):
        """
        Calculate the azimuthal integrated Saxs curve in q(nm^-1) by
        default

        Multi algorithm implementation (tries to be bullet proof)

        @param data: 2D array from the Detector/CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: int
        @param filename: output filename in 2/3 column ascii format
        @type filename: str
        @param correctSolidAngle: correct for solid angle of each pixel if True
        @type correctSolidAngle: bool
        @param variance: array containing the variance of the data. If not available, no error propagation is done
        @type variance: ndarray
        @param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        @type error_model: str
        @param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type radial_range: (float, float), optional
        @param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type azimuth_range: (float, float), optional
        @param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor between -1 and +1. 0 for no correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray
        @param method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "lut_ocl" if you want to go on GPU, ....
        @type method: str
        @param unit: can be Q, TTh, R for now
        @type unit: pyFAI.units.Enum
        @param safe: Do some extra checks to ensure LUT is still valid. False is faster.
        @type safe: bool
        @param normalization_factor: Value of a normalization monitor
        @type normalization_factor: float

        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays
        """
        method = method.lower()
        unit = units.to_unit(unit)
        pos0_scale = 1.0  # nota we need anyway to make a copy !

        if mask is None:
            mask = self.mask

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
            azimuth_range = tuple([numpy.deg2rad(i) for i in azimuth_range])
            chi = self.chiArray(shape)
        else:
            chi = None

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(shape, float(polarization_factor))

        if dark is None:
            dark = self.darkcurrent

        if flat is None:
            flat = self.flatfield

        I = None
        sigma = None

        if (I is None) and ("lut" in method):
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._lut_integrator is None:
                    reset = "init"
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                if (not reset) and safe:
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                    if self._lut_integrator.unit != unit:
                        reset = "unit changed"
                    if self._lut_integrator.bins != nbPt:
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
                error = False
                if reset:
                    logger.info("AI.integrate1d: Resetting integrator because %s" % reset)
                    try:
                        self._lut_integrator = self.setup_LUT(shape, nbPt, mask,
                                                              radial_range, azimuth_range,
                                                              mask_checksum=mask_crc, unit=unit)
                        error = False
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        self._ocl_lut_integr = None
                        gc.collect()
                        method = "splitbbox"
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
                            if (self._ocl_lut_integr is None) or\
                                    (self._ocl_lut_integr.on_device["lut"] != self._lut_integrator.lut_checksum):
                                self._ocl_lut_integr = ocl_azim_lut.OCL_LUT_Integrator(self._lut_integrator.lut,
                                                                                       self._lut_integrator.size,
                                                                                       devicetype=devicetype,
                                                                                       platformid=platformid,
                                                                                       deviceid=deviceid,
                                                                                       checksum=self._lut_integrator.lut_checksum)
                            I, _, _ = self._ocl_lut_integr.integrate(data, dark=dark, flat=flat,
                                                                     solidAngle=solidangle,
                                                                     solidAngle_checksum=self._dssa_crc,
                                                                     dummy=dummy,
                                                                     delta_dummy=delta_dummy,
                                                                     polarization=polarization,
                                                                     polarization_checksum=self._polarization_crc)
                            qAxis = self._lut_integrator.outPos  # this will be copied later
                            if error_model == "azimuthal":
                                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                            if variance is not None:
                                var1d, a, b = self._ocl_lut_integr.integrate(variance,
                                                                             solidAngle=None,
                                                                             dummy=dummy,
                                                                             delta_dummy=delta_dummy)
                                sigma = numpy.sqrt(a) / numpy.maximum(b, 1)
                    else:
                        qAxis, I, a, b = self._lut_integrator.integrate(data, dark=dark, flat=flat,
                                                           solidAngle=solidangle,
                                                           dummy=dummy,
                                                           delta_dummy=delta_dummy,
                                                           polarization=polarization)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                        if variance is not None:
                            _, var1d, a, b = self._lut_integrator.integrate(variance,
                                                               solidAngle=None,
                                                               dummy=dummy,
                                                               delta_dummy=delta_dummy)
                            sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if (I is None) and ("splitpix" in method):
            if splitPixel is None:
                logger.warning("SplitPixel is not available,"
                               " falling back on splitbbox histogram !")
                method = "bbox"
            else:
                logger.debug("integrate1d uses SplitPixel implementation")
                pos = self.array_from_unit(shape, "corner", unit)
                qAxis, I, a, b = splitPixel.fullSplit1D(pos=pos,
                                                        weights=data,
                                                        bins=nbPt,
                                                        pos0Range=radial_range,
                                                        pos1Range=azimuth_range,
                                                        dummy=dummy,
                                                        delta_dummy=delta_dummy,
                                                        mask=mask,
                                                        dark=dark,
                                                        flat=flat,
                                                        solidangle=solidangle,
                                                        polarization=polarization
                                                        )
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                if variance is not None:
                    _, var1d, a, b = splitPixel.fullSplit1D(pos=pos,
                                                            weights=variance,
                                                            bins=nbPt,
                                                            pos0Range=radial_range,
                                                            pos1Range=azimuth_range,
                                                            dummy=dummy,
                                                            delta_dummy=delta_dummy,
                                                            mask=mask,
                                                            )
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

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
                pos0 = self.array_from_unit(shape, "center", unit)
                dpos0 = self.array_from_unit(shape, "delta", unit)
                qAxis, I, a, b = splitBBox.histoBBox1d(weights=data,
                                                       pos0=pos0,
                                                       delta_pos0=dpos0,
                                                       pos1=chi,
                                                       delta_pos1=dchi,
                                                       bins=nbPt,
                                                       pos0Range=radial_range,
                                                       pos1Range=azimuth_range,
                                                       dummy=dummy,
                                                       delta_dummy=delta_dummy,
                                                       mask=mask,
                                                       dark=dark,
                                                       flat=flat,
                                                       solidangle=solidangle,
                                                       polarization=polarization)
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit)) ** 2
                if variance is not None:
                    _, var1d, a, b = splitBBox.histoBBox1d(weights=variance,
                                                           pos0=pos0,
                                                           delta_pos0=dpos0,
                                                           pos1=chi,
                                                           delta_pos1=dchi,
                                                           bins=nbPt,
                                                           pos0Range=radial_range,
                                                           pos1Range=azimuth_range,
                                                           dummy=dummy,
                                                           delta_dummy=delta_dummy,
                                                           mask=mask,
                                                           )
                    sigma = numpy.sqrt(a) / numpy.maximum(b, 1)

        if I is None:
            #Common part for  Numpy and Cython
            data = data.astype(numpy.float32)
            mask = self.makeMask(data, mask, dummy, delta_dummy, mode="numpy")
            pos0 = self.array_from_unit(shape, "center", unit)
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
                    if dummy is None:
                        dummy = 0
                    qAxis, I, a, b = histogram.histogram(pos=pos0,
                                                         weights=data,
                                                         bins=nbPt,
                                                         pixelSize_in_Pos=0,
                                                         dummy=dummy)
                    if error_model == "azimuthal":
                        variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, correctSolidAngle=False)[mask]) ** 2
                    if variance is not None:
                        _, var1d, a, b = histogram.histogram(pos=pos0,
                                                             weights=variance,
                                                             bins=nbPt,
                                                             pixelSize_in_Pos=1,
                                                             dummy=dummy)
                        sigma = numpy.sqrt(a) / numpy.maximum(b, 1)
                else:
                    logger.warning("pyFAI.histogram is not available,"
                               " falling back on numpy")
                    method = "numpy"
        if I is None:
            logger.debug("integrate1d uses Numpy implementation")
            method = "numpy"
            ref, b = numpy.histogram(pos0, nbPt, range=radial_range)
            qAxis = (b[1:] + b[:-1]) / 2.0
            count = numpy.maximum(1, ref)
            val, b = numpy.histogram(pos0, nbPt, weights=data, range=radial_range)
            if error_model == "azimuthal":
                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, correctSolidAngle=False)[mask]) ** 2
            if variance is not None:
                var1d, b = numpy.histogram(pos0, nbPt, weights=variance, range=radial_range)
                sigma = numpy.sqrt(var1d) / count
            I = val / count
        if pos0_scale:
            qAxis = qAxis * pos0_scale
        if normalization_factor:
            I /= normalization_factor
            if sigma is not None:
                sigma /= normalization_factor

        self.save1D(filename, qAxis, I, sigma, unit,
                    dark, flat, polarization_factor, normalization_factor)

        if sigma is not None:
            return qAxis, I, sigma
        else:
            return qAxis, I

    def integrate2d(self, data, nbPt_rad, nbPt_azim=360,
                    filename=None, correctSolidAngle=True, variance=None,
                    error_model=None, radial_range=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None,
                    method="bbox", unit=units.Q, safe=True,
                    normalization_factor=None):
        """
        Calculate the azimuthal regrouped 2d image in q(nm^-1)/deg by default

        Multi algorithm implementation (tries to be bullet proof)

        @param data: 2D array from the Detector/CCD camera
        @type data: ndarray
        @param nbPt_rad: number of points in the radial direction
        @type nbPt_rad: int
        @param nbPt_azim: number of points in the azimuthal direction
        @type nbPt_azim: int
        @param filename: output image (as edf format)
        @type filename: str
        @param correctSolidAngle: correct for solid angle of each pixel if True
        @type correctSolidAngle: bool
        @param variance: array containing the variance of the data. If not available, no error propagation is done
        @type variance: ndarray
        @param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        @type error_model: str
        @param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type radial_range: (float, float), optional
        @param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type azimuth_range: (float, float), optional
        @param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor between -1 and +1. 0 for no correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray
        @param method: can be "numpy", "cython", "BBox" or "splitpixel", "lut", "lut_ocl" if you want to go on GPU, ....
        @type method: str
        @param unit: can be Q, TTH, R for now
        @type unit: pyFAI.units.Enum
        @param safe: Do some extra checks to ensure LUT is still valid. False is faster.
        @type safe: bool
        @param normalization_factor: Value of a normalization monitor
        @type normalization_factor: float

        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays (2d, 1d, 1d)
        """
        method = method.lower()
        nbPt = (nbPt_rad, nbPt_azim)
        unit = units.to_unit(unit)
        pos0_scale = unit.scale
        if mask is None:
            mask = self.mask
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
            azimuth_range = tuple([numpy.deg2rad(i) for i in azimuth_range])

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = None
        else:
            polarization = self.polarization(shape, polarization_factor)

        if dark is None:
            dark = self.darkcurrent

        if flat is None:
            flat = self.flatfield

        I = None
        sigma = None

        if (I is None) and ("lut" in method):
            logger.debug("in lut")
            mask_crc = None
            with self._lut_sem:
                reset = None
                if self._lut_integrator is None:
                    reset = "init"
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                if (not reset) and safe:
                    if mask is None:
                        mask = self.detector.mask
                        mask_crc = self.detector._mask_crc
                    else:
                        mask_crc = crc32(mask)
                    if self._lut_integrator.unit != unit:
                        reset = "unit changed"
                    if self._lut_integrator.bins != nbPt:
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
                    logger.info("AI.integrate2d: Resetting integrator because %s" % reset)
                    try:
                        self._lut_integrator = self.setup_LUT(shape, nbPt, mask, radial_range, azimuth_range, mask_checksum=mask_crc, unit=unit)
                        error = False
                    except MemoryError:  # LUT method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        self._ocl_lut_integr = None
                        gc.collect()
                        method = "splitbbox"
                        error = True
                if not error:  # not yet implemented...
                    if  ("ocl" in method) and ocl_azim_lut:
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
                            I, _, _ = self._ocl_lut_integr.integrate(data, dark=dark, flat=flat,
                                                                     solidAngle=solidangle,
                                                                     solidAngle_checksum=self._dssa_crc,
                                                                     dummy=dummy,
                                                                     delta_dummy=delta_dummy,
                                                                     polarization=polarization,
                                                                     polarization_checksum=self._polarization_crc)
                            I.shape = nbPt
                            I = I.T
                            bins_rad = self._lut_integrator.outPos0  # this will be copied later
                            bins_azim = self._lut_integrator.outPos1
                    else:
                        I, bins_rad, bins_azim, _, _ = self._lut_integrator.integrate(data, dark=dark, flat=flat,
                                                                                      solidAngle=solidangle,
                                                                                      dummy=dummy,
                                                                                      delta_dummy=delta_dummy,
                                                                                      polarization=polarization)

        if (I is None) and ("splitpix" in method):
            if splitPixel is None:
                logger.warning("splitPixel is not available;"
                               " falling back on splitBBox method")
                method = "bbox"
            else:
                logger.debug("integrate2d uses SplitPixel implementation")
                pos = self.array_from_unit(shape, "corner", unit)
                I, bins_rad, bins_azim, _, _ = splitPixel.fullSplit2D(pos=pos,
                                                                      weights=data,
                                                                      bins=(nbPt_rad, nbPt_azim),
                                                                      pos0Range=radial_range,
                                                                      pos1Range=azimuth_range,
                                                                      dummy=dummy,
                                                                      delta_dummy=delta_dummy,
                                                                      mask=mask,
                                                                      dark=dark,
                                                                      flat=flat,
                                                                      solidangle=solidangle,
                                                                      polarization=polarization)
        if (I is None) and ("bbox" in method):
            if splitBBox is None:
                logger.warning("splitBBox is not available;"
                               " falling back on cython histogram method")
                method = "cython"
            else:
                logger.debug("integrate2d uses BBox implementation")
                chi = self.chiArray(shape)
                dchi = self.deltaChi(shape)
                pos0 = self.array_from_unit(shape, "center", unit)
                dpos0 = self.array_from_unit(shape, "delta", unit)
                I, bins_rad, bins_azim, _a, b = splitBBox.histoBBox2d(weights=data,
                                                                      pos0=pos0,
                                                                      delta_pos0=dpos0,
                                                                      pos1=chi,
                                                                      delta_pos1=dchi,
                                                                      bins=(nbPt_rad, nbPt_azim),
                                                                      pos0Range=radial_range,
                                                                      pos1Range=azimuth_range,
                                                                      dummy=dummy,
                                                                      delta_dummy=delta_dummy,
                                                                      mask=mask,
                                                                      dark=dark,
                                                                      flat=flat,
                                                                      solidangle=solidangle,
                                                                      polarization=polarization)

        if (I is None):
            logger.debug("integrate2d uses cython implementation")
            data = data.astype(numpy.float32) # it is important to make a copy see issue #88
            mask = self.makeMask(data, mask, dummy, delta_dummy,
                                 mode="numpy")
            pos0 = self.array_from_unit(shape, "center", unit)
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
                    if dummy is None:
                        dummy = 0
                    I, bins_azim, bins_rad, _a, _b = histogram.histogram2d(pos0=pos1,
                                                                           pos1=pos0,
                                                                           weights=data,
                                                                           bins=(nbPt_azim, nbPt_rad),
                                                                           split=False,
                                                                           dummy=dummy)

        if I is None:
            logger.debug("integrate2d uses Numpy implementation")
            ref, b, c = numpy.histogram2d(pos1, pos0, (nbPt_azim, nbPt_rad), range=[azimuth_range, radial_range])
            bins_azim = (b[1:] + b[:-1]) / 2.0
            bins_rad = (c[1:] + c[:-1]) / 2.0
            count = numpy.maximum(1, ref)
            val, b, c = numpy.histogram2d(pos1, pos0, (nbPt_azim, nbPt_rad),
                                          weights=data, range=[azimuth_range, radial_range])
            I = val / count
        # I know I make copies ....
        bins_rad = bins_rad * pos0_scale
        bins_azim = bins_azim * 180.0 / pi

        if normalization_factor:
            I /= normalization_factor

        self.save2D(filename, I, bins_rad, bins_azim, sigma, unit,
                    dark=dark, flat=flat, polarization_factor=polarization_factor,
                    normalization_factor=normalization_factor)
        if sigma is not None:
            return I, bins_rad, bins_azim, sigma
        else:
            return I, bins_rad, bins_azim


    def saxs(self, data, nbPt, filename=None,
             correctSolidAngle=True, variance=None,
             error_model=None, qRange=None, chiRange=None,
             mask=None, dummy=None, delta_dummy=None,
             polarization_factor=None, dark=None, flat=None,
             method="bbox", unit=units.Q):
        """
        Calculate the azimuthal integrated Saxs curve in q in nm^-1.

        Wrapper for integrate1d emulating behavour of old saxs method

        @param data: 2D array from the CCD camera
        @type data: ndarray
        @param nbPt: number of points in the output pattern
        @type nbPt: int
        @param filename: file to save data to
        @type filename: str
        @param correctSolidAngle: if True, the data are devided by the solid angle of each pixel
        @type correctSolidAngle: bool
        @param variance: array containing the variance of the data, if you know it
        @type variance: ndarray
        @param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        @type error_model: str
        @param qRange: The lower and upper range of the sctter vector q. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type qRange: (float, float), optional
        @param chiRange: The lower and upper range of the chi angle. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        @type chiRange: (float, float), optional
        @param mask: array (same size as image) with 1 for masked pixels, and 0 for valid pixels
        @type mask: ndarray
        @param dummy: value for dead/masked pixels
        @type dummy: float
        @param delta_dummy: precision for dummy value
        @type delta_dummy: float
        @param polarization_factor: polarization factor between -1 and +1. 0 for no correction
        @type polarization_factor: float
        @param dark: dark noise image
        @type dark: ndarray
        @param flat: flat field image
        @type flat: ndarray
        @param method: can be "numpy", "cython", "BBox" or "splitpixel"
        @type method: str

        @return: azimuthaly regrouped data, 2theta pos. and chi pos.
        @rtype: 3-tuple of ndarrays
        """
        out = self.integrate1d(data, nbPt,
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

    def makeHeaders(self, hdr="#", dark=None, flat=None,
                    polarization_factor=None, normalization_factor=None):
        """
        @param hdr: string used as comment in the header
        @type hdr: str
        @param dark: save the darks filenames (default: no)
        @type dark: ???
        @param flat: save the flat filenames (default: no)
        @type flat: ???
        @param polarization_factor: the polarization factor
        @type polarization_factor: float

        @return: the header
        @rtype: str
        """
        if self.header is None:
            headerLst = ["== pyFAI calibration =="]
            headerLst.append("SplineFile: %s" % self.splineFile)
            headerLst.append("PixelSize: %.3e, %.3e m" %
                             (self.pixel1, self.pixel2))
            headerLst.append("PONI: %.3e, %.3e m" % (self.poni1, self.poni2))
            headerLst.append("Distance Sample to Detector: %s m" %
                             self.dist)
            headerLst.append("Rotations: %.6f %.6f %.6f rad" %
                             (self.rot1, self.rot2, self.rot3))
            headerLst += ["", "== Fit2d calibration =="]
            f2d = self.getFit2D()
            headerLst.append("Distance Sample-beamCenter: %.3f mm" %
                             f2d["directDist"])
            headerLst.append("Center: x=%.3f, y=%.3f pix" %
                             (f2d["centerX"], f2d["centerY"]))
            headerLst.append("Tilt: %.3f deg  TiltPlanRot: %.3f deg" %
                             (f2d["tilt"], f2d["tiltPlanRotation"]))
            headerLst.append("")
            if self._wavelength is not None:
                headerLst.append("Wavelength: %s" % self.wavelength)
            if self.maskfile is not None:
                headerLst.append("Mask File: %s" % self.maskfile)
            if (dark is not None) or (self.darkcurrent is not None):
                if self.darkfiles:
                    headerLst.append("Dark current: %s" % self.darkfiles)
                else:
                    headerLst.append("Dark current: Done with unknown file")
            if (flat is not None) or (self.flatfield is not None):
                if self.flatfiles:
                    headerLst.append("Flat field: %s" % self.flatfiles)
                else:
                    headerLst.append("Flat field: Done with unknown file")
            if polarization_factor is None and self._polarization is not None:
                polarization_factor = self._polarization_factor
            headerLst.append("Polarization factor: %s" % polarization_factor)
            headerLst.append("Normalization factor: %s" % normalization_factor)
            self.header = os.linesep.join([hdr + " " + i for i in headerLst])
        return self.header

    def save1D(self, filename, dim1, I, error=None, dim1_unit=units.TTH,
               dark=None, flat=None, polarization_factor=None, normalization_factor=None):
        """
        @param filename: the filename used to save the 1D integration
        @type filename: str
        @param dim1: the x coordinates of the integrated curve
        @type dim1: numpy.ndarray
        @param I: The integrated intensity
        @type I: numpy.mdarray
        @param error: the error bar for each intensity
        @type error: numpy.ndarray or None
        @param dim1_unit: the unit of the dim1 array
        @type dim1_unit: pyFAI.units.Unit
        @param dark: save the darks filenames (default: no)
        @type dark: ???
        @param flat: save the flat filenames (default: no)
        @type flat: ???
        @param polarization_factor: the polarization factor
        @type polarization_factor: float
        @param normalization_factor: the monitor value
        @type normalization_factor: float

        This method save the result of a 1D integration.
        """
        dim1_unit = units.to_unit(dim1_unit)
        if filename:
            with open(filename, "w") as f:
                f.write(self.makeHeaders(dark=dark, flat=flat,
                                         polarization_factor=polarization_factor,
                                         normalization_factor=normalization_factor))
                f.write("%s# --> %s%s" % (os.linesep, filename, os.linesep))
                if error is None:
                    f.write("#%14s %14s %s" % (dim1_unit.REPR, "I ", os.linesep))
                    f.write(os.linesep.join(["%14.6e  %14.6e" % (t, i) for t, i in zip(dim1, I)]))
                else:
                    f.write("#%14s  %14s  %14s%s" %
                            (dim1_unit.REPR, "I ", "sigma ", os.linesep))
                    f.write(os.linesep.join(["%14.6e  %14.6e %14.6e" % (t, i, s) for t, i, s in zip(dim1, I, error)]))
                f.write(os.linesep)

    def save2D(self, filename, I, dim1, dim2, error=None, dim1_unit=units.TTH,
               dark=None, flat=None, polarization_factor=None, normalization_factor=None):
        """
        @param filename: the filename used to save the 2D histogram
        @type filename: str
        @param dim1: the 1st coordinates of the histogram
        @type dim1: numpy.ndarray
        @param dim1: the 2nd coordinates of the histogram
        @type dim1: numpy.ndarray
        @param I: The integrated intensity
        @type I: numpy.mdarray
        @param error: the error bar for each intensity
        @type error: numpy.ndarray or None
        @param dim1_unit: the unit of the dim1 array
        @type dim1_unit: pyFAI.units.Unit
        @param dark: save the darks filenames (default: no)
        @type dark: ???
        @param flat: save the flat filenames (default: no)
        @type flat: ???
        @param polarization_factor: the polarization factor
        @type polarization_factor: float
        @param normalization_factor: the monitor value
        @type normalization_factor: float

        This method save the result of a 2D integration.
        """
        if not filename:
            return

        dim1_unit = units.to_unit(dim1_unit)
        header_keys = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3",
                       "chi_min", "chi_max",
                       dim1_unit.REPR + "_min",
                       dim1_unit.REPR + "_max",
                       "pixelX", "pixelY",
                       "dark", "flat", "polarization_factor", "normalization_factor"]
        header = {"dist": str(self._dist),
                  "poni1": str(self._poni1),
                  "poni2": str(self._poni2),
                  "rot1": str(self._rot1),
                  "rot2": str(self._rot2),
                  "rot3": str(self._rot3),
                  "chi_min": str(dim2.min()),
                  "chi_max": str(dim2.max()),
                  dim1_unit.REPR + "_min": str(dim1.min()),
                  dim1_unit.REPR + "_max": str(dim1.max()),
                  "pixelX": str(self.pixel2),  # this is not a bug ... most people expect dim1 to be X
                  "pixelY": str(self.pixel1),  # this is not a bug ... most people expect dim2 to be Y
                  "polarization_factor": str(polarization_factor),
                  "normalization_factor":str(normalization_factor)
                  }

        if self.splineFile:
            header["spline"] = str(self.splineFile)

        if dark is not None:
            if self.darkfiles:
                header["dark"] = self.darkfiles
            else:
                header["dark"] = 'unknown dark applied'
        if flat is not None:
            if self.flatfiles:
                header["flat"] = self.flatfiles
            else:
                header["flat"] = 'unknown flat applied'
        f2d = self.getFit2D()
        for key in f2d:
            header["key"] = f2d[key]
        try:
            img = fabio.edfimage.edfimage(data=I.astype("float32"),
                                          header=header,
                                          header_keys=header_keys)

            if error is not None:
                img.appendFrame(data=error, header={"EDF_DataBlockID": "1.Image.Error"})
            img.write(filename)
        except IOError:
            logger.error("IOError while writing %s" % filename)

################################################################################
# Some properties
################################################################################

    def set_maskfile(self, maskfile):
        self.detector.set_maskfile(maskfile)

    def get_maskfile(self):
        return self.detector.get_maskfile()

    maskfile = property(get_maskfile, set_maskfile)

    def set_mask(self, mask):
        self.detector.set_mask(mask)

    def get_mask(self):
        return self.detector.get_mask()

    mask = property(get_mask, set_mask)

    def set_darkcurrent(self, dark):
        self._darkcurrent = dark
        if dark is not None:
            self._darkcurrent_crc = crc32(dark)
        else:
            self._darkcurrent_crc = None

    def get_darkcurrent(self):
        return self._darkcurrent

    darkcurrent = property(get_darkcurrent, set_darkcurrent)

    def set_flatfield(self, flat):
        self._flatfield = flat
        if flat is not None:
            self._flatfield_crc = crc32(flat)
        else:
            self._flatfield_crc = None

    def get_flatfield(self):
        return self._flatfield

    flatfield = property(get_flatfield, set_flatfield)

    def set_darkfiles(self, files=None, method="mean"):
        """
        @param files: file(s) used to compute the dark.
        @type files: str or list(str) or None
        @param method: method used to compute the dark, "mean" or "median"
        @type method: str

        Set the dark current from one or mutliple files, avaraged
        according to the method provided
        """
        if type(files) in types.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_darkcurrent(None)
        elif len(files) == 1:
            self.set_darkcurrent(fabio.open(files[0]).data.astype(numpy.float32))
            self.darkfiles = files[0]
        else:
            self.set_darkcurrent(utils.averageImages(files, filter_=method, format=None, threshold=0))
            self.darkfiles = "%s(%s)" % (method, ",".join(files))

    def set_flatfiles(self, files, method="mean"):
        """
        @param files: file(s) used to compute the dark.
        @type files: str or list(str) or None
        @param method: method used to compute the dark, "mean" or "median"
        @type method: str

        Set the flat field from one or mutliple files, averaged
        according to the method provided
        """
        if type(files) in types.StringTypes:
            files = [i.strip() for i in files.split(",")]
        elif not files:
            files = []
        if len(files) == 0:
            self.set_flatfield(None)
        elif len(files) == 1:
            self.set_flatfield(fabio.open(files[0]).data.astype(numpy.float32))
            self.flatfiles = files[0]
        else:
            self.set_flatfield(utils.averageImages(files, filter_=method, format=None, threshold=0))
            self.flatfiles = "%s(%s)" % (method, ",".join(files))
