#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2019 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/03/2022"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import warnings
import threading
import gc
from math import pi, log
import numpy
from numpy import rad2deg
from .geometry import Geometry
from . import units
from .utils import EPS32, deg2rad, crc32
from .utils.decorators import deprecated, deprecated_warning
from .containers import Integrate1dResult, Integrate2dResult, SeparateResult
from .io import DefaultAiWriter
error = None
from .method_registry import IntegrationMethod

from .engines.preproc import preproc as preproc_np

try:
    from .ext.preproc import preproc as preproc_cy
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    preproc = preproc_np
else:
    preproc = preproc_cy

from .load_integrators import ocl_azim_csr, ocl_azim_lut, ocl_sort, histogram, splitBBox, \
                                splitPixel, splitBBoxCSR, splitBBoxLUT, splitPixelFullCSR, \
                                histogram_engine, splitPixelFullLUT
from .engines import Engine

# Few constants for engine names:
OCL_CSR_ENGINE = "ocl_csr_integr"
OCL_LUT_ENGINE = "ocl_lut_integr"
OCL_HIST_ENGINE = "ocl_histogram"
OCL_SORT_ENGINE = "ocl_sorter"
EXT_LUT_ENGINE = "lut_integrator"
EXT_CSR_ENGINE = "csr_integrator"

PREFERED_METHODS_1D = IntegrationMethod.select_method(1, split="full", algo="histogram") + \
                      IntegrationMethod.select_method(1, split="pseudo", algo="histogram") + \
                      IntegrationMethod.select_method(1, split="bbox", algo="histogram") + \
                      IntegrationMethod.select_method(1, split="no", algo="histogram")
PREFERED_METHODS_2D = IntegrationMethod.select_method(2, split="full", algo="histogram") + \
                      IntegrationMethod.select_method(2, split="pseudo", algo="histogram") + \
                      IntegrationMethod.select_method(2, split="bbox", algo="histogram") + \
                      IntegrationMethod.select_method(2, split="no", algo="histogram")


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

    DEFAULT_METHOD_1D = PREFERED_METHODS_1D[0]
    DEFAULT_METHOD_2D = PREFERED_METHODS_2D[0]
    "Fail-safe low-memory integrator"

    USE_LEGACY_MASK_NORMALIZATION = True
    """If true, the Python engine integrator will normalize the mask to use the
    most frequent value of the mask as the non-masking value.

    This behaviour is not consistant with other engines and is now deprecated.
    This flag will be turned off in the comming releases.

    Turning off this flag force the user to provide a mask with 0 as non-masking
    value. And any non-zero as masking value (negative or positive value). A
    boolean mask is also accepted (`True` is the masking value).
    """

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

        # mask, maskfile, darkcurrent and flatfield are properties pointing to
        # self.detector now (16/06/2017)

        self._lock = threading.Semaphore()
        self.engines = {}  # key: name of the engine,

        self._empty = 0.0

    def reset(self):
        """Reset azimuthal integrator in addition to other arrays.
        """
        Geometry.reset(self)
        self.reset_engines()

    def reset_engines(self):
        """Urgently free memory by deleting all regrid-engines"""
        with self._lock:
            for key in list(self.engines.keys()):  # explicit copy
                self.engines.pop(key).reset()
        gc.collect()

    def create_mask(self, data, mask=None,
                    dummy=None, delta_dummy=None,
                    unit=None, radial_range=None,
                    azimuth_range=None,
                    mode="normal"):
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
            * "where": does a numpy.where on the "numpy" output

        This method tries to accomodate various types of masks (like
        valid=0 & masked=-1, ...) 
        
        Note for the developper: we use a lot of numpy.logical_or in this method,
        the out= argument allows to recycle buffers and save considerable time in 
        allocating temporary arrays.  
        """
        logical_or = numpy.logical_or
        shape = data.shape
        #       ^^^^   this is why data is mandatory !
        if mask is None:
            mask = self.mask
        if mask is None:
            mask = numpy.zeros(shape, dtype=bool)
        else:
            mask = mask.astype(bool)
        if self.USE_LEGACY_MASK_NORMALIZATION:
            if mask.sum(dtype=int) > mask.size // 2:
                reason = "The provided mask is not complient with other engines. "\
                    "The feature which automatically invert it will be removed soon. "\
                    "For more information see https://github.com/silx-kit/pyFAI/pull/868"
                deprecated_warning(__name__, name="provided mask content", reason=reason)
                numpy.logical_not(mask, mask)
        if (mask.shape != shape):
            try:
                mask = mask[:shape[0],:shape[1]]
            except Exception as error:  # IGNORE:W0703
                logger.error("Mask provided has wrong shape:"
                             " expected: %s, got %s, error: %s",
                             shape, mask.shape, error)
                mask = numpy.zeros(shape, dtype=bool)
        if dummy is not None:
            if delta_dummy is None:
                logical_or(mask, (data == dummy), out=mask)
            else:
                logical_or(mask, abs(data - dummy) <= delta_dummy, out=mask)

        if radial_range is not None:
            assert unit, "unit is needed when building a mask based on radial_range"
            rad = self.array_from_unit(shape, "center", unit, scale=False)
            logical_or(mask, rad < radial_range[0], out=mask)
            logical_or(mask, rad > radial_range[1], out=mask)
        if azimuth_range is not None:
            chi = self.chiArray(shape)
            logical_or(mask, chi < azimuth_range[0], out=mask)
            logical_or(mask, chi > azimuth_range[1], out=mask)

        # Prepare alternative representation for output:
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

    def _normalize_method(self, method, dim, default):
        """
        :rtype: IntegrationMethod
        """
        requested_method = method
        method = IntegrationMethod.select_one_available(method, dim=dim, default=None, degradable=False)
        if method is not None:
            return method
        method = IntegrationMethod.select_one_available(requested_method, dim=dim, default=default, degradable=True)
        logger.warning("Method requested '%s' not available. Method '%s' will be used", requested_method, method)
        return default

    def setup_LUT(self, shape, npt, mask=None,
                  pos0_range=None, pos1_range=None,
                  mask_checksum=None, unit=units.TTH,
                  split="bbox", empty=None, scale=True):
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
        :param empty: override the default empty value
        :param scale: set to False for working in S.I. units for pos0_range
                      which is faster. By default assumes pos0_range has `units`
                      Note that pos1_range, the chi-angle, is expected in radians


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
        if scale and pos0_range:
            unit = units.to_unit(unit)
            pos0_scale = unit.scale
            pos0_range = tuple(pos0_range[i] / pos0_scale for i in (0, -1))
        empty = self._empty if empty is None else empty
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
            if (pos1_range is None) and (not int2d):
                pos1 = None
                dpos1 = None
            else:
                pos1 = self.chiArray(shape)
                dpos1 = self.deltaChi(shape)

        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape
        if split == "full":
            if int2d:
                return splitPixelFullLUT.HistoLUT2dFullSplit(pos,
                                                bins=npt,
                                                pos0_range=pos0_range,
                                                pos1_range=pos1_range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                chiDiscAtPi=self.chiDiscAtPi,
                                                empty=empty)
            else:
                return splitPixelFullLUT.HistoLUT1dFullSplit(pos,
                                                             bins=npt,
                                                             pos0_range=pos0_range,
                                                             pos1_range=pos1_range,
                                                             mask=mask,
                                                             mask_checksum=mask_checksum,
                                                             allow_pos0_neg=False,
                                                             unit=unit,
                                                             empty=empty)
        else:
            if int2d:
                return splitBBoxLUT.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0_range=pos0_range,
                                                pos1_range=pos1_range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                empty=empty)
            else:
                return splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0_range=pos0_range,
                                                pos1_range=pos1_range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                empty=empty)

    def setup_CSR(self, shape, npt, mask=None,
                  pos0_range=None, pos1_range=None,
                  mask_checksum=None, unit=units.TTH,
                  split="bbox", empty=None, scale=True):
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
        :param empty: Override the empty value
        :param scale: set to False for working in S.I. units for pos0_range
                      which is faster. By default assumes pos0_range has `units`
                      Note that pos1_range, the chi-angle, is expected in radians

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

        if scale and pos0_range:
            unit = units.to_unit(unit)
            pos0_scale = unit.scale
            pos0_range = tuple(pos0_range[i] / pos0_scale for i in (0, -1))
        empty = self._empty if empty is None else empty
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
        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape
        if split == "full":

            if int2d:
                return splitPixelFullCSR.FullSplitCSR_2d(pos,
                                                         bins=npt,
                                                         pos0_range=pos0_range,
                                                         pos1_range=pos1_range,
                                                         mask=mask,
                                                         mask_checksum=mask_checksum,
                                                         allow_pos0_neg=False,
                                                         unit=unit,
                                                         chiDiscAtPi=self.chiDiscAtPi,
                                                         empty=empty)
            else:
                return splitPixelFullCSR.FullSplitCSR_1d(pos,
                                                         bins=npt,
                                                         pos0_range=pos0_range,
                                                         pos1_range=pos1_range,
                                                         mask=mask,
                                                         mask_checksum=mask_checksum,
                                                         allow_pos0_neg=False,
                                                         unit=unit,
                                                         empty=empty)
        else:
            if int2d:
                return splitBBoxCSR.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0_range=pos0_range,
                                                pos1_range=pos1_range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                empty=empty)
            else:
                return splitBBoxCSR.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                bins=npt,
                                                pos0_range=pos0_range,
                                                pos1_range=pos1_range,
                                                mask=mask,
                                                mask_checksum=mask_checksum,
                                                allow_pos0_neg=False,
                                                unit=unit,
                                                empty=empty)

    @deprecated(since_version="0.20", only_once=True, deprecated_since="0.20.0")
    def integrate1d_legacy(self, data, npt, filename=None,
                           correctSolidAngle=True,
                           variance=None, error_model=None,
                           radial_range=None, azimuth_range=None,
                           mask=None, dummy=None, delta_dummy=None,
                           polarization_factor=None, dark=None, flat=None,
                           method="csr", unit=units.Q, safe=True,
                           normalization_factor=1.0,
                           block_size=None, profile=False, metadata=None):
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
        :type method: can be Method named tuple, IntegrationMethod instance or str to be parsed
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
        :return: q/2th/r bins center positions and regrouped intensity (and error array if variance or variance model provided)
        :rtype: Integrate1dResult, dict
        """
        method = self._normalize_method(method, dim=1, default=self.DEFAULT_METHOD_1D)
        assert method.dimension == 1
        unit = units.to_unit(unit)

        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "provided"
            mask = numpy.ascontiguousarray(mask)
            mask_crc = crc32(mask)

        shape = data.shape
        pos0_scale = unit.scale

        if radial_range:
            radial_range = tuple(radial_range[i] / pos0_scale for i in (0, -1))
        if azimuth_range is not None:
            azimuth_range = self.normalize_azimuth_range(azimuth_range)

        if variance is not None:
            assert variance.size == data.size
        elif error_model:
            error_model = error_model.lower()
            if error_model == "poisson":
                variance = numpy.ascontiguousarray(data, numpy.float32)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(shape, polarization_factor, with_checksum=True)

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

        if method.algo_lower == "lut":
            if EXT_LUT_ENGINE not in self.engines:
                engine = self.engines[EXT_LUT_ENGINE] = Engine()
            else:
                engine = self.engines[EXT_LUT_ENGINE]
            with engine.lock:
                integr = engine.engine
                reset = None
                if integr is None:
                    reset = "init"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and\
                            (not integr.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and\
                            (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and\
                            (integr.pos0_range is not None):
                        reset = "radial_range was defined in LUT"
                    elif (radial_range is not None) and\
                            (integr.pos0_range != radial_range):
                        reset = ("radial_range is defined"
                                 " but not the same as in LUT")
                    if (azimuth_range is None) and\
                            (integr.pos1_range is not None):
                        reset = ("azimuth_range not defined and"
                                 " LUT had azimuth_range defined")
                    elif (azimuth_range is not None) and\
                            (integr.pos1_range != azimuth_range[0]):
                        reset = ("azimuth_range requested and"
                                 " LUT's azimuth_range don't match")
                if reset:
                    logger.info("AI.integrate1d: Resetting integrator because %s", reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        integr = self.setup_LUT(shape, npt, mask,
                                                radial_range, azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split,
                                                scale=False)

                    except MemoryError:
                        # LUT method is hungry...
                        logger.warning("MemoryError: falling back on default forward implementation")
                        integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        engine.set_engine(integr)
                if integr:
                    if method.impl_lower == "opencl":
                        # TODO: manage the target
                        if OCL_LUT_ENGINE in self.engines:
                            ocl_engine = self.engines[OCL_LUT_ENGINE]
                        else:
                            ocl_engine = self.engines[OCL_LUT_ENGINE] = Engine()
                        with ocl_engine.lock:
                            if method.target is not None:
                                platformid, deviceid = method.target
                            ocl_integr = ocl_engine.engine
                            if (ocl_integr is None) or \
                                    (ocl_integr.on_device["lut"] != integr.lut_checksum):
                                ocl_integr = ocl_azim_lut.OCL_LUT_Integrator(integr.lut,
                                                                             integr.size,
                                                                             platformid=platformid,
                                                                             deviceid=deviceid,
                                                                             checksum=integr.lut_checksum)
                                ocl_engine.set_engine(ocl_integr)
                            if ocl_integr is not None:
                                I, sum_, count = ocl_integr.integrate_legacy(data, dark=dark, flat=flat,
                                                                             solidangle=solidangle,
                                                                             solidangle_checksum=self._dssa_crc,
                                                                             dummy=dummy,
                                                                             delta_dummy=delta_dummy,
                                                                             polarization=polarization,
                                                                             polarization_checksum=polarization_crc,
                                                                             normalization_factor=normalization_factor)
                                qAxis = integr.bin_centers  # this will be copied later
                                if error_model == "azimuthal":

                                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
                                if variance is not None:
                                    var1d, a, b = ocl_integr.integrate_legacy(variance,
                                                                              solidangle=None,
                                                                              dummy=dummy,
                                                                              delta_dummy=delta_dummy,
                                                                              normalization_factor=1.0,
                                                                              coef_power=2)
                                    with numpy.errstate(divide='ignore', invalid='ignore'):
                                        sigma = numpy.sqrt(a) / (b * normalization_factor)
                                    sigma[b == 0] = dummy if dummy is not None else self._empty
                    else:
                        qAxis, I, sum_, count = integr.integrate_legacy(data, dark=dark, flat=flat,
                                                                        solidAngle=solidangle,
                                                                        dummy=dummy,
                                                                        delta_dummy=delta_dummy,
                                                                        polarization=polarization,
                                                                        normalization_factor=normalization_factor)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
                        if variance is not None:
                            _, var1d, a, b = integr.integrate_legacy(variance,
                                                                     solidAngle=None,
                                                                     dummy=dummy,
                                                                     delta_dummy=delta_dummy,
                                                                     coef_power=2,
                                                                     normalization_factor=1.0)
                            with numpy.errstate(divide='ignore', invalid='ignore'):
                                sigma = numpy.sqrt(a) / (b * normalization_factor)
                            sigma[b == 0] = dummy if dummy is not None else self._empty

        if method.algo_lower == "csr":
            if EXT_CSR_ENGINE not in self.engines:
                engine = self.engines[EXT_CSR_ENGINE] = Engine()
            else:
                engine = self.engines[EXT_CSR_ENGINE]
            with engine.lock:
                integr = engine.engine
                reset = None

                if integr is None:
                    reset = "init"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and\
                            (not integr.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and\
                            (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if radial_range != integr.pos0_range:
                        reset = "radial_range changed"
                    if azimuth_range != integr.pos1_range:
                        reset = "azimuth_range changed"
                if reset:
                    logger.info("AI.integrate1d: Resetting integrator because %s", reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        integr = self.setup_CSR(shape, npt, mask,
                                                radial_range, azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split,
                                                scale=False)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        engine.set_engine(integr)
                if integr:
                    if method.impl_lower == "opencl":
                        # TODO: manage OpenCL targets
                        if OCL_CSR_ENGINE not in self.engines:
                            self.engines[OCL_CSR_ENGINE] = Engine()
                        ocl_engine = self.engines[OCL_CSR_ENGINE]
                        with ocl_engine.lock:
                            if method.target is not None:
                                platformid, deviceid = method.target
                            ocl_integr = ocl_engine.engine
                            if (ocl_integr is None) or \
                                    (ocl_integr.on_device["data"] != integr.lut_checksum):
                                ocl_integr = ocl_azim_csr.OCL_CSR_Integrator(integr.lut,
                                                                             integr.size,
                                                                             platformid=platformid,
                                                                             deviceid=deviceid,
                                                                             checksum=integr.lut_checksum,
                                                                             block_size=block_size,
                                                                             profile=profile)
                                ocl_engine.set_engine(ocl_integr)
                            I, sum_, count = ocl_integr.integrate_legacy(data, dark=dark, flat=flat,
                                                                         solidangle=solidangle,
                                                                         solidangle_checksum=self._dssa_crc,
                                                                         dummy=dummy,
                                                                         delta_dummy=delta_dummy,
                                                                         polarization=polarization,
                                                                         polarization_checksum=polarization_crc,
                                                                         normalization_factor=normalization_factor)
                            qAxis = integr.bin_centers  # this will be copied later
                            if error_model == "azimuthal":
                                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
                            if variance is not None:
                                var1d, a, b = ocl_integr.integrate(variance,
                                                                   solidangle=None,
                                                                   dummy=dummy,
                                                                   delta_dummy=delta_dummy)
                                with numpy.errstate(divide='ignore', invalid='ignore'):
                                    sigma = numpy.sqrt(a) / (b * normalization_factor)
                                sigma[b == 0] = dummy if dummy is not None else self._empty
                    else:
                        qAxis, I, sum_, count = integr.integrate_legacy(data, dark=dark, flat=flat,
                                                                        solidAngle=solidangle,
                                                                        dummy=dummy,
                                                                        delta_dummy=delta_dummy,
                                                                        polarization=polarization,
                                                                        normalization_factor=normalization_factor)

                        if error_model == "azimuthal":
                            variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
                        if variance is not None:
                            _, var1d, a, b = integr.integrate_legacy(variance,
                                                                     solidAngle=None,
                                                                     dummy=dummy,
                                                                     delta_dummy=delta_dummy,
                                                                     normalization_factor=1.0)
                            with numpy.errstate(divide='ignore', invalid='ignore'):
                                sigma = numpy.sqrt(a) / (b * normalization_factor)
                            sigma[b == 0] = dummy if dummy is not None else self._empty

        if method.method[1:4] == ("full", "histogram", "cython"):
            logger.debug("integrate1d uses SplitPixel implementation")
            pos = self.array_from_unit(shape, "corner", unit, scale=False)
            qAxis, I, sum_, count = splitPixel.fullSplit1D(pos=pos,
                                                           weights=data,
                                                           bins=npt,
                                                           pos0_range=radial_range,
                                                           pos1_range=azimuth_range,
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
                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
            if variance is not None:
                _, var1d, a, b = splitPixel.fullSplit1D(pos=pos,
                                                        weights=variance,
                                                        bins=npt,
                                                        pos0_range=radial_range,
                                                        pos1_range=azimuth_range,
                                                        dummy=dummy,
                                                        delta_dummy=delta_dummy,
                                                        mask=mask,
                                                        normalization_factor=1.0
                                                        )
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    sigma = numpy.sqrt(a) / (b * normalization_factor)
                sigma[b == 0] = dummy if dummy is not None else self._empty

        if method.method[1:4] == ("bbox", "histogram", "cython"):
            logger.debug("integrate1d uses BBox implementation")
            if azimuth_range is not None:
                chi = self.chiArray(shape)
                dchi = self.deltaChi(shape)
            else:
                chi = None
                dchi = None
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
            qAxis, I, sum_, count = splitBBox.histoBBox1d(weights=data,
                                                          pos0=pos0,
                                                          delta_pos0=dpos0,
                                                          pos1=chi,
                                                          delta_pos1=dchi,
                                                          bins=npt,
                                                          pos0_range=radial_range,
                                                          pos1_range=azimuth_range,
                                                          dummy=dummy,
                                                          delta_dummy=delta_dummy,
                                                          mask=mask,
                                                          dark=dark,
                                                          flat=flat,
                                                          solidangle=solidangle,
                                                          polarization=polarization,
                                                          normalization_factor=normalization_factor)
            if error_model == "azimuthal":
                variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)) ** 2
            if variance is not None:
                _, var1d, a, b = splitBBox.histoBBox1d(weights=variance,
                                                       pos0=pos0,
                                                       delta_pos0=dpos0,
                                                       pos1=chi,
                                                       delta_pos1=dchi,
                                                       bins=npt,
                                                       pos0_range=radial_range,
                                                       pos1_range=azimuth_range,
                                                       dummy=dummy,
                                                       delta_dummy=delta_dummy,
                                                       mask=mask,
                                                       )
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    sigma = numpy.sqrt(a) / (b * normalization_factor)
                sigma[b == 0] = dummy if dummy is not None else self._empty

        if method.method[1:3] == ("no", "histogram") and method.impl_lower != "opencl":
            # Common part for  Numpy and Cython
            data = data.astype(numpy.float32)
            mask = self.create_mask(data, mask, dummy, delta_dummy,
                                    unit=unit,
                                    radial_range=radial_range,
                                    azimuth_range=azimuth_range,
                                    mode="where")
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            if radial_range is None:
                radial_range = (pos0.min(), pos0.max())
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

            if method.impl_lower == "cython":
                logger.debug("integrate1d uses cython implementation")
                qAxis, I, sum_, count = histogram.histogram(pos=pos0,
                                                            weights=data,
                                                            bins=npt,
                                                            bin_range=radial_range,
                                                            pixelSize_in_Pos=0,
                                                            empty=dummy if dummy is not None else self._empty,
                                                            normalization_factor=normalization_factor)
                if error_model == "azimuthal":
                    variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)[mask]) ** 2
                if variance is not None:
                    _, var1d, a, b = histogram.histogram(pos=pos0,
                                                         weights=variance,
                                                         bins=npt,
                                                         bin_range=radial_range,
                                                         pixelSize_in_Pos=1,
                                                         empty=dummy if dummy is not None else self._empty)
                    with numpy.errstate(divide='ignore', invalid='ignore'):
                        sigma = numpy.sqrt(a) / (b * normalization_factor)
                    sigma[b == 0] = dummy if dummy is not None else self._empty
            elif method.impl_lower == "python":
                logger.debug("integrate1d uses Numpy implementation")
                count, b = numpy.histogram(pos0, npt, range=radial_range)
                qAxis = (b[1:] + b[:-1]) / 2.0
                sum_, b = numpy.histogram(pos0, npt, weights=data, range=radial_range)
                with numpy.errstate(divide='ignore', invalid='ignore'):
                    if error_model == "azimuthal":
                        variance = (data - self.calcfrom1d(qAxis * pos0_scale, I, dim1_unit=unit, shape=shape)[mask]) ** 2
                    if variance is not None:
                        var1d, b = numpy.histogram(pos0, npt, weights=variance, range=radial_range)
                        sigma = numpy.sqrt(var1d) / (count * normalization_factor)
                        sigma[count == 0] = dummy if dummy is not None else self._empty
                    with numpy.errstate(divide='ignore', invalid='ignore'):
                        I = sum_ / count / normalization_factor
                    I[count == 0] = dummy if dummy is not None else self._empty

        if pos0_scale:
            # not in place to make a copy
            qAxis = qAxis * pos0_scale

        result = Integrate1dResult(qAxis, I, sigma)
        result._set_method_called("integrate1d")
        result._set_method(method)
        result._set_compute_engine(str(method))
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

        return result

    _integrate1d_legacy = integrate1d_legacy

    def integrate1d_ng(self, data, npt, filename=None,
                       correctSolidAngle=True,
                       variance=None, error_model=None,
                       radial_range=None, azimuth_range=None,
                       mask=None, dummy=None, delta_dummy=None,
                       polarization_factor=None, dark=None, flat=None,
                       method="csr", unit=units.Q, safe=True,
                       normalization_factor=1.0,
                       metadata=None):
        """Calculate the azimuthal integration (1d) of a 2D image.

        Multi algorithm implementation (tries to be bullet proof), suitable for SAXS, WAXS, ... and much more
        Takes extra care of normalization and performs proper variance propagation.

        :param ndarray data: 2D array from the Detector/CCD camera
        :param int npt: number of points in the output pattern
        :param str filename: output filename in 2/3 column ascii format
        :param bool correctSolidAngle: correct for solid angle of each pixel if True 
        :param ndarray variance: array containing the variance of the data. 
        :param str error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (min, max). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (min, max). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param ndarray mask: array with  0 for valid pixels, all other are masked (static mask)
        :param float dummy: value for dead/masked pixels (dynamic mask)
        :param float delta_dummy: precision for dummy value 
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
               0 for circular polarization or random,
               None for no correction,
               True for using the former correction 
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param Unit unit: Output units, can be "q_nm^-1" (default), "2th_deg", "r_mm" for now.
        :param bool safe: Perform some extra checks to ensure LUT/CSR is still valid. False is faster.
        :param float normalization_factor: Value of a normalization monitor 
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: Integrate1dResult namedtuple with (q,I,sigma) +extra informations in it.
        """
        method = self._normalize_method(method, dim=1, default=self.DEFAULT_METHOD_1D)
        assert method.dimension == 1
        unit = units.to_unit(unit)
        empty = dummy if dummy is not None else self._empty
        shape = data.shape
        pos0_scale = unit.scale

        if radial_range:
            radial_range = tuple(radial_range[i] / pos0_scale for i in (0, -1))
        if azimuth_range is not None:
            azimuth_range = self.normalize_azimuth_range(azimuth_range)

        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "user provided"
            mask = numpy.ascontiguousarray(mask)
            mask_crc = crc32(mask)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
            solidangle_crc = self._cached_array[f"solid_angle#{self._dssa_order}_crc"]
        else:
            solidangle_crc = solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(shape, polarization_factor, with_checksum=True)

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

        poissonian = None
        if variance is not None:
            assert variance.size == data.size
            do_variance = True
        elif error_model:
            error_model = error_model.lower()
            do_variance = True
            if error_model.startswith("poisson"):
                if method.manage_variance:
                    poissonian = True
                else:
                    if dark is None:
                        variance = numpy.maximum(data, 1.0).astype(numpy.float32)
                    else:
                        variance = (numpy.maximum(data, 1.0) + numpy.maximum(dark, 0.0)).astype(numpy.float32)
            elif error_model.startswith("azim") and method.manage_variance:
                poissonian = False
        else:
            do_variance = False
        # Prepare LUT if needed!
        if method.algo_lower in ("csr", "lut"):
            # initialize the CSR/LUT integrator in Cython as it may be needed later on.
            cython_method = IntegrationMethod.select_method(method.dimension, method.split_lower, method.algo_lower, "cython")[0]
            if cython_method not in self.engines:
                cython_engine = self.engines[cython_method] = Engine()
            else:
                cython_engine = self.engines[cython_method]
            with cython_engine.lock:
                # Validate that the engine used is the proper one
                cython_integr = cython_engine.engine
                cython_reset = None
                if cython_integr is None:
                    cython_reset = "of first initialization"
                if (not cython_reset) and safe:
                    if cython_integr.unit != unit:
                        cython_reset = "unit was changed"
                    if cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    if cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    if cython_integr.empty != empty:
                        cython_reset = "empty value changed"
                    if (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = f"mask but {method.algo_lower.upper()} was without mask"
                    elif (mask is None) and (cython_integr.cmask is not None):
                        cython_reset = f"no mask but { method.algo_lower.upper()} has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    if (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = f"radial_range was defined in { method.algo_lower.upper()}"
                    elif (radial_range is not None) and (cython_integr.pos0_range != radial_range):
                        cython_reset = f"radial_range is defined but differs in %s" % method.algo_lower.upper()
                    if (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = f"azimuth_range not defined and {method.algo_lower.upper()} had azimuth_range defined"
                    elif (azimuth_range is not None) and (cython_integr.pos1_range != azimuth_range):
                        cython_reset = f"azimuth_range requested and {method.algo_lower.upper()}'s azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.integrate1d_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        if method.algo_lower == "csr":
                            cython_integr = self.setup_CSR(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                        else:
                            cython_integr = self.setup_LUT(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        cython_integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        cython_engine.set_engine(cython_integr)
            # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
            if method.impl_lower == "cython":
                # The integrator has already been initialized previously
                integr = self.engines[method].engine
                intpl = integr.integrate_ng(data,
                                            variance=variance,
                                            poissonian=poissonian,
                                            dummy=dummy,
                                            delta_dummy=delta_dummy,
                                            dark=dark,
                                            flat=flat,
                                            solidangle=solidangle,
                                            polarization=polarization,
                                            normalization_factor=normalization_factor)
            else:  # method.impl_lower in ("opencl", "python"):
                if method not in self.engines:
                    # instanciated the engine
                    engine = self.engines[method] = Engine()
                else:
                    engine = self.engines[method]
                with engine.lock:
                    # Validate that the engine used is the proper one
                    integr = engine.engine
                    reset = None
                    if integr is None:
                        reset = "of first initialization"
                    if (not reset) and safe:
                        if integr.unit != unit:
                            reset = "unit was changed"
                        if integr.bins != npt:
                            reset = "number of points changed"
                        if integr.size != data.size:
                            reset = "input image size changed"
                        if integr.empty != empty:
                            reset = "empty value changed"
                        if (mask is not None) and (not integr.check_mask):
                            reset = f"mask but {method.algo_lower.upper()} was without mask"
                        elif (mask is None) and (integr.check_mask):
                            reset = f"no mask but {method.algo_lower.upper()} has mask"
                        elif (mask is not None) and (integr.mask_checksum != mask_crc):
                            reset = "mask changed"
                        if (radial_range is None) and (integr.pos0_range is not None):
                            reset = f"radial_range was defined in {method.algo_lower.upper()}"
                        elif (radial_range is not None) and (integr.pos0_range != radial_range):
                            reset = f"radial_range is defined but differs in {method.algo_lower.upper()}"
                        if (azimuth_range is None) and (integr.pos1_range is not None):
                            reset = f"azimuth_range not defined and {method.algo_lower.upper()} had azimuth_range defined"
                        elif (azimuth_range is not None) and (integr.pos1_range != azimuth_range):
                            reset = f"azimuth_range requested and {method.algo_lower.upper()}'s azimuth_range don't match"

                    if reset:
                        logger.info("ai.integrate1d_ng: Resetting ocl_csr integrator because %s", reset)
                        csr_integr = self.engines[cython_method].engine
                        if method.impl_lower == "opencl":
                            try:
                                integr = method.class_funct_ng.klass(csr_integr.lut,
                                                                     image_size=data.size,
                                                                     checksum=csr_integr.lut_checksum,
                                                                     empty=empty,
                                                                     unit=unit,
                                                                     bin_centers=csr_integr.bin_centers,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1],
                                                                     mask_checksum=csr_integr.mask_checksum)
                                # Copy some properties from the cython integrator
                                integr.pos0_range = csr_integr.pos0_range
                                integr.pos1_range = csr_integr.pos1_range
                                # print(csr_integr.mask_checksum, integr.mask_checksum)
                            except MemoryError:
                                logger.warning("MemoryError: falling back on default forward implementation")
                                self.reset_engines()
                                method = self.DEFAULT_METHOD_1D
                            else:
                                engine.set_engine(integr)
                        elif method.impl_lower == "python":
                            integr = method.class_funct_ng.klass(image_size=data.size,
                                                                 lut=csr_integr.lut,
                                                                 empty=empty,
                                                                 unit=unit,
                                                                 bin_centers=csr_integr.bin_centers,
                                                                 mask_checksum=csr_integr.mask_checksum)
                            # Copy some properties from the cython integrator
                            integr.pos0_range = csr_integr.pos0_range
                            integr.pos1_range = csr_integr.pos1_range
                            engine.set_engine(integr)
                        else:
                            raise RuntimeError("Unexpected configuration")

                    else:
                        integr = self.engines[method].engine
                kwargs = {"poissonian": None,
                          "variance": variance}
                if method.impl_lower == "opencl":
                    kwargs["polarization_checksum"] = polarization_crc
                    kwargs["solidangle_checksum"] = solidangle_crc
                if error_model:
                    if error_model.startswith("poisson"):
                        kwargs["poissonian"] = True
                        kwargs["variance"] = None
                    elif error_model.startswith("azim"):
                        kwargs["poissonian"] = False
                        kwargs["variance"] = None
                intpl = integr.integrate_ng(data, dark=dark,
                                            dummy=dummy, delta_dummy=delta_dummy,
                                            flat=flat, solidangle=solidangle,
                                            polarization=polarization,
                                            normalization_factor=normalization_factor,
                                            **kwargs)
            # This section is common to all 3 CSR implementations...
            if do_variance:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity,
                                           intpl.sigma)
                result._set_sum_variance(intpl.variance)
            else:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity)
            result._set_compute_engine(integr.__module__ + "." + integr.__class__.__name__)
            result._set_unit(integr.unit)
            result._set_sum_signal(intpl.signal)
            result._set_sum_normalization(intpl.normalization)
            result._set_count(intpl.count)
        # END of CSR/LUT common implementations
        elif (method.method[1:3] == ("no", "histogram") and
              method.method[3] in ("python", "cython")):
            integr = method.class_funct_ng.function  # should be histogram[_engine].histogram1d_engine
            if azimuth_range:
                chi_min, chi_max = azimuth_range
                chi = self.chiArray(shape)
                azim_mask = numpy.logical_or(chi > chi_max, chi < chi_min)
                if mask is None:
                    mask = azim_mask
                else:
                    mask = numpy.logical_or(mask, azim_mask)
            radial = self.array_from_unit(shape, "center", unit, scale=False)

            intpl = integr(radial, npt, data,
                           dark=dark,
                           dummy=dummy, delta_dummy=delta_dummy,
                           variance=variance,
                           flat=flat, solidangle=solidangle,
                           polarization=polarization,
                           normalization_factor=normalization_factor,
                           mask=mask,
                           radial_range=radial_range,
                           poissonian=poissonian)

            if do_variance:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity,
                                           intpl.sigma)
                result._set_sum_variance(intpl.variance)
            else:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity)
            result._set_compute_engine(integr.__module__ + "." + integr.__name__)
            result._set_unit(unit)
            result._set_sum_signal(intpl.signal)
            result._set_sum_normalization(intpl.normalization)
            result._set_count(intpl.count)
        elif method.method[1:4] == ("no", "histogram", "opencl"):
            if method not in self.engines:
                # instanciated the engine
                engine = self.engines[method] = Engine()
            else:
                engine = self.engines[method]
            with engine.lock:
                # Validate that the engine used is the proper one
                integr = engine.engine
                reset = None
                if integr is None:
                    reset = "of first initialization"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit was changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if integr.empty != empty:
                        reset = "empty value changed"
                if reset:
                    logger.info("ai.integrate1d: Resetting integrator because %s", reset)
                    pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                    azimuthal = self.chiArray(shape)
                    try:
                        integr = method.class_funct_ng.klass(pos0,
                                                             npt,
                                                             empty=empty,
                                                             azimuthal=azimuthal,
                                                             unit=unit,
                                                             mask=mask,
                                                             mask_checksum=mask_crc,
                                                             platformid=method.target[0],
                                                             deviceid=method.target[1])
                    except MemoryError:
                        logger.warning("MemoryError: falling back on default forward implementation")
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        engine.set_engine(integr)
                intpl = integr(data, dark=dark,
                               dummy=dummy,
                               delta_dummy=delta_dummy,
                               variance=variance,
                               flat=flat, solidangle=solidangle,
                               polarization=polarization,
                               polarization_checksum=polarization_crc,
                               normalization_factor=normalization_factor,
                               radial_range=radial_range,
                               azimuth_range=azimuth_range,
                               poissonian=poissonian)

            if do_variance:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity,
                                           intpl.sigma)
                result._set_sum_variance(intpl.variance)
            else:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity)
            result._set_compute_engine(integr.__module__ + "." + integr.__class__.__name__)
            result._set_unit(integr.unit)
            result._set_sum_signal(intpl.signal)
            result._set_sum_normalization(intpl.normalization)
            result._set_count(intpl.count)
        elif (method.method[2:4] == ("histogram", "cython")):
            integr = method.class_funct_ng.function  # should be histogram[_engine].histogram1d_engine
            if method.method[1] == "bbox":
                if azimuth_range is None:
                    chi = None
                    delta_chi = None
                else:
                    chi = self.chiArray(shape)
                    delta_chi = self.deltaChi(shape)
                radial = self.array_from_unit(shape, "center", unit, scale=False)
                delta_radial = self.array_from_unit(shape, "delta", unit, scale=False)
                intpl = integr(weights=data, variance=variance,
                               pos0=radial, delta_pos0=delta_radial,
                               pos1=chi, delta_pos1=delta_chi,
                               bins=npt,
                               dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                               dark=dark, flat=flat, solidangle=solidangle,
                               polarization=polarization,
                               normalization_factor=normalization_factor,
                               mask=mask,
                               pos0_range=radial_range,
                               pos1_range=azimuth_range)
            elif method.method[1] == "full":
                pos = self.array_from_unit(shape, "corner", unit, scale=False)
                intpl = integr(weights=data, variance=variance,
                               pos=pos,
                               bins=npt,
                               dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                               dark=dark, flat=flat, solidangle=solidangle,
                               polarization=polarization,
                               normalization_factor=normalization_factor,
                               mask=mask,
                               pos0_range=radial_range,
                               pos1_range=azimuth_range)
            else:
                raise RuntimeError("Should not arrive here")
            if do_variance:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity,
                                           intpl.sigma)
                result._set_sum_variance(intpl.variance)
            else:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity)
            result._set_compute_engine(integr.__module__ + "." + integr.__name__)
            result._set_unit(unit)
            result._set_sum_signal(intpl.signal)
            result._set_sum_normalization(intpl.normalization)
            result._set_count(intpl.count)

        else:
            raise RuntimeError("Fallback method ... should no more be used: %s" % method)
            if radial_range:
                radial_range = tuple(radial_range[i] * pos0_scale for i in (0, -1))
            if azimuth_range is not None:
                azimuth_range = tuple(rad2deg(azimuth_range[i]) for i in (0, -1))

            logger.warning("Failed to find method: %s", method)
            kwargs = {"npt": npt,
                      "error_model": None,
                      "variance": None,
                      "correctSolidAngle": False,
                      "polarization_factor": None,
                      "flat": None,
                      "radial_range": radial_range,
                      "azimuth_range": azimuth_range,
                      "mask": mask,
                      "dummy": dummy,
                      "delta_dummy": delta_dummy,
                      "method": method,
                      "unit": unit,
                      }

            normalization_image = numpy.ones(data.shape) * normalization_factor
            if correctSolidAngle:
                normalization_image *= self.solidAngleArray(self.detector.shape)

            if polarization_factor:
                normalization_image *= self.polarization(self.detector.shape, factor=polarization_factor)

            if flat is not None:
                normalization_image *= flat

            norm = self.integrate1d(normalization_image, **kwargs)
            signal = self._integrate1d_legacy(data, dark=dark, ** kwargs)
            sigma2 = self._integrate1d_legacy(variance, **kwargs)
            result = Integrate1dResult(norm.radial * unit.scale,
                                       signal.sum / norm.sum,
                                       numpy.sqrt(sigma2.sum) / norm.sum)
            result._set_compute_engine(norm.compute_engine)
            result._set_unit(signal.unit)
            result._set_sum_signal(signal.sum)
            result._set_sum_normalization(norm.sum)
            result._set_sum_variance(sigma2.sum)
            result._set_count(signal.count)
        result._set_method(method)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_method_called("integrate1d_ng")
        result._set_metadata(metadata)
        if filename is not None:
            writer = DefaultAiWriter(filename, self)
            writer.write(result)

        return result

    _integrate1d_ng = integrate1d_ng
    integrate1d = integrate1d_ng

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
        res = self.integrate2d_ng(data, npt_rad, npt,
                                  correctSolidAngle=correctSolidAngle,
                                  mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                  polarization_factor=polarization_factor,
                                  dark=dark, flat=flat, method=method,
                                  normalization_factor=normalization_factor,
                                  radial_range=radial_range,
                                  azimuth_range=azimuth_range,
                                  unit=radial_unit)

        azim_scale = unit.scale / units.CHI_DEG.scale

        sum_signal = res.sum_signal.sum(axis=-1)
        count = res.count.sum(axis=-1)
        sum_normalization = res._sum_normalization.sum(axis=-1)

        mask = numpy.where(count == 0)
        empty = dummy if dummy is not None else self._empty
        intensity = sum_signal / sum_normalization
        intensity[mask] = empty

        if res.sigma is not None:
            sum_variance = res.sum_variance.sum(axis=-1)
            sigma = numpy.sqrt(sum_variance) / sum_normalization
            sigma[mask] = empty
        else:
            sum_variance = None
            sigma = None
        result = Integrate1dResult(res.azimuthal * azim_scale, intensity, sigma)
        result._set_method_called("integrate_radial")
        result._set_unit(unit)
        result._set_sum_normalization(sum_normalization)
        result._set_count(count)
        result._set_sum_signal(sum_signal)
        result._set_sum_variance(sum_variance)
        result._set_has_dark_correction(dark is not None)
        result._set_has_flat_correction(flat is not None)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_method = res.method
        result._set_compute_engine = res.compute_engine

        return result

    @deprecated(since_version="0.21", only_once=True, deprecated_since="0.21.0")
    def integrate2d_legacy(self, data, npt_rad, npt_azim=360,
                            filename=None, correctSolidAngle=True, variance=None,
                            error_model=None, radial_range=None, azimuth_range=None,
                            mask=None, dummy=None, delta_dummy=None,
                            polarization_factor=None, dark=None, flat=None,
                            method=None, unit=units.Q, safe=True,
                            normalization_factor=1.0, metadata=None):
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
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        assert method.dimension == 2
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
            mask = numpy.ascontiguousarray(mask)
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
            self.check_chi_disc(azimuth_range)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(shape, polarization_factor, with_checksum=True)

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

        if method.algo_lower == "lut":
            if EXT_LUT_ENGINE not in self.engines:
                engine = self.engines[EXT_LUT_ENGINE] = Engine()
            else:
                engine = self.engines[EXT_LUT_ENGINE]
            with engine.lock:
                integr = engine.engine
                reset = None
                if integr is None:
                    reset = "init"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and (not integr.check_mask):
                        reset = "mask but LUT was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but LUT has mask"
                    elif (mask is not None) and (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if radial_range != integr.pos0_range:
                        reset = "radial_range changed"
                    if azimuth_range != integr.pos1_range:
                        reset = "azimuth_range changed"
                error = False
                if reset:
                    logger.info("ai.integrate2d: Resetting integrator because %s", reset)
                    try:
                        integr = self.setup_LUT(shape, npt, mask, radial_range, azimuth_range,
                                                mask_checksum=mask_crc, unit=unit, scale=False)
                    except MemoryError:
                        # LUT method is hungry im memory...
                        logger.warning("MemoryError: falling back on forward implementation")
                        integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_2D
                        error = True
                    else:
                        error = False
                        engine.set_engine(integr)
                if not error:
                    if method.impl_lower == "opencl":
                        if OCL_LUT_ENGINE in self.engines:
                            ocl_engine = self.engines[OCL_LUT_ENGINE]
                        else:
                            ocl_engine = self.engines[OCL_LUT_ENGINE] = Engine()
                        with ocl_engine.lock:
                            platformid, deviceid = method.target
                            ocl_integr = ocl_engine.engine
                            if (ocl_integr is None) or \
                                    (ocl_integr.on_device["lut"] != integr.lut_checksum):
                                ocl_integr = ocl_azim_lut.OCL_LUT_Integrator(integr.lut,
                                                                             integr.size,
                                                                             platformid=platformid,
                                                                             deviceid=deviceid,
                                                                             checksum=integr.lut_checksum)
                                ocl_engine.set_engine(ocl_integr)

                            if (not error) and (ocl_integr is not None):
                                I, sum_, count = ocl_integr.integrate(data, dark=dark, flat=flat,
                                                                      solidangle=solidangle,
                                                                      solidangle_checksum=self._dssa_crc,
                                                                      dummy=dummy,
                                                                      delta_dummy=delta_dummy,
                                                                      polarization=polarization,
                                                                      polarization_checksum=polarization_crc,
                                                                      normalization_factor=normalization_factor,
                                                                      safe=safe)
                                I.shape = npt
                                I = I.T
                                bins_rad = integr.bin_centers0  # this will be copied later
                                bins_azim = integr.bin_centers1
                    else:
                        I, bins_rad, bins_azim, sum_, count = integr.integrate(data, dark=dark, flat=flat,
                                                                               solidAngle=solidangle,
                                                                               dummy=dummy,
                                                                               delta_dummy=delta_dummy,
                                                                               polarization=polarization,
                                                                               normalization_factor=normalization_factor
                                                                               )

        if method.algo_lower == "csr":
            if EXT_CSR_ENGINE not in self.engines:
                engine = self.engines[EXT_CSR_ENGINE] = Engine()
            else:
                engine = self.engines[EXT_CSR_ENGINE]
            with engine.lock:
                integr = engine.engine
                reset = None
                if integr is None:
                    reset = "init"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if (mask is not None) and (not integr.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    if (radial_range is None) and (integr.pos0_range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                        reset = "radial_range is defined but differs in CSR"
                    if (azimuth_range is None) and (integr.pos1_range is not None):
                        reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                        reset = "azimuth_range requested and CSR's azimuth_range don't match"
                error = False
                if reset:
                    logger.info("AI.integrate2d: Resetting integrator because %s", reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        integr = self.setup_CSR(shape, npt, mask,
                                                radial_range, azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split,
                                                scale=False)
                    except MemoryError:
                        logger.warning("MemoryError: falling back on default forward implementation")
                        integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_2D
                        error = True
                    else:
                        error = False
                        engine.set_engine(integr)
                if not error:
                    if method.impl_lower == "opencl":
                        if OCL_CSR_ENGINE in self.engines:
                            ocl_engine = self.engines[OCL_CSR_ENGINE]
                        else:
                            ocl_engine = self.engines[OCL_CSR_ENGINE] = Engine()
                        with ocl_engine.lock:
                            platformid, deviceid = method.target
                            ocl_integr = ocl_engine.engine
                            if (ocl_integr is None) or (ocl_integr.on_device["data"] != integr.lut_checksum):
                                ocl_integr = ocl_azim_csr.OCL_CSR_Integrator(integr.lut,
                                                                             integr.size,
                                                                             platformid=platformid,
                                                                             deviceid=deviceid,
                                                                             checksum=integr.lut_checksum)
                                ocl_engine.set_engine(ocl_integr)
                        if (not error) and (ocl_integr is not None):
                                I, sum_, count = ocl_integr.integrate(data, dark=dark, flat=flat,
                                                                      solidangle=solidangle,
                                                                      solidangle_checksum=self._dssa_crc,
                                                                      dummy=dummy,
                                                                      delta_dummy=delta_dummy,
                                                                      polarization=polarization,
                                                                      polarization_checksum=polarization_crc,
                                                                      safe=safe,
                                                                      normalization_factor=normalization_factor)
                                I.shape = npt
                                I = I.T
                                bins_rad = integr.bin_centers0  # this will be copied later
                                bins_azim = integr.bin_centers1
                    else:
                        I, bins_rad, bins_azim, sum_, count = integr.integrate(data, dark=dark, flat=flat,
                                                                               solidAngle=solidangle,
                                                                               dummy=dummy,
                                                                               delta_dummy=delta_dummy,
                                                                               polarization=polarization,
                                                                               normalization_factor=normalization_factor)

        if method.method[1:4] in (("pseudo", "histogram", "cython"), ("full", "histogram", "cython")):
            logger.debug("integrate2d uses SplitPixel implementation")
            pos = self.array_from_unit(shape, "corner", unit, scale=False)
            I, bins_rad, bins_azim, sum_, count = splitPixel.fullSplit2D(pos=pos,
                                                                         weights=data,
                                                                         bins=(npt_rad, npt_azim),
                                                                         pos0_range=radial_range,
                                                                         pos1_range=azimuth_range,
                                                                         dummy=dummy,
                                                                         delta_dummy=delta_dummy,
                                                                         mask=mask,
                                                                         dark=dark,
                                                                         flat=flat,
                                                                         solidangle=solidangle,
                                                                         polarization=polarization,
                                                                         normalization_factor=normalization_factor,
                                                                         chiDiscAtPi=self.chiDiscAtPi,
                                                                         empty=dummy if dummy is not None else self._empty)
        if method.method[1:4] == ("bbox", "histogram", "cython"):
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
                                                                        pos0_range=radial_range,
                                                                        pos1_range=azimuth_range,
                                                                        dummy=dummy,
                                                                        delta_dummy=delta_dummy,
                                                                        mask=mask,
                                                                        dark=dark,
                                                                        flat=flat,
                                                                        solidangle=solidangle,
                                                                        polarization=polarization,
                                                                        normalization_factor=normalization_factor,
                                                                        chiDiscAtPi=self.chiDiscAtPi,
                                                                        empty=dummy if dummy is not None else self._empty)

        if method.method[1:3] == ("no", "histogram") and method.impl_lower != "opencl":
            logger.debug("integrate2d uses numpy or cython implementation")
            data = data.astype(numpy.float32)  # it is important to make a copy see issue #88
            mask = self.create_mask(data, mask, dummy, delta_dummy,
                                    unit=unit,
                                    radial_range=radial_range,
                                    azimuth_range=azimuth_range,
                                    mode="where")
            pos0 = self.array_from_unit(shape, "center", unit, scale=False)
            pos1 = self.chiArray(shape)

            if radial_range is None:
                radial_range = [pos0.min(), pos0.max() * EPS32]

            if azimuth_range is None:
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
            if method.impl_lower == "cython":
                I, bins_azim, bins_rad, sum_, count = histogram.histogram2d(pos0=pos1,
                                                                            pos1=pos0,
                                                                            weights=data,
                                                                            bins=(npt_azim, npt_rad),
                                                                            split=False,
                                                                            empty=dummy if dummy is not None else self._empty,
                                                                            normalization_factor=normalization_factor)
            elif method.impl_lower == "python":
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
        result._set_compute_engine(str(method))
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

        return result

    _integrate2d_legacy = integrate2d_legacy

    def integrate2d_ng(self, data, npt_rad, npt_azim=360,
                        filename=None, correctSolidAngle=True, variance=None,
                        error_model=None, radial_range=None, azimuth_range=None,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method="bbox", unit=units.Q, safe=True,
                        normalization_factor=1.0, metadata=None):
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
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: azimuthaly regrouped intensity, q/2theta/r pos. and chi pos.
        :rtype: Integrate2dResult, dict
        """
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        assert method.dimension == 2
        npt = (npt_rad, npt_azim)
        unit = units.to_unit(unit)
        pos0_scale = unit.scale
        empty = dummy if dummy is not None else self._empty
        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "provided"
            mask = numpy.ascontiguousarray(mask)
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
            self.check_chi_disc(azimuth_range)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(shape, polarization_factor, with_checksum=True)

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
        signal2d = None
        norm2d = None
        var2d = None

        if method.algo_lower in ("csr", "lut"):
            intpl = None
            cython_method = IntegrationMethod.select_method(method.dimension, method.split_lower, method.algo_lower, "cython")[0]
            if cython_method not in self.engines:
                cython_engine = self.engines[cython_method] = Engine()
            else:
                cython_engine = self.engines[cython_method]
            with cython_engine.lock:
                cython_integr = cython_engine.engine
                cython_reset = None

                if cython_integr is None:
                    cython_reset = "of first initialization"
                if (not cython_reset) and safe:
                    if cython_integr.unit != unit:
                        cython_reset = "unit was changed"
                    if cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    if cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    if cython_integr.empty != empty:
                        cython_reset = "empty value changed"
                    if (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = f"mask but {method.algo_lower.upper()} was without mask"
                    elif (mask is None) and (cython_integr.cmask is not None):
                        cython_reset = f"no mask but { method.algo_lower.upper()} has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    if (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = f"radial_range was defined in { method.algo_lower.upper()}"
                    elif (radial_range is not None) and (cython_integr.pos0_range != radial_range):
                        cython_reset = f"radial_range is defined but differs in %s" % method.algo_lower.upper()
                    if (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = f"azimuth_range not defined and {method.algo_lower.upper()} had azimuth_range defined"
                    elif (azimuth_range is not None) and (cython_integr.pos1_range != azimuth_range):
                        cython_reset = f"azimuth_range requested and {method.algo_lower.upper()}'s azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.integrate2d_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        if method.algo_lower == "csr":
                            cython_integr = self.setup_CSR(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                        else:
                            cython_integr = self.setup_LUT(shape, npt, mask,
                                                           radial_range, azimuth_range,
                                                           mask_checksum=mask_crc,
                                                           unit=unit, split=split,
                                                           empty=empty, scale=False)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        cython_integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        cython_engine.set_engine(cython_integr)
            # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.
            if method.impl_lower != "cython":
                # method.impl_lower in ("opencl", "python"):
                if method not in self.engines:
                    # instanciated the engine
                    engine = self.engines[method] = Engine()
                else:
                    engine = self.engines[method]
                with engine.lock:
                    # Validate that the engine used is the proper one
                    integr = engine.engine
                    reset = None
                    if integr is None:
                        reset = "init"
                    if (not reset) and safe:
                        if integr.unit != unit:
                            reset = "unit changed"
                        if integr.bins != numpy.prod(npt):
                            reset = "number of points changed"
                        if integr.size != data.size:
                            reset = "input image size changed"
                        if integr.empty != empty:
                            reset = "empty value changed"
                        if (mask is not None) and (not integr.check_mask):
                            reset = "mask but CSR was without mask"
                        elif (mask is None) and (integr.check_mask):
                            reset = "no mask but CSR has mask"
                        elif (mask is not None) and (integr.mask_checksum != mask_crc):
                            reset = "mask changed"
                        if (radial_range is None) and (integr.pos0_range is not None):
                            reset = "radial_range was defined in CSR"
                        elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                            reset = "radial_range is defined but differs in CSR"
                        if (azimuth_range is None) and (integr.pos1_range is not None):
                            reset = "azimuth_range not defined and CSR had azimuth_range defined"
                        elif (azimuth_range is not None) and integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                            reset = "azimuth_range requested and CSR's azimuth_range don't match"
                    error = False
                    if reset:
                        logger.info("AI.integrate2d: Resetting integrator because %s", reset)
                        split = method.split_lower
                        try:
                            if method.algo_lower == "csr":
                                cython_integr = self.setup_CSR(shape, npt, mask,
                                                               radial_range, azimuth_range,
                                                               mask_checksum=mask_crc,
                                                               unit=unit, split=split,
                                                               empty=empty, scale=False)
                            else:
                                cython_integr = self.setup_LUT(shape, npt, mask,
                                                               radial_range, azimuth_range,
                                                               mask_checksum=mask_crc,
                                                               unit=unit, split=split,
                                                               empty=empty, scale=False)
                        except MemoryError:
                            logger.warning("MemoryError: falling back on default implementation")
                            cython_integr = None
                            self.reset_engines()
                            method = self.DEFAULT_METHOD_2D
                            error = True
                        else:
                            error = False
                            cython_engine.set_engine(cython_integr)
                if not error:
                    if method in self.engines:
                        ocl_py_engine = self.engines[method]
                    else:
                        ocl_py_engine = self.engines[method] = Engine()
                    integr = ocl_py_engine.engine
                    if integr is None or integr.checksum != cython_integr.lut_checksum:
                        if (method.impl_lower == "opencl"):
                            with ocl_py_engine.lock:
                                # print(method)
                                integr = method.class_funct_ng.klass(cython_integr.lut,
                                                                     cython_integr.size,
                                                                     bin_centers=cython_integr.bin_centers0,
                                                                     azim_centers=cython_integr.bin_centers1,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1],
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc
                                                                     )

                        elif (method.impl_lower == "python"):
                            with ocl_py_engine.lock:
                                integr = method.class_funct_ng.klass(cython_integr.lut,
                                                                     cython_integr.size,
                                                                     bin_centers=cython_integr.bin_centers0,
                                                                     azim_centers=cython_integr.bin_centers1,
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc)
                        ocl_py_engine.set_engine(integr)

                    if (integr is not None):
                            intpl = integr.integrate_ng(data,
                                                       variance=variance,
                                                       dark=dark, flat=flat,
                                                       solidangle=solidangle,
                                                       solidangle_checksum=self._dssa_crc,
                                                       dummy=dummy,
                                                       delta_dummy=delta_dummy,
                                                       polarization=polarization,
                                                       polarization_checksum=polarization_crc,
                                                       safe=safe,
                                                       normalization_factor=normalization_factor)
            if intpl is None:  # fallback if OpenCL failed or default cython
                # The integrator has already been initialized previously
                intpl = cython_integr.integrate_ng(data,
                                                   variance=variance,
                                                   # poissonian=poissonian,
                                                   dummy=dummy,
                                                   delta_dummy=delta_dummy,
                                                   dark=dark,
                                                   flat=flat,
                                                   solidangle=solidangle,
                                                   polarization=polarization,
                                                   normalization_factor=normalization_factor)
            I = intpl.intensity
            bins_rad = intpl.radial
            bins_azim = intpl.azimuthal
            signal2d = intpl.signal
            norm2d = intpl.normalization
            count = intpl.count
            if variance is not None:
                sigma = intpl.sigma
                var2d = intpl.variance

        elif method.algo_lower == "histogram":
            if method.split_lower in ("pseudo", "full"):
                logger.debug("integrate2d uses (full, histogram, cython) implementation")
                pos = self.array_from_unit(shape, "corner", unit, scale=False)
                integrator = method.class_funct_ng.function
                intpl = integrator(pos=pos,
                                 weights=data,
                                 bins=(npt_rad, npt_azim),
                                 pos0_range=radial_range,
                                 pos1_range=azimuth_range,
                                 dummy=dummy,
                                 delta_dummy=delta_dummy,
                                 mask=mask,
                                 dark=dark,
                                 flat=flat,
                                 solidangle=solidangle,
                                 polarization=polarization,
                                 normalization_factor=normalization_factor,
                                 chiDiscAtPi=self.chiDiscAtPi,
                                 empty=empty,
                                 variance=variance)

            elif method.split_lower == "bbox":
                logger.debug("integrate2d uses BBox implementation")
                chi = self.chiArray(shape)
                dchi = self.deltaChi(shape)
                pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                dpos0 = self.array_from_unit(shape, "delta", unit, scale=False)
                intpl = splitBBox.histoBBox2d_ng(weights=data,
                                               pos0=pos0,
                                               delta_pos0=dpos0,
                                               pos1=chi,
                                               delta_pos1=dchi,
                                               bins=(npt_rad, npt_azim),
                                               pos0_range=radial_range,
                                               pos1_range=azimuth_range,
                                               dummy=dummy,
                                               delta_dummy=delta_dummy,
                                               mask=mask,
                                               dark=dark,
                                               flat=flat,
                                               solidangle=solidangle,
                                               polarization=polarization,
                                               normalization_factor=normalization_factor,
                                               chiDiscAtPi=self.chiDiscAtPi,
                                               empty=empty,
                                               variance=variance)
            elif method.split_lower == "no":
                if method.impl_lower == "opencl":
                    logger.debug("integrate2d uses OpenCL histogram implementation")
                    if method not in self.engines:
                    # instanciated the engine
                        engine = self.engines[method] = Engine()
                    else:
                        engine = self.engines[method]
                    with engine.lock:
                        # Validate that the engine used is the proper one #TODO!!!!
                        integr = engine.engine
                        reset = None
                        if integr is None:
                            reset = "init"
                        if (not reset) and safe:
                            if integr.unit != unit:
                                reset = "unit changed"
                            if (integr.bins_radial, integr.bins_azimuthal) != npt:
                                reset = "number of points changed"
                            if integr.size != data.size:
                                reset = "input image size changed"
                            if (mask is not None) and (not integr.check_mask):
                                reset = "mask but CSR was without mask"
                            elif (mask is None) and (integr.check_mask):
                                reset = "no mask but CSR has mask"
                            elif (mask is not None) and (integr.on_device.get("mask") != mask_crc):
                                reset = "mask changed"
                            if self._cached_array[unit.name.split("_")[0] + "_crc"] != integr.on_device.get("radial"):
                                reset = "radial array changed"
                            if self._cached_array["chi_crc"] != integr.on_device.get("azimuthal"):
                                reset = "azimuthal array changed"
                            # Nota: Ranges are enforced at runtime, not initialization
                        error = False
                        if reset:
                            logger.info("AI.integrate2d: Resetting OCL_Histogram2d integrator because %s", reset)
                            rad = self.array_from_unit(shape, typ="center", unit=unit, scale=False)
                            rad_crc = self._cached_array[unit.name.split("_")[0] + "_crc"] = crc32(rad)
                            azi = self.chiArray(shape)
                            azi_crc = self._cached_array["chi_crc"] = crc32(azi)
                            try:
                                integr = method.class_funct_ng.klass(rad,
                                                                     azi,
                                                                     *npt,
                                                                     radial_checksum=rad_crc,
                                                                     azimuthal_checksum=azi_crc,
                                                                     empty=empty, unit=unit,
                                                                     mask=mask, mask_checksum=mask_crc,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1]
                                                                     )
                            except MemoryError:
                                logger.warning("MemoryError: falling back on default forward implementation")
                                integr = None
                                self.reset_engines()
                                method = self.DEFAULT_METHOD_2D
                                error = True
                            else:
                                error = False
                                engine.set_engine(integr)
                    if not error:
                        intpl = integr.integrate(data, dark=dark, flat=flat,
                                                 solidangle=solidangle,
                                                 solidangle_checksum=self._dssa_crc,
                                                 dummy=dummy,
                                                 delta_dummy=delta_dummy,
                                                 polarization=polarization,
                                                 polarization_checksum=polarization_crc,
                                                 safe=safe,
                                                 normalization_factor=normalization_factor,
                                                 radial_range=radial_range,
                                                 azimuthal_range=azimuth_range)
###################3
                elif method.impl_lower == "cython":
                    logger.debug("integrate2d uses Cython histogram implementation")
                    prep = preproc(data,
                                   dark=dark,
                                   flat=flat,
                                   solidangle=solidangle,
                                   polarization=polarization,
                                   absorption=None,
                                   mask=mask,
                                   dummy=dummy,
                                   delta_dummy=delta_dummy,
                                   normalization_factor=normalization_factor,
                                   empty=self._empty,
                                   split_result=4,
                                   variance=variance,
                                   # dark_variance=None,
                                   # poissonian=False,
                                   dtype=numpy.float32)
                    pos0 = self.array_from_unit(shape, "center", unit, scale=False)
                    chi = self.chiArray(shape)
                    intpl = histogram.histogram2d_engine(pos0=pos0,
                                                       pos1=chi,
                                                       weights=prep,
                                                       bins=(npt_rad, npt_azim),
                                                       pos0_range=radial_range,
                                                       pos1_range=azimuth_range,
                                                       split=False,
                                                       empty=empty,
                                                       )

                else:  # Python implementation:
                    logger.debug("integrate2d uses python implementation")
                    data = data.astype(numpy.float32)  # it is important to make a copy see issue #88
                    mask = self.create_mask(data, mask, dummy, delta_dummy,
                                            unit=unit,
                                            radial_range=radial_range,
                                            azimuth_range=azimuth_range,
                                            mode="normal").ravel()
                    pos0 = self.array_from_unit(shape, "center", unit, scale=False).ravel()
                    pos1 = self.chiArray(shape).ravel()

                    if radial_range is None:
                        radial_range = [pos0.min(), pos0.max()]
                    if azimuth_range is None:
                        azimuth_range = [pos1.min(), pos1.max()]

                    if method.method[1:4] == ("no", "histogram", "python"):
                        logger.debug("integrate2d uses Numpy implementation")
                        intpl = histogram_engine.histogram2d_engine(radial=pos0,
                                                                    azimuthal=pos1,
                                                                    npt=(npt_rad, npt_azim),
                                                                    raw=data,
                                                                    dark=dark,
                                                                    flat=flat,
                                                                    solidangle=solidangle,
                                                                    polarization=polarization,
                                                                    absorption=None,
                                                                    mask=mask,
                                                                    dummy=dummy,
                                                                    delta_dummy=delta_dummy,
                                                                    normalization_factor=normalization_factor,
                                                                    empty=self._empty,
                                                                    split_result=False,
                                                                    variance=variance,
                                                                    dark_variance=None,
                                                                    poissonian=False,
                                                                    radial_range=radial_range,
                                                                    azimuth_range=azimuth_range)
            I = intpl.intensity
            bins_azim = intpl.azimuthal
            bins_rad = intpl.radial
            signal2d = intpl.signal
            norm2d = intpl.normalization
            count = intpl.count
            if variance is not None:
                sigma = intpl.sigma
                var2d = intpl.variance

        # Duplicate arrays on purpose ....
        bins_rad = bins_rad * pos0_scale
        bins_azim = bins_azim * (180.0 / pi)

        result = Integrate2dResult(I, bins_rad, bins_azim, sigma)
        result._set_method_called("integrate2d")
        result._set_compute_engine(str(method))
        result._set_method(method)
        result._set_unit(unit)
        result._set_count(count)
        result._set_sum(sum_)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_metadata(metadata)

        result._set_sum_signal(signal2d)
        result._set_sum_normalization(norm2d)
        result._set_sum_variance(var2d)

        if filename is not None:
            writer = DefaultAiWriter(filename, self)
            writer.write(result)

        return result

    integrate2d = _integrate2d_ng = integrate2d_ng

    @deprecated(since_version="0.14", reason="Use the class DefaultAiWriter")
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
        self.__save1D(filename=filename,
                      dim1=dim1,
                      I=I,
                      error=error,
                      dim1_unit=dim1_unit,
                      has_dark=has_dark,
                      has_flat=has_flat,
                      polarization_factor=polarization_factor,
                      normalization_factor=normalization_factor)

    def __save1D(self, filename, dim1, I, error=None, dim1_unit=units.TTH,
                 has_dark=False, has_flat=False, polarization_factor=None, normalization_factor=None):
        """This method save the result of a 1D integration.

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

    @deprecated(since_version="0.14", reason="Use the class DefaultAiWriter")
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
        self.__save2D(filename=filename,
                      I=I,
                      dim1=dim1,
                      dim2=dim2,
                      error=error,
                      dim1_unit=dim1_unit,
                      has_dark=has_dark,
                      has_flat=has_flat,
                      polarization_factor=polarization_factor,
                      normalization_factor=normalization_factor)

    def __save2D(self, filename, I, dim1, dim2, error=None, dim1_unit=units.TTH,
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
                  radial_range=None, azimuth_range=None,
                  polarization_factor=None, dark=None, flat=None,
                  method="splitpixel", unit=units.Q,
                  percentile=50, dummy=None, delta_dummy=None,
                  mask=None, normalization_factor=1.0, metadata=None):
        """Perform the 2D integration and filter along each row using a median
        filter

        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param correctSolidAngle: correct for solid angle of each pixel if True
        :type correctSolidAngle: bool
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional

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
        if dummy is None:
            dummy = numpy.finfo(numpy.float32).min
            delta_dummy = None
        unit = units.to_unit(unit)
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        if (method.impl_lower == "opencl") and npt_azim and (npt_azim > 1):
            old = npt_azim
            npt_azim = 1 << int(round(log(npt_azim, 2)))  # power of two above
            if npt_azim != old:
                logger.warning("Change number of azimuthal bins to nearest power of two: %s->%s",
                               old, npt_azim)
        res2d = self.integrate2d(data, npt_rad, npt_azim, mask=mask,
                                 flat=flat, dark=dark,
                                 radial_range=radial_range,
                                 azimuth_range=azimuth_range,
                                 unit=unit, method=method.method,
                                 dummy=dummy, delta_dummy=delta_dummy,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 normalization_factor=normalization_factor)
        integ2d = res2d.intensity
        if (method.impl_lower == "opencl"):
            if (method.algo_lower == "csr") and \
                    (OCL_CSR_ENGINE in self.engines) and \
                    (self.engines[OCL_CSR_ENGINE].engine is not None):
                ctx = self.engines[OCL_CSR_ENGINE].engine.ctx
            elif (method.algo_lower == "lut") and \
                    (OCL_LUT_ENGINE in self.engines) and \
                    (self.engines[OCL_LUT_ENGINE].engine is not None):
                ctx = self.engines[OCL_LUT_ENGINE].engine.ctx
            else:
                ctx = None

            if numpy.isfortran(integ2d) and integ2d.dtype == numpy.float32:
                rdata = integ2d.T
                horizontal = True
            else:
                rdata = numpy.ascontiguousarray(integ2d, dtype=numpy.float32)
                horizontal = False

            if OCL_SORT_ENGINE not in self.engines:
                with self._lock:
                    if OCL_SORT_ENGINE not in self.engines:
                        self.engines[OCL_SORT_ENGINE] = Engine()
            engine = self.engines[OCL_SORT_ENGINE]
            with engine.lock:
                sorter = engine.engine
                if (sorter is None) or \
                   (sorter.npt_width != rdata.shape[1]) or\
                   (sorter.npt_height != rdata.shape[0]):
                    logger.info("reset opencl sorter")
                    sorter = ocl_sort.Separator(npt_height=rdata.shape[0], npt_width=rdata.shape[1], ctx=ctx)
                    engine.set_engine(sorter)
            if "__len__" in dir(percentile):
                if horizontal:
                    spectrum = sorter.trimmed_mean_horizontal(rdata, dummy, [(i / 100.0) for i in percentile]).get()
                else:
                    spectrum = sorter.trimmed_mean_vertical(rdata, dummy, [(i / 100.0) for i in percentile]).get()
            else:
                if horizontal:
                    spectrum = sorter.filter_horizontal(rdata, dummy, percentile / 100.0).get()
                else:
                    spectrum = sorter.filter_vertical(rdata, dummy, percentile / 100.0).get()
        else:
            dummies = (integ2d == dummy).sum(axis=0)
            # add a line of zeros at the end (along npt_azim) so that the value for no valid pixel is 0
            sorted_ = numpy.zeros((npt_azim + 1, npt_rad))
            sorted_[:npt_azim,:] = numpy.sort(integ2d, axis=0)

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
        result._set_compute_engine(str(method))
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

    def _sigma_clip_legacy(self, data, npt_rad=1024, npt_azim=512,
                           correctSolidAngle=True, polarization_factor=None,
                           radial_range=None, azimuth_range=None,
                           dark=None, flat=None,
                           method="splitpixel", unit=units.Q,
                           thres=3, max_iter=5, dummy=None, delta_dummy=None,
                           mask=None, normalization_factor=1.0, metadata=None):
        """Perform the 2D integration and perform a sigm-clipping iterative
        filter along each row. see the doc of scipy.stats.sigmaclip for the
        options.

        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param npt_azim: number of azimuthal points
        :param bool correctSolidAngle: correct for solid angle of each pixel
                if True
        :param float polarization_factor: polarization factor between -1 (vertical)
                and +1 (horizontal).

                - 0 for circular polarization or random,
                - None for no correction,
                - True for using the former correction
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional
        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param unit: unit to be used for integration
        :param method: pathway for integration and sort
        :param thres: cut-off for n*sigma: discard any values with (I-<I>)/sigma > thres.
                The threshold can be a 2-tuple with sigma_low and sigma_high.
        :param max_iter: maximum number of iterations        :param mask: masked out pixels array
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: any other metadata,
        :type metadata: JSON serializable dict
        :return: Integrate1D like result like
        """
        # We use NaN as dummies
        if dummy is None:
            dummy = numpy.NaN
            delta_dummy = None
        unit = units.to_unit(unit)
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        if "__len__" in dir(thres) and len(thres) > 0:
            sigma_lo = thres[0]
            sigma_hi = thres[-1]
        else:
            sigma_lo = sigma_hi = thres

        if (method.impl_lower == "opencl") and npt_azim and (npt_azim > 1):
            old = npt_azim
            npt_azim = 1 << int(round(log(npt_azim, 2)))  # power of two above
            if npt_azim != old:
                logger.warning("Change number of azimuthal bins to nearest power of two: %s->%s",
                               old, npt_azim)

        res2d = self.integrate2d(data, npt_rad, npt_azim, mask=mask,
                                 azimuth_range=azimuth_range,
                                 radial_range=radial_range,
                                 flat=flat, dark=dark,
                                 unit=unit, method=method,
                                 dummy=dummy, delta_dummy=delta_dummy,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 normalization_factor=normalization_factor)
        image = res2d.intensity
        if (method.impl_lower == "opencl"):
            if (method.algo_lower == "csr") and \
                    (OCL_CSR_ENGINE in self.engines) and \
                    (self.engines[OCL_CSR_ENGINE].engine is not None):
                ctx = self.engines[OCL_CSR_ENGINE].engine.ctx
            elif (method.algo_lower == "csr") and \
                    (OCL_LUT_ENGINE in self.engines) and \
                    (self.engines[OCL_LUT_ENGINE].engine is not None):
                ctx = self.engines[OCL_LUT_ENGINE].engine.ctx
            else:
                ctx = None

            if numpy.isfortran(image) and image.dtype == numpy.float32:
                rdata = image.T
                horizontal = True
            else:
                rdata = numpy.ascontiguousarray(image, dtype=numpy.float32)
                horizontal = False

            if OCL_SORT_ENGINE not in self.engines:
                with self._lock:
                    if OCL_SORT_ENGINE not in self.engines:
                        self.engines[OCL_SORT_ENGINE] = Engine()
            engine = self.engines[OCL_SORT_ENGINE]
            with engine.lock:
                sorter = engine.engine
                if (sorter is None) or \
                   (sorter.npt_width != rdata.shape[1]) or\
                   (sorter.npt_height != rdata.shape[0]):
                    logger.info("reset opencl sorter")
                    sorter = ocl_sort.Separator(npt_height=rdata.shape[0], npt_width=rdata.shape[1], ctx=ctx)
                    engine.set_engine(sorter)

            if horizontal:
                res = sorter.sigma_clip_horizontal(rdata, dummy=dummy,
                                                   sigma_lo=sigma_lo,
                                                   sigma_hi=sigma_hi,
                                                   max_iter=max_iter)
            else:
                res = sorter.sigma_clip_vertical(rdata, dummy=dummy,
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
            for _ in range(max_iter):
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
        result._set_compute_engine(str(method))
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

    def sigma_clip_ng(self, data,
                      npt=1024,
                      correctSolidAngle=True,
                      polarization_factor=None,
                      variance=None,
                      error_model=None,
                      radial_range=None,
                      azimuth_range=None,
                      dark=None,
                      flat=None,
                      method=("no", "csr", "cython"),
                      unit=units.Q,
                      thres=5.0,
                      max_iter=5,
                      dummy=None,
                      delta_dummy=None,
                      mask=None,
                      normalization_factor=1.0,
                      metadata=None,
                      safe=True,
                       **kwargs):
        """Performs iteratively the 1D integration with variance propagation 
        and performs a sigm-clipping at each iteration, i.e.
        all pixel which intensity differs more than thres*std is 
        discarded for next iteration.
        
        Keep only pixels with intensty:
        
            |I - <I>| < thres * std(I)

        This enforces a gaussian distibution and is very good at extracting 
        background or amorphous isotropic scattering out of Bragg peaks.

        :param data: input image as numpy array
        :param npt_rad: number of radial points
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param float polarization_factor: polarization factor between:
                -1 (vertical) 
                +1 (horizontal).
                - 0 for circular polarization or random,
                - None for no correction,
                - True for using the former correction
        :param radial_range: The lower and upper range of the radial unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type radial_range: (float, float), optional
        :param azimuth_range: The lower and upper range of the azimuthal angle in degree. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored.
        :type azimuth_range: (float, float), optional

        :param ndarray dark: dark noise image
        :param ndarray flat: flat field image
        :param ndarray variance: the variance of the signal 
        :param str error_model: can be "poisson" to assume a poissonian detector (variance=I) or "azimuthal" to take the std² in each ring (better, more expenive)  
        :param unit: unit to be used for integration
        :param method: pathway for integration and sort
        :param thres: cut-off for n*sigma: discard any values with (I-<I>)/sigma > thres.
        :param max_iter: maximum number of iterations        
        :param mask: masked out pixels array
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: any other metadata,
        :type metadata: JSON serializable dict
        :param safe: set to False to skip some tests
        :return: Integrate1D like result like
        
        The difference with the previous version is that there is no 2D regrouping, hence this is faster. 
        The standard deviation is usually smaller than previously and the signal cleaner. It is also slightly faster.
        
        The case neither `error_model`, nor `variance` is provided, fall-back on a poissonian model.
        
        """
        for k in kwargs:
            if k == "npt_azim":
                logger.warning("'npt_azim' argument is not used in sigma_clip_ng as not 2D intergration is performed anymore")
            else:
                logger.warning("Got unknown argument %s %s", k, kwargs[k])
        if (error_model is None) and (variance is None):
            logger.warning("Either `variance` or `error_model` is needed for sigma-clipping, using a Poissonnian model as default !")
            error_model = "poisson"
        unit = units.to_unit(unit)
        if radial_range:
            radial_range = tuple(radial_range[i] / unit.scale for i in (0, -1))
        if azimuth_range is not None:
            azimuth_range = self.normalize_azimuth_range(azimuth_range)

        method = self._normalize_method(method, dim=1, default=self.DEFAULT_METHOD_1D)

        if mask is None:
            has_mask = "from detector"
            mask = self.mask
            mask_crc = self.detector.get_mask_crc()
            if mask is None:
                has_mask = False
                mask_crc = None
        else:
            has_mask = "user provided"
            mask = numpy.ascontiguousarray(mask)
            mask_crc = crc32(mask)

        if correctSolidAngle:
            solidangle = self.solidAngleArray(data.shape, correctSolidAngle)
        else:
            solidangle = None

        if polarization_factor is None:
            polarization = polarization_crc = None
        else:
            polarization, polarization_crc = self.polarization(data.shape, polarization_factor, with_checksum=True)

        if (method.algo_lower == "csr"):
            "This is the only method implemented for now ..."
            # Prepare LUT if needed!
            # initialize the CSR integrator in Cython as it may be needed later on.
            cython_method = IntegrationMethod.select_method(method.dimension, method.split_lower, method.algo_lower, "cython")[0]
            if cython_method not in self.engines:
                cython_engine = self.engines[cython_method] = Engine()
            else:
                cython_engine = self.engines[cython_method]
            with cython_engine.lock:
                # Validate that the engine used is the proper one
                cython_integr = cython_engine.engine
                cython_reset = None
                if cython_integr is None:
                    cython_reset = "of first initialization"
                if (not cython_reset) and safe:
                    if cython_integr.unit != unit:
                        cython_reset = "unit was changed"
                    if cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    if cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    if cython_integr.empty != self._empty:
                        cython_reset = "empty value changed "
                    if (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = "mask but CSR was without mask"
                    elif (mask is None) and (cython_integr.check_mask):
                        cython_reset = "no mask but CSR has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    if (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and cython_integr.pos0_range != (min(radial_range), max(radial_range) * EPS32):
                        cython_reset = "radial_range is defined but not the same as in CSR"
                    if (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and cython_integr.pos1_range != (min(azimuth_range), max(azimuth_range) * EPS32):
                        cython_reset = "azimuth_range requested and CSR's azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.sigma_clip_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        cython_integr = self.setup_CSR(data.shape, npt, mask,
                                                       mask_checksum=mask_crc,
                                                       unit=unit, split=split,
                                                       pos0_range=radial_range,
                                                       pos1_range=azimuth_range,
                                                       empty=self._empty,
                                                       scale=False)
                    except MemoryError:  # CSR method is hungry...
                        logger.warning("MemoryError: falling back on forward implementation")
                        cython_integr = None
                        self.reset_engines()
                        method = self.DEFAULT_METHOD_1D
                    else:
                        cython_engine.set_engine(cython_integr)
            if method not in self.engines:
                # instanciated the engine
                engine = self.engines[method] = Engine()
            else:
                engine = self.engines[method]
            with engine.lock:
                # Validate that the engine used is the proper one
                integr = engine.engine
                reset = None
                # This whole block uses CSR, Now we should treat all the various implementation: Cython, OpenCL and finally Python.

                # Validate that the engine used is the proper one
                if integr is None:
                    reset = "of first initialization"
                if (not reset) and safe:
                    if integr.unit != unit:
                        reset = "unit was changed"
                    if integr.bins != npt:
                        reset = "number of points changed"
                    if integr.size != data.size:
                        reset = "input image size changed"
                    if integr.empty != self._empty:
                        reset = "empty value changed "
                    if (mask is not None) and (not integr.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    # TODO
                    if (radial_range is None) and (integr.pos0_range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                        reset = "radial_range is defined but not the same as in CSR"
                    if (azimuth_range is None) and (integr.pos1_range is not None):
                        reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                        reset = "azimuth_range requested and CSR's azimuth_range don't match"

                if reset:
                    logger.info("ai.sigma_clip_ng: Resetting ocl_csr integrator because %s", reset)
                    csr_integr = self.engines[cython_method].engine
                    if method.impl_lower == "opencl":
                        try:
                            integr = method.class_funct_ng.klass(csr_integr.lut,
                                                                 image_size=data.size,
                                                                 checksum=csr_integr.lut_checksum,
                                                                 empty=self._empty,
                                                                 unit=unit,
                                                                 mask_checksum=csr_integr.mask_checksum,
                                                                 bin_centers=csr_integr.bin_centers,
                                                                 platformid=method.target[0],
                                                                 deviceid=method.target[1])
                        except MemoryError:
                            logger.warning("MemoryError: falling back on default forward implementation")
                            self.reset_engines()
                            method = self.DEFAULT_METHOD_1D
                        else:
                            # Copy some properties from the cython integrator
                            integr.pos0_range = csr_integr.pos0_range
                            integr.pos1_range = csr_integr.pos1_range
                            engine.set_engine(integr)
                    elif method.impl_lower in ("python", "cython"):
                        integr = method.class_funct_ng.klass(lut=csr_integr.lut,
                                                             image_size=data.size,
                                                             empty=self._empty,
                                                             unit=unit,
                                                             mask_checksum=csr_integr.mask_checksum,
                                                             bin_centers=csr_integr.bin_centers)
                        # Copy some properties from the cython integrator
                        integr.pos0_range = csr_integr.pos0_range
                        integr.pos1_range = csr_integr.pos1_range
                        engine.set_engine(integr)
                    else:
                        logger.error(f"Implementation {method.impl_lower} not supported")
                else:
                    integr = self.engines[method].engine
                kwargs = {"dark":dark, "dummy":dummy, "delta_dummy":delta_dummy,
                          "variance":variance, "dark_variance":None,
                          "flat":flat, "solidangle":solidangle, "polarization":polarization, "absorption":None,
                          "error_model":error_model, "normalization_factor":normalization_factor,
                          "cutoff":thres, "cycle":max_iter}

                intpl = integr.sigma_clip(data, **kwargs)
        else:
            raise RuntimeError("Not yet implemented. Sorry")
        result = Integrate1dResult(intpl.position * unit.scale, intpl.intensity, intpl.sigma)
        result._set_method_called("sigma_clip_ng")
        result._set_method(method)
        result._set_compute_engine(str(method))
        result._set_percentile(thres)
        result._set_unit(unit)
        result._set_has_mask_applied(has_mask)
        result._set_metadata(metadata)
        result._set_sum_signal(intpl.signal)
        result._set_sum_normalization(intpl.normalization)
        result._set_sum_variance(intpl.variance)
        result._set_count(intpl.count)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        return result

    sigma_clip = _sigma_clip_legacy

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
        :return: SeparateResult which the bragg & amorphous signal
        
        Note: the filtered 1D spectrum can be retrieved from
        SeparateResult.radial and SeparateResult.intensity 
        """

        filter_result = self.medfilt1d(data, npt_rad=npt_rad, npt_azim=npt_azim,
                                       unit=unit, method=method,
                                       percentile=percentile, mask=mask)
        # This takes 100ms and is the next to be optimized.
        amorphous = self.calcfrom1d(filter_result.radial, filter_result.intensity,
                                    data.shape, mask=None,
                                    dim1_unit=unit,
                                    correctSolidAngle=True)
        bragg = data - amorphous
        if restore_mask:
            wmask = numpy.where(mask)
            maskdata = data[wmask]
            bragg[wmask] = maskdata
            amorphous[wmask] = maskdata

        result = SeparateResult(bragg, amorphous)
        result._radial = filter_result.radial
        result._intensity = filter_result.intensity
        result._sigma = filter_result.sigma

        result._set_sum_signal(filter_result.sum_signal)
        result._set_sum_variance(filter_result.sum_variance)
        result._set_sum_normalization(filter_result.sum_normalization)
        result._set_count(filter_result.count)

        result._set_method_called("medfilt1d")
        result._set_compute_engine(str(method))
        result._set_percentile(percentile)
        result._set_npt_azim(npt_azim)
        result._set_unit(unit)
        result._set_has_mask_applied(filter_result.has_mask_applied)
        result._set_metadata(filter_result.metadata)
        result._set_has_dark_correction(filter_result.has_dark_correction)
        result._set_has_flat_correction(filter_result.has_flat_correction)

        # TODO when switching to sigma-clipped filtering
        # result._set_polarization_factor(polarization_factor)
        # result._set_normalization_factor(normalization_factor)

        return result

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
        :param grow_mask: grow mask in polar coordinated to accomodate pixel
            splitting algoritm
        :return: inpainting object which contains the restored image as .data
        """
        from .ext import inpainting
        dummy = -1
        delta_dummy = 0.9
        method = IntegrationMethod.select_one_available(method, dim=2,
                                                        default=self.DEFAULT_METHOD_2D)

        assert mask.shape == self.detector.shape
        mask = numpy.ascontiguousarray(mask, numpy.int8)
        blank_data = numpy.zeros(mask.shape, dtype=numpy.float32)
        ones_data = numpy.ones(mask.shape, dtype=numpy.float32)

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
                  "azimuth_range": azimuth_range,
                  "radial_range": (0, rmax),
                  "polarization_factor": None,
                  # Nullify the masks to avoid to use the detector once
                  "dark": blank_mask,
                  "mask": blank_mask,
                  "flat": ones_data}

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

    @deprecated(reason="Not maintained", since_version="0.17")
    def set_darkfiles(self, files=None, method="mean"):
        """Set the dark current from one or mutliple files, avaraged
        according to the method provided.

        Moved to Detector.

        :param files: file(s) used to compute the dark.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str
        """
        self.detector.set_darkfiles(files, method)

    @property
    @deprecated(reason="Not maintained", since_version="0.17")
    def darkfiles(self):
        return self.detector.darkfiles

    @deprecated(reason="Not maintained", since_version="0.17")
    def set_flatfiles(self, files, method="mean"):
        """Set the flat field from one or mutliple files, averaged
        according to the method provided.

        Moved to Detector.

        :param files: file(s) used to compute the flat-field.
        :type files: str or list(str) or None
        :param method: method used to compute the dark, "mean" or "median"
        :type method: str
        """
        self.detector.set_flatfiles(files, method)

    @property
    @deprecated(reason="Not maintained", since_version="0.17")
    def flatfiles(self):
        return self.detector.flatfiles

    def get_empty(self):
        return self._empty

    def set_empty(self, value):
        self._empty = float(value)
        # propagate empty values to integrators
        for engine in self.engines.values():
            with engine.lock:
                if engine.engine is not None:
                    try:
                        engine.engine.empty = self._empty
                    except Exception as exeption:
                        logger.error(exeption)

    empty = property(get_empty, set_empty)

    def __getnewargs_ex__(self):
        "Helper function for pickling ai"
        return (self.dist, self.poni1, self.poni2,
                self.rot1, self.rot2, self.rot3,
                self.pixel1, self.pixel2,
                self.splineFile, self.detector, self.wavelength), {}

    def __getstate__(self):
        """Helper function for pickling ai

        :return: the state of the object
        """

        state_blacklist = ('_lock', "engines")
        state = Geometry.__getstate__(self)
        for key in state_blacklist:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        """Helper function for unpickling ai

        :param state: the state of the object
        """
        for statekey, statevalue in state.items():
            setattr(self, statekey, statevalue)
        self._sem = threading.Semaphore()
        self._lock = threading.Semaphore()
        self.engines = {}
