#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "10/10/2024"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import warnings
import threading
import gc
from math import pi, log
import numpy
from ..geometry import Geometry
from .. import units
from ..utils import EPS32, deg2rad, crc32
from ..utils.decorators import deprecated, deprecated_warning
from ..containers import Integrate1dResult, Integrate2dResult, SeparateResult, ErrorModel
from ..io import DefaultAiWriter, save_integrate_result
from ..io.ponifile import PoniFile
error = None
from ..method_registry import IntegrationMethod
from .load_engines import ocl_azim_csr, ocl_azim_lut, ocl_sort, histogram, splitBBox, \
                          splitPixel, splitBBoxCSR, splitBBoxLUT, splitPixelFullCSR, \
                          histogram_engine, splitPixelFullLUT, splitBBoxCSC, splitPixelFullCSC, \
                          PREFERED_METHODS_1D, PREFERED_METHODS_2D
from ..engines import Engine

# Few constants for engine names:
OCL_CSR_ENGINE = "ocl_csr_integr"
OCL_LUT_ENGINE = "ocl_lut_integr"
OCL_HIST_ENGINE = "ocl_histogram"
OCL_SORT_ENGINE = "ocl_sorter"
EXT_LUT_ENGINE = "lut_integrator"
EXT_CSR_ENGINE = "csr_integrator"


class Integrator(Geometry):
    """
    This class is a base class for azimuthal or fiber integrator
    All geometry calculation are done in the parent Geometry class

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
                 splineFile=None, detector=None, wavelength=None, orientation=0):
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
        :param float wavelength: Wave length used in meter
        :param int orientation: orientation of the detector, see pyFAI.detectors.orientation.Orientation
        """
        Geometry.__init__(self, dist, poni1, poni2,
                          rot1, rot2, rot3,
                          pixel1, pixel2, splineFile, detector, wavelength, orientation)

        # mask, maskfile, darkcurrent and flatfield are properties pointing to
        # self.detector now (16/06/2017)

        self._lock = threading.Semaphore()
        self.engines = {}  # key: name of the engine,

        self._empty = 0.0

    def reset(self, collect_garbage=True):
        """Reset azimuthal integrator in addition to other arrays.

        :param collect_garbage: set to False to prevent garbage collection, faster
        """
        Geometry.reset(self, collect_garbage=False)
        self.reset_engines(collect_garbage)

    def reset_engines(self, collect_garbage=True):
        """Urgently free memory by deleting all regrid-engines

        :param collect_garbage: set to False to prevent garbage collection, faster
        """
        with self._lock:
            for key in list(self.engines.keys()):  # explicit copy
                self.engines.pop(key).reset()
        if collect_garbage:
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
            if isinstance(unit, (tuple, list)) and len(unit) == 2:
                radial_unit = units.to_unit(unit[0])
            else:
                radial_unit = units.to_unit(unit)
            rad = self.array_from_unit(shape, "center", radial_unit, scale=False)
            logical_or(mask, rad < radial_range[0], out=mask)
            logical_or(mask, rad > radial_range[1], out=mask)
        if azimuth_range is not None:
            if isinstance(unit, (tuple, list)) and len(unit) == 2:
                azimuth_unit = units.to_unit(unit[1])
            chi = self.array_from_unit(shape, "center", azimuth_unit, scale=False)
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

    def setup_sparse_integrator(self,
                                shape,
                                npt,
                                mask=None,
                                pos0_range=None, pos1_range=None,
                                mask_checksum=None, unit=units.TTH,
                                split="bbox", algo="CSR",
                                empty=None, scale=True):
        """
        Prepare a sparse-matrix integrator based on LUT, CSR or CSC format

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
        :type unit: pyFAI.units.Unit or 2-tuple of them for 2D integration
        :param split: Splitting scheme: valid options are "no", "bbox", "full"
        :param algo: Sparse matrix format to use: "LUT", "CSR" or "CSC"
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
        pattern with the *pos0_range* (radial) and *pos1_range* (azimuthal).

        The *unit* parameter is just propagated to the LUT integrator
        for further checkings: The aim is to prevent an integration to
        be performed in 2th-space when the LUT was setup in q space.
        Unit can also be a 2-tuple in the case of a 2D integration
        """
        if isinstance(unit, (list, tuple)) and len(unit) == 2:
            unit0, unit1 = tuple(units.to_unit(u) for u in unit)
        else:
            unit0 = units.to_unit(unit)
            unit1 = units.CHI_DEG
        if scale and pos0_range:
            pos0_scale = unit0.scale
            pos0_range = tuple(pos0_range[i] / pos0_scale for i in (0, -1))
        if "__len__" in dir(npt) and len(npt) == 2:
            int2d = True
            if scale and pos1_range:
                pos1_scale = unit1.scale
                pos1_range = tuple(pos1_range[i] / pos1_scale for i in (0, -1))
        else:
            int2d = False
        empty = self._empty if empty is None else empty
        if split == "full":
            pos = self.array_from_unit(shape, "corner", unit, scale=False)
        else:
            pos0 = self.array_from_unit(shape, "center", unit0, scale=False)
            if split == "no":
                dpos0 = None
            else:
                dpos0 = self.array_from_unit(shape, "delta", unit0, scale=False)

            pos1 = None
            dpos1 = None
            if int2d or pos1_range:
                pos1 = self.array_from_unit(shape, "center", unit1, scale=False)
                if split == "no":
                    dpos1 = None
                else:
                    dpos1 = self.array_from_unit(shape, "delta", unit1, scale=False)

        if mask is None:
            mask_checksum = None
        else:
            assert mask.shape == shape
        algo = algo.upper()
        if algo == "LUT":
            if split == "full":
                if int2d:
                    return splitPixelFullLUT.HistoLUT2dFullSplit(pos,
                                                    bins=npt,
                                                    pos0_range=pos0_range,
                                                    pos1_range=pos1_range,
                                                    mask=mask,
                                                    mask_checksum=mask_checksum,
                                                    allow_pos0_neg=not unit0.positive,
                                                    unit=unit,
                                                    empty=empty,
                                                    chiDiscAtPi=self.chiDiscAtPi,
                                                    clip_pos1=bool(unit1.period),
                                                    )
                else:
                    return splitPixelFullLUT.HistoLUT1dFullSplit(pos,
                                                                 bins=npt,
                                                                 pos0_range=pos0_range,
                                                                 pos1_range=pos1_range,
                                                                 mask=mask,
                                                                 mask_checksum=mask_checksum,
                                                                 allow_pos0_neg=not unit0.positive,
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
                                                    allow_pos0_neg=not unit0.positive,
                                                    clip_pos1=bool(unit1.period),
                                                    unit=unit,
                                                    empty=empty)
                else:
                    return splitBBoxLUT.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                    bins=npt,
                                                    pos0_range=pos0_range,
                                                    pos1_range=pos1_range,
                                                    mask=mask,
                                                    mask_checksum=mask_checksum,
                                                    allow_pos0_neg=not unit0.positive,
                                                    unit=unit,
                                                    empty=empty)
        elif algo == "CSR":
            if split == "full":
                if int2d:
                    return splitPixelFullCSR.FullSplitCSR_2d(pos,
                                                             bins=npt,
                                                             pos0_range=pos0_range,
                                                             pos1_range=pos1_range,
                                                             mask=mask,
                                                             mask_checksum=mask_checksum,
                                                             allow_pos0_neg=not unit0.positive,
                                                             unit=unit,
                                                             empty=empty,
                                                             chiDiscAtPi=self.chiDiscAtPi,
                                                             clip_pos1=bool(unit1.period),
                                                             )
                else:
                    return splitPixelFullCSR.FullSplitCSR_1d(pos,
                                                             bins=npt,
                                                             pos0_range=pos0_range,
                                                             pos1_range=pos1_range,
                                                             mask=mask,
                                                             mask_checksum=mask_checksum,
                                                             allow_pos0_neg=not unit0.positive,
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
                                                    unit=unit,
                                                    empty=empty,
                                                    allow_pos0_neg=not unit0.positive,
                                                    clip_pos1=bool(unit1.period)
)
                else:
                    return splitBBoxCSR.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                    bins=npt,
                                                    pos0_range=pos0_range,
                                                    pos1_range=pos1_range,
                                                    mask=mask,
                                                    mask_checksum=mask_checksum,
                                                    allow_pos0_neg=not unit0.positive,
                                                    unit=unit,
                                                    empty=empty)
        elif algo == "CSC":
            if split == "full":
                if int2d:
                    return splitPixelFullCSC.FullSplitCSC_2d(pos,
                                                             bins=npt,
                                                             pos0_range=pos0_range,
                                                             pos1_range=pos1_range,
                                                             mask=mask,
                                                             mask_checksum=mask_checksum,
                                                             allow_pos0_neg=not unit0.positive,
                                                             unit=unit,
                                                             empty=empty,
                                                             chiDiscAtPi=self.chiDiscAtPi,
                                                             clip_pos1=bool(unit1.period)
                                                             )
                else:
                    return splitPixelFullCSC.FullSplitCSC_1d(pos,
                                                             bins=npt,
                                                             pos0_range=pos0_range,
                                                             pos1_range=pos1_range,
                                                             mask=mask,
                                                             mask_checksum=mask_checksum,
                                                             allow_pos0_neg=not unit0.positive,
                                                             unit=unit,
                                                             empty=empty)
            else:
                if int2d:
                    return splitBBoxCSC.HistoBBox2d(pos0, dpos0, pos1, dpos1,
                                                    bins=npt,
                                                    pos0_range=pos0_range,
                                                    pos1_range=pos1_range,
                                                    mask=mask,
                                                    mask_checksum=mask_checksum,
                                                    unit=unit,
                                                    empty=empty,
                                                    allow_pos0_neg=not unit0.positive,
                                                    clip_pos1=bool(unit1.period)
)
                else:
                    return splitBBoxCSC.HistoBBox1d(pos0, dpos0, pos1, dpos1,
                                                    bins=npt,
                                                    pos0_range=pos0_range,
                                                    pos1_range=pos1_range,
                                                    mask=mask,
                                                    mask_checksum=mask_checksum,
                                                    allow_pos0_neg=not unit0.positive,
                                                    unit=unit,
                                                    empty=empty)

    @deprecated(since_version="0.22", only_once=True, replacement="setup_sparse_integrator", deprecated_since="0.22.0")
    def setup_LUT(self, shape, npt, mask=None,
                  pos0_range=None, pos1_range=None,
                  mask_checksum=None, unit=units.TTH,
                  split="bbox", empty=None, scale=True):
        """See documentation of setup_sparse_integrator where algo=LUT"""
        return self.setup_sparse_integrator(shape, npt, mask,
                                            pos0_range, pos1_range,
                                            mask_checksum, unit,
                                            split=split, algo="LUT",
                                            empty=empty, scale=scale)

    @deprecated(since_version="0.22", only_once=True, replacement="setup_sparse_integrator", deprecated_since="0.22.0")
    def setup_CSR(self, shape, npt, mask=None,
                  pos0_range=None, pos1_range=None,
                  mask_checksum=None, unit=units.TTH,
                  split="bbox", empty=None, scale=True):
        """See documentation of setup_sparse_integrator where algo=CSR"""
        return self.setup_sparse_integrator(shape, npt, mask,
                                            pos0_range, pos1_range,
                                            mask_checksum, unit,
                                            split=split, algo="CSR",
                                            empty=empty, scale=scale)

    @deprecated(since_version="0.20", only_once=True, replacement="integrate1d_ng", deprecated_since="0.20.0")
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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
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
                    reset = "of first initialization"
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
                        integr = self.setup_sparse_integrator(shape, npt, mask=mask,
                                                pos0_range=radial_range, pos1_range=azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split, algo="LUT",
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
                    reset = "of first initialization"
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
                        integr = self.setup_sparse_integrator(shape, npt, mask=mask,
                                                pos0_range=radial_range, pos1_range=azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split, algo="CSR",
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
                                var1d, a, b = ocl_integr.integrate_legacy(variance,
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
            save_integrate_result(filename, result)

        return result

    _integrate1d_legacy = integrate1d_legacy

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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
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
            azimuth_range = tuple(deg2rad(azimuth_range[i], self.chiDiscAtPi) for i in (0, -1))
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
                    reset = "of first initialization"
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
                        integr = self.setup_sparse_integrator(shape, npt, mask=mask,
                                                pos0_range=radial_range, pos1_range=azimuth_range,
                                                mask_checksum=mask_crc, algo="LUT", unit=unit, scale=False)
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
                                I, sum_, count = ocl_integr.integrate_legacy(data, dark=dark, flat=flat,
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
                        I, bins_rad, bins_azim, sum_, count = integr.integrate_legacy(data, dark=dark, flat=flat,
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
                    reset = "of first initialization"
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
                        integr = self.setup_sparse_integrator(shape, npt, mask=mask,
                                                pos0_range=radial_range, pos1_range=azimuth_range,
                                                mask_checksum=mask_crc,
                                                unit=unit, split=split, algo="CSR",
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
                                I, sum_, count = ocl_integr.integrate_legacy(data, dark=dark, flat=flat,
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
                        I, bins_rad, bins_azim, sum_, count = integr.integrate_legacy(data, dark=dark, flat=flat,
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
            save_integrate_result(filename, result)

        return result

    _integrate2d_legacy = integrate2d_legacy

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
