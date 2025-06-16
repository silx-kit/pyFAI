#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2025 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "16/06/2025"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import warnings
from math import pi, log
import numpy
from .common import Integrator
# from ..geometry import Geometry
from .. import units
from ..utils import EPS32, deg2rad, crc32, rad2rad
from ..utils.mathutil import nan_equal
from ..containers import Integrate1dResult, Integrate2dResult, SeparateResult, ErrorModel
from ..io import DefaultAiWriter, save_integrate_result
from ..io.ponifile import PoniFile
error = None
from ..method_registry import IntegrationMethod
from ..utils.decorators import deprecated
#
from .load_engines import ocl_sort
#ocl_azim_csr, ocl_azim_lut, , histogram, splitBBox, \
#                           splitPixel, splitBBoxCSR, splitBBoxLUT, splitPixelFullCSR, \
#                           histogram_engine, splitPixelFullLUT, splitBBoxCSC, splitPixelFullCSC, \
#                           PREFERED_METHODS_1D, PREFERED_METHODS_2D
#
from ..engines import Engine

# Few constants for engine names:
OCL_CSR_ENGINE = "ocl_csr_integr"
OCL_LUT_ENGINE = "ocl_lut_integr"
OCL_HIST_ENGINE = "ocl_histogram"
OCL_SORT_ENGINE = "ocl_sorter"
EXT_LUT_ENGINE = "lut_integrator"
EXT_CSR_ENGINE = "csr_integrator"


class AzimuthalIntegrator(Integrator):
    """
    This class is an azimuthal integrator based on P. Boesecke's
    geometry and histogram algorithm by Manolo S. del Rio and V.A Sole

    All geometry calculation are done in the Geometry class

    main methods are:

        >>> tth, I = ai.integrate1d(data, npt, unit="2th_deg")
        >>> q, I, sigma = ai.integrate1d(data, npt, unit="q_nm^-1", error_model="poisson")
        >>> regrouped = ai.integrate2d(data, npt_rad, npt_azim, unit="q_nm^-1")[0]
    """


    def integrate1d(self, data, npt, filename=None,
                    correctSolidAngle=True,
                    variance=None, error_model=None,
                    radial_range=None, azimuth_range=None,
                    mask=None, dummy=None, delta_dummy=None,
                    polarization_factor=None, dark=None, flat=None, absorption=None,
                    method=("bbox", "csr", "cython"), unit=units.Q, safe=True,
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
        :param ndarray absorption: absorption correction image
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param Unit unit: Output units, can be "q_nm^-1" (default), "2th_deg", "r_mm" for now.
        :param bool safe: Perform some extra checks to ensure LUT/CSR is still valid. False is faster.
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :param ndarray absorption: detector absorption
        :return: Integrate1dResult namedtuple with (q,I,sigma) +extra informations in it.
        """
        method = self._normalize_method(method, dim=1, default=self.DEFAULT_METHOD_1D)
        if method.dimension != 1:
            raise RuntimeError("integration method is not 1D")
        unit = units.to_unit(unit)
        if dummy is None:
            dummy, delta_dummy = self.detector.get_dummies(data)
        else:
            dummy = numpy.float32(dummy)
            delta_dummy = None if delta_dummy is None else numpy.float32(delta_dummy)
        empty = self._empty

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

        error_model = ErrorModel.parse(error_model)
        if variance is not None:
            if variance.size != data.size:
                raise RuntimeError("Variance array shape does not match data shape")
            error_model = ErrorModel.VARIANCE
        if error_model.poissonian and not method.manage_variance:
            error_model = ErrorModel.VARIANCE
            if dark is None:
                variance = numpy.maximum(data, 1.0).astype(numpy.float32)
            else:
                variance = (numpy.maximum(data, 1.0) + numpy.maximum(dark, 0.0)).astype(numpy.float32)

        # Prepare LUT if needed!
        if method.algo_is_sparse:
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
                    elif cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    elif cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    elif not nan_equal(cython_integr.empty, empty):
                        cython_reset = f"empty value changed {cython_integr.empty}!={empty}"
                    elif (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = f"mask but {method.algo_lower.upper()} was without mask"
                    elif (mask is None) and (cython_integr.cmask is not None):
                        cython_reset = f"no mask but { method.algo_lower.upper()} has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    elif (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = f"radial_range was defined in { method.algo_lower.upper()}"
                    elif (radial_range is not None) and (cython_integr.pos0_range != radial_range):
                        cython_reset = f"radial_range is defined but differs in %s" % method.algo_lower.upper()
                    elif (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = f"azimuth_range not defined and {method.algo_lower.upper()} had azimuth_range defined"
                    elif (azimuth_range is not None) and (cython_integr.pos1_range != azimuth_range):
                        cython_reset = f"azimuth_range requested and {method.algo_lower.upper()}'s azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.integrate1d_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        cython_integr = self.setup_sparse_integrator(shape, npt, mask,
                                                                     radial_range, azimuth_range,
                                                                     mask_checksum=mask_crc,
                                                                     unit=unit, split=split, algo=method.algo_lower,
                                                                     empty=empty, scale=False)
                    except MemoryError:  # sparse methods are hungry...
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
                                            error_model=error_model,
                                            dummy=dummy,
                                            delta_dummy=delta_dummy,
                                            dark=dark,
                                            flat=flat,
                                            solidangle=solidangle,
                                            polarization=polarization,
                                            absorption=absorption,
                                            normalization_factor=normalization_factor,
                                            weighted_average=method.weighted_average)
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
                        elif integr.bins != npt:
                            reset = "number of points changed"
                        elif integr.size != data.size:
                            reset = "input image size changed"
                        elif not nan_equal(integr.empty, empty):
                            reset = f"empty value changed {integr.empty}!={empty}"
                        elif (mask is not None) and (not integr.check_mask):
                            reset = f"mask but {method.algo_lower.upper()} was without mask"
                        elif (mask is None) and (integr.check_mask):
                            reset = f"no mask but {method.algo_lower.upper()} has mask"
                        elif (mask is not None) and (integr.mask_checksum != mask_crc):
                            reset = "mask changed"
                        elif (radial_range is None) and (integr.pos0_range is not None):
                            reset = f"radial_range was defined in {method.algo_lower.upper()}"
                        elif (radial_range is not None) and (integr.pos0_range != radial_range):
                            reset = f"radial_range is defined but differs in {method.algo_lower.upper()}"
                        elif (azimuth_range is None) and (integr.pos1_range is not None):
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

                kwargs = {"error_model": error_model,
                          "variance": variance}
                if method.impl_lower == "opencl":
                    kwargs["polarization_checksum"] = polarization_crc
                    kwargs["solidangle_checksum"] = solidangle_crc
                intpl = integr.integrate_ng(data, dark=dark,
                                            dummy=dummy, delta_dummy=delta_dummy,
                                            flat=flat, solidangle=solidangle,
                                            absorption=absorption, polarization=polarization,
                                            normalization_factor=normalization_factor,
                                            weighted_average=method.weighted_average,
                                            ** kwargs)
            # This section is common to all 3 CSR implementations...
            if error_model.do_variance:
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
            result._set_sum_normalization2(intpl.norm_sq)
            result._set_count(intpl.count)
            result._set_sem(intpl.sem)
            result._set_std(intpl.std)

        # END of CSR/CSC/LUT common implementations
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
                           dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                           variance=variance,
                           flat=flat, solidangle=solidangle,
                           polarization=polarization,
                           absorption=absorption,
                           normalization_factor=normalization_factor,
                           weighted_average=method.weighted_average,
                           mask=mask,
                           radial_range=radial_range,
                           error_model=error_model)

            if error_model.do_variance:
                result = Integrate1dResult(intpl.position * unit.scale,
                                           intpl.intensity,
                                           intpl.sigma)
                result._set_sum_variance(intpl.variance)
                result._set_std(intpl.std)
                result._set_sem(intpl.sem)
                result._set_sum_normalization2(intpl.norm_sq)
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
                    elif integr.bins != npt:
                        reset = "number of points changed"
                    elif integr.size != data.size:
                        reset = "input image size changed"
                    elif not nan_equal(integr.empty, empty):
                        reset = f"empty value changed {integr.empty}!={empty}"
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
                               polarization=polarization, absorption=absorption,
                               polarization_checksum=polarization_crc,
                               normalization_factor=normalization_factor,
                               weighted_average=method.weighted_average,
                               radial_range=radial_range,
                               azimuth_range=azimuth_range,
                               error_model=error_model)

            if error_model.do_variance:
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
                               polarization=polarization, absorption=absorption,
                               normalization_factor=normalization_factor,
                               weighted_average=method.weighted_average,
                               mask=mask,
                               pos0_range=radial_range,
                               pos1_range=azimuth_range,
                               error_model=error_model)
            elif method.method[1] == "full":
                pos = self.array_from_unit(shape, "corner", unit, scale=False)
                intpl = integr(weights=data, variance=variance,
                               pos=pos,
                               bins=npt,
                               dummy=dummy, delta_dummy=delta_dummy, empty=empty,
                               dark=dark, flat=flat, solidangle=solidangle,
                               polarization=polarization, absorption=absorption,
                               normalization_factor=normalization_factor,
                               weighted_average=method.weighted_average,
                               mask=mask,
                               pos0_range=radial_range,
                               pos1_range=azimuth_range,
                               error_model=error_model)
            else:
                raise RuntimeError("Should not arrive here")
            if error_model.do_variance:
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
            result._set_sum_normalization2(intpl.norm_sq)
            result._set_count(intpl.count)
            result._set_sem(intpl.sem)
            result._set_std(intpl.std)

        else:
            raise RuntimeError(f"Fallback method ... should no more be used: {method}")
        result._set_method(method)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_method_called("integrate1d_ng")
        result._set_metadata(metadata)
        result._set_error_model(error_model)
        result._set_poni(PoniFile(self))
        result._set_has_solidangle_correction(correctSolidAngle)
        result._set_weighted_average(method.weighted_average)

        if filename is not None:
            save_integrate_result(filename, result)
        return result

    _integrate1d_ng = integrate1d_ng = integrate1d

    def integrate_radial(self, data, npt, npt_rad=100,
                         correctSolidAngle=True,
                         radial_range=None, azimuth_range=None,
                         mask=None, dummy=None, delta_dummy=None,
                         polarization_factor=None, dark=None, flat=None,
                         method=("bbox", "csr", "cython"), unit=units.CHI_DEG, radial_unit=units.Q,
                         normalization_factor=1.0):
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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param pyFAI.units.Unit unit: Output units, can be "chi_deg" or "chi_rad"
        :param pyFAI.units.Unit radial_unit: unit used for radial representation, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for now
        :param float normalization_factor: Value of a normalization monitor
        :return: chi bins center positions and regrouped intensity
        :rtype: Integrate1dResult
        """
        azimuth_unit = units.to_unit(unit, type_=units.AZIMUTHAL_UNITS)
        res = self.integrate2d_ng(data, npt_rad, npt,
                                  correctSolidAngle=correctSolidAngle,
                                  mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                  polarization_factor=polarization_factor,
                                  dark=dark, flat=flat, method=method,
                                  normalization_factor=normalization_factor,
                                  radial_range=radial_range,
                                  azimuth_range=azimuth_range,
                                  unit=radial_unit)

        azim_scale = azimuth_unit.scale / units.CHI_DEG.scale

        sum_signal = res.sum_signal.sum(axis=-1)
        count = res.count.sum(axis=-1)
        sum_normalization = res._sum_normalization.sum(axis=-1)

        mask = numpy.where(count == 0)

        intensity = sum_signal / sum_normalization
        intensity[mask] = self._empty

        if res.sigma is not None:
            sum_variance = res.sum_variance.sum(axis=-1)
            sigma = numpy.sqrt(sum_variance) / sum_normalization
            sigma[mask] = self._empty
        else:
            sum_variance = None
            sigma = None
        result = Integrate1dResult(res.azimuthal * azim_scale, intensity, sigma)
        result._set_method_called("integrate_radial")
        result._set_unit(azimuth_unit)
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


    def integrate2d_ng(self, data, npt_rad, npt_azim=360,
                        filename=None, correctSolidAngle=True, variance=None,
                        error_model=None, radial_range=None, azimuth_range=None,
                        mask=None, dummy=None, delta_dummy=None,
                        polarization_factor=None, dark=None, flat=None,
                        method=("bbox", "csr", "cython"), unit=units.Q,
                        safe=True, normalization_factor=1.0, metadata=None):
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
        :type method: str
        :param pyFAI.units.Unit unit: Output units, can be "q_nm^-1", "q_A^-1", "2th_deg", "2th_rad", "r_mm" for anything defined as pyFAI.units.RADIAL_UNITS
                                      can also be a 2-tuple of (RADIAL_UNITS, AZIMUTHAL_UNITS) (advanced usage)
        :param safe: Do some extra checks to ensure LUT is still valid. False is faster.
        :type safe: bool
        :param normalization_factor: Value of a normalization monitor
        :type normalization_factor: float
        :param metadata: JSON serializable object containing the metadata, usually a dictionary.
        :return: azimuthaly regrouped intensity, q/2theta/r pos. and chi pos.
        :rtype: Integrate2dResult, dict
        """
        method = self._normalize_method(method, dim=2, default=self.DEFAULT_METHOD_2D)
        if  method.dimension != 2:
            raise RuntimeError("Integration method is not 2D")
        npt = (npt_rad, npt_azim)
        if isinstance(unit, (tuple, list)) and len(unit) == 2:
            radial_unit, azimuth_unit = unit
        else:
            radial_unit = unit
            azimuth_unit = units.CHI_DEG
        radial_unit = units.to_unit(radial_unit, units.RADIAL_UNITS)
        azimuth_unit = units.to_unit(azimuth_unit, units.AZIMUTHAL_UNITS)
        unit = (radial_unit, azimuth_unit)
        space = (radial_unit.space, azimuth_unit.space)
        pos0_scale = radial_unit.scale
        pos1_scale = azimuth_unit.scale
        if dummy is None:
            dummy, delta_dummy = self.detector.get_dummies(data)
        else:
            dummy = numpy.float32(dummy)
            delta_dummy = None if delta_dummy is None else numpy.float32(delta_dummy)
        empty = self._empty

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

        error_model = ErrorModel.parse(error_model)
        if variance is not None:
            if variance.size != data.size:
                raise RuntimeError("Variance array shape does not match data shape")
            error_model = ErrorModel.VARIANCE
        if error_model.poissonian and not method.manage_variance:
            error_model = ErrorModel.VARIANCE
            if dark is None:
                variance = numpy.maximum(data, 1.0).astype(numpy.float32)
            else:
                variance = (numpy.maximum(data, 1.0) + numpy.maximum(dark, 0.0)).astype(numpy.float32)

        if azimuth_range is not None and azimuth_unit.period:
            if azimuth_unit.name.split("_")[-1] == "deg":
                azimuth_range = tuple(deg2rad(azimuth_range[i], self.chiDiscAtPi) for i in (0, -1))
            elif azimuth_unit.name.split("_")[-1] == "rad":
                azimuth_range = tuple(rad2rad(azimuth_range[i], self.chiDiscAtPi) for i in (0, -1))
            if azimuth_range[1] <= azimuth_range[0]:
                azimuth_range = (azimuth_range[0], azimuth_range[1] + 2 * pi)
            self.check_chi_disc(azimuth_range)
        elif azimuth_range is not None:
            azimuth_range = tuple([i / pos1_scale for i in azimuth_range])

        if radial_range is not None and radial_unit.period:
            if radial_unit.name.split("_")[-1] == "deg":
                radial_range = tuple(deg2rad(radial_range[i], self.chiDiscAtPi) for i in (0, -1))
            elif azimuth_unit.name.split("_")[-1] == "rad":
                azimuth_range = tuple(rad2rad(azimuth_range[i], self.chiDiscAtPi) for i in (0, -1))
            if radial_range[1] <= radial_range[0]:
                radial_range = (radial_range[0], radial_range[1] + 2 * pi)
            self.check_chi_disc(radial_range)
        elif radial_range is not None:
            radial_range = tuple([i / pos0_scale for i in radial_range])

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

        if method.algo_is_sparse:
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
                    if cython_integr.space != space:
                        cython_reset = f"unit {cython_integr.unit} incompatible with requested {unit}"
                    if cython_integr.bins != npt:
                        cython_reset = f"number of points {cython_integr.bins} incompatible with requested {npt}"
                    if cython_integr.size != data.size:
                        cython_reset = f"input image size {cython_integr.size} incompatible with requested {data.size}"
                    if not nan_equal(cython_integr.empty, empty):
                        cython_reset = f"empty value changed {cython_integr.empty}!={empty}"
                    if (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = f"mask but {method.algo_lower.upper()} was without mask"
                    elif (mask is None) and (cython_integr.cmask is not None):
                        cython_reset = f"no mask but { method.algo_lower.upper()} has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    if (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = f"radial_range was defined in { method.algo_lower.upper()}"
                    elif (radial_range is not None) and (cython_integr.pos0_range != radial_range):
                        cython_reset = f"radial_range is defined but differs in {method.algo_lower.upper()}"
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
                        cython_integr = self.setup_sparse_integrator(shape, npt, mask,
                                                                     radial_range, azimuth_range,
                                                                     mask_checksum=mask_crc,
                                                                     unit=unit, split=split, algo=method.algo_lower,
                                                                     empty=empty, scale=False)
                    except MemoryError:  # sparse method are hungry...
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
                        reset = "of first initialization"
                    if (not reset) and safe:
                        if integr.space != space:
                            reset = f"unit {integr.unit} incompatible with requested {unit}"
                        if numpy.prod(integr.bins) != numpy.prod(npt):
                            reset = f"number of points {integr.bins} incompatible with requested {npt}"
                        if integr.size != data.size:
                            reset = f"input image size {integr.size} incompatible with requested {data.size}"
                        if integr.empty != empty:
                            reset = f"empty value {integr.empty} incompatible with requested {empty}"
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
                            cython_integr = self.setup_sparse_integrator(shape, npt, mask,
                                                                         radial_range, azimuth_range,
                                                                         mask_checksum=mask_crc,
                                                                         unit=unit, split=split, algo=method.algo_lower,
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
                                integr = method.class_funct_ng.klass(cython_integr.lut,
                                                                     cython_integr.size,
                                                                     bin_centers=cython_integr.bin_centers0,
                                                                     azim_centers=cython_integr.bin_centers1,
                                                                     platformid=method.target[0],
                                                                     deviceid=method.target[1],
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc)

                        elif (method.impl_lower == "python"):
                            with ocl_py_engine.lock:
                                integr = method.class_funct_ng.klass(cython_integr.size,
                                                                     cython_integr.lut,
                                                                     bin_centers0=cython_integr.bin_centers0,
                                                                     bin_centers1=cython_integr.bin_centers1,
                                                                     checksum=cython_integr.lut_checksum,
                                                                     unit=unit, empty=empty,
                                                                     mask_checksum=mask_crc)
                        integr.pos0_range = cython_integr.pos0_range
                        integr.pos1_range = cython_integr.pos1_range
                        ocl_py_engine.set_engine(integr)

                    if (integr is not None):
                            intpl = integr.integrate_ng(data,
                                                       variance=variance,
                                                       error_model=error_model,
                                                       dark=dark, flat=flat,
                                                       solidangle=solidangle,
                                                       solidangle_checksum=self._dssa_crc,
                                                       dummy=dummy,
                                                       delta_dummy=delta_dummy,
                                                       polarization=polarization,
                                                       polarization_checksum=polarization_crc,
                                                       safe=safe,
                                                       normalization_factor=normalization_factor,
                                                       weighted_average=method.weighted_average,)
            if intpl is None:  # fallback if OpenCL failed or default cython
                # The integrator has already been initialized previously
                intpl = cython_integr.integrate_ng(data,
                                                   variance=variance,
                                                   error_model=error_model,
                                                   dummy=dummy,
                                                   delta_dummy=delta_dummy,
                                                   dark=dark,
                                                   flat=flat,
                                                   solidangle=solidangle,
                                                   polarization=polarization,
                                                   normalization_factor=normalization_factor,
                                                   weighted_average=method.weighted_average,)

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
                                   variance=variance,
                                   error_model=error_model,
                                   allow_pos0_neg=not radial_unit.positive,
                                   weighted_average=method.weighted_average,)

            elif method.split_lower == "bbox":
                logger.debug("integrate2d uses BBox implementation")
                pos0 = self.array_from_unit(shape, "center", radial_unit, scale=False)
                dpos0 = self.array_from_unit(shape, "delta", radial_unit, scale=False)
                pos1 = self.array_from_unit(shape, "center", azimuth_unit, scale=False)
                dpos1 = self.array_from_unit(shape, "delta", azimuth_unit, scale=False)
                integrator = method.class_funct_ng.function
                intpl = integrator(weights=data,
                                     pos0=pos0,
                                     delta_pos0=dpos0,
                                     pos1=pos1,
                                     delta_pos1=dpos1,
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
                                     variance=variance,
                                     error_model=error_model,
                                     allow_pos0_neg=not radial_unit.positive,
                                     clip_pos1=bool(azimuth_unit.period),
                                     weighted_average=method.weighted_average,)
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
                            reset = "of first initialization"
                        if (not reset) and safe:
                            if integr.space != space:
                                reset = f"unit {integr.unit} incompatible with requested {unit}"
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
                            if self._cached_array[f"{radial_unit.space}_crc"] != integr.on_device.get("radial"):
                                reset = "radial array changed"
                            if self._cached_array[f"{azimuth_unit.space}_crc"] != integr.on_device.get("azimuthal"):
                                reset = "azimuthal array changed"
                            # Nota: Ranges are enforced at runtime, not initialization
                        error = False
                        if reset:
                            logger.info("AI.integrate2d: Resetting OCL_Histogram2d integrator because %s", reset)
                            rad = self.array_from_unit(shape, typ="center", unit=radial_unit, scale=False)
                            rad_crc = self._cached_array[f"{radial_unit.space}_crc"] = crc32(rad)
                            azi = self.array_from_unit(shape, typ="center", unit=azimuth_unit, scale=False)
                            azi_crc = self._cached_array[f"{azimuth_unit.space}_crc"] = crc32(azi)
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
                                                 azimuthal_range=azimuth_range,
                                                 error_model=error_model,
                                                 weighted_average=method.weighted_average,)
####################
                else:  # if method.impl_lower in ["python", "cython"]:
                    logger.debug("integrate2d uses [CP]ython histogram implementation")
                    radial = self.array_from_unit(shape, "center", radial_unit, scale=False)
                    azim = self.array_from_unit(shape, "center", azimuth_unit, scale=False)
                    if method.impl_lower == "python":
                        data = data.astype(numpy.float32)  # it is important to make a copy see issue #88
                        mask = self.create_mask(data, mask, dummy, delta_dummy,
                                                unit=unit,
                                                radial_range=radial_range,
                                                azimuth_range=azimuth_range,
                                                mode="normal").ravel()
                    histogrammer = method.class_funct_ng.function
                    intpl = histogrammer(radial=radial,
                                         azimuthal=azim,
                                         bins=(npt_rad, npt_azim),
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
                                         empty=empty,
                                         variance=variance,
                                         dark_variance=None,
                                         error_model=error_model,
                                         radial_range=radial_range,
                                         azimuth_range=azimuth_range,
                                         allow_radial_neg=not radial_unit.positive,
                                         clip_pos1=bool(azimuth_unit.period),
                                         weighted_average=method.weighted_average,)

        I = intpl.intensity
        bins_azim = intpl.azimuthal
        bins_rad = intpl.radial
        signal2d = intpl.signal
        norm2d = intpl.normalization
        count = intpl.count
        if error_model.do_variance:
            std = intpl.std
            sem = intpl.sem
            var2d = intpl.variance
            norm2d_sq = intpl.norm_sq
        else:
            std = sem = var2d = norm2d_sq = None

        # Duplicate arrays on purpose ....
        bins_rad = bins_rad * pos0_scale
        bins_azim = bins_azim * pos1_scale

        result = Integrate2dResult(I, bins_rad, bins_azim, sem)
        result._set_method_called("integrate2d")
        result._set_compute_engine(str(method))
        result._set_method(method)
        result._set_radial_unit(radial_unit)
        result._set_azimuthal_unit(azimuth_unit)
        result._set_count(count)
        # result._set_sum(sum_)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_has_mask_applied(has_mask)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_metadata(metadata)

        result._set_sum_signal(signal2d)
        result._set_sum_normalization(norm2d)
        if error_model.do_variance:
            result._set_sum_normalization2(norm2d_sq)
            result._set_sum_variance(var2d)
            result._set_std(std)
            result._set_std(sem)

        if filename is not None:
            save_integrate_result(filename, result)

        return result

    integrate2d = _integrate2d_ng = integrate2d_ng

    @deprecated(since_version="2024.12.0", only_once=True, replacement="medfilt1d_ng", deprecated_since="2024.12.0")
    def medfilt1d_legacy(self, data, npt_rad=1024, npt_azim=512,
                  correctSolidAngle=True,
                  radial_range=None, azimuth_range=None,
                  polarization_factor=None, dark=None, flat=None,
                  method="splitpixel", unit=units.Q,
                  percentile=50, dummy=None, delta_dummy=None,
                  mask=None, normalization_factor=1.0, metadata=None):
        """Perform the 2D integration and filter along each row using a median filter

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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
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
            ctx = self.engines[res2d.method].engine.ctx
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
                if not ((lower >= 0).all() and (upper <= npt_azim).all()):
                    raise RuntimeError("Empty ensemble!")

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
                if not ((pos >= 0).all() and (pos <= npt_azim).all()):
                    raise RuntimeError("Empty ensemble!")
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

    medfilt1d = medfilt1d_legacy

    def medfilt1d_ng(self, data,
                     npt=1024,
                     correctSolidAngle=True,
                     polarization_factor=None,
                     variance=None,
                     error_model=ErrorModel.NO,
                     radial_range=None,
                     azimuth_range=None,
                     dark=None,
                     flat=None,
                     absorption=None,
                     method=("full", "csr", "cython"),
                     unit=units.Q,
                     percentile=50,
                     dummy=None,
                     delta_dummy=None,
                     mask=None,
                     normalization_factor=1.0,
                     metadata=None,
                     safe=True,
                     **kwargs):
        """Performs a median filter in azimuthal space:

        All pixels contributing to an azimuthal bin are sorted according to their corrected intensity (i.e. signal/norm).
        Then a cumulative sum is performed on their weight which allows to determine the location of the different quantiles.
        The percentile parameter (in the range [1:100]) can be:
        - either a single scalar, then the pixel with the nearest value to the quantile is used (i.e. the default value 50 provides the median).
        - either a 2-tuple, then the weighted average is calculated for all pixels between the two quantiles provided.

        Unlike sigma-clipping, this method is compatible with any kind of pixel splitting but much slower.

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
        :param ndarray absorption: Detector absorption (image)
        :param ndarray variance: the variance of the signal
        :param str error_model: can be "poisson" to assume a poissonian detector (variance=I) or "azimuthal" to take the std² in each ring (better, more expenive)
        :param unit: unit to be used for integration
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param percentile: which percentile use for cutting out.
                           percentil can be a 2-tuple to specify a region to average out,
                           like: (25,75) to average the second and third quartile.
        :param mask: masked out pixels array
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: any other metadata,
        :type metadata: JSON serializable dict
        :param safe: set to False to skip some tests
        :return: Integrate1D like result like

        The difference with the previous `medfilt_legacy` implementation is that there is no 2D regrouping.
        """
        for k in kwargs:
            if k == "npt_azim":
                logger.warning("'npt_azim' argument is not used in sigma_clip_ng as not 2D intergration is performed anymore")
            else:
                logger.warning("Got unknown argument %s %s", k, kwargs[k])

        error_model = ErrorModel.parse(error_model)
        if variance is not None:
            if variance.size != data.size:
                raise RuntimeError("variance array shape does not match data")
            error_model = ErrorModel.VARIANCE

        unit = units.to_unit(unit)
        if radial_range:
            radial_range = tuple(radial_range[i] / unit.scale for i in (0, -1))
        if azimuth_range is not None:
            azimuth_range = self.normalize_azimuth_range(azimuth_range)
        try:
            quant_min = min(percentile)/100
            quant_max = max(percentile)/100
        except:
            quant_min = quant_max = percentile/100.0

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
                    elif cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    elif cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    elif not nan_equal(cython_integr.empty, self._empty):
                        cython_reset = f"empty value changed {cython_integr.empty}!={self._empty}"
                    elif (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = "mask but CSR was without mask"
                    elif (mask is None) and (cython_integr.check_mask):
                        cython_reset = "no mask but CSR has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    elif (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and cython_integr.pos0_range != (min(radial_range), max(radial_range)):
                        cython_reset = "radial_range is defined but not the same as in CSR"
                    elif (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and cython_integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                        cython_reset = "azimuth_range requested and CSR's azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.sigma_clip_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        cython_integr = self.setup_sparse_integrator(data.shape, npt, mask=mask,
                                                       mask_checksum=mask_crc,
                                                       unit=unit, split=split, algo="CSR",
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
                    elif integr.bins != npt:
                        reset = "number of points changed"
                    elif integr.size != data.size:
                        reset = "input image size changed"
                    elif not nan_equal(integr.empty, self._empty):
                        reset = f"empty value changed {integr.empty}!={self._empty}"
                    elif (mask is not None) and (not integr.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    elif (radial_range is None) and (integr.pos0_range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                        reset = "radial_range is defined but not the same as in CSR"
                    elif (azimuth_range is None) and (integr.pos1_range is not None):
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
                          "flat":flat, "solidangle":solidangle, "polarization":polarization, "absorption":absorption,
                          "error_model":error_model, "normalization_factor":normalization_factor,
                          "quant_min":quant_min, "quant_max":quant_max}

                intpl = integr.medfilt(data, **kwargs)
        else:
            raise RuntimeError(f"Method {method} is not yet implemented. Please report an issue on https://github.com/silx-kit/pyFAI/issues/new")
        if intpl.variance is not None:
            if numpy.all(intpl.variance == numpy.zeros_like(intpl.variance)):
                result = Integrate1dResult(intpl.position * unit.scale, intpl.intensity)
            else:
                result = Integrate1dResult(intpl.position * unit.scale, intpl.intensity, intpl.sem)
        else:
                result = Integrate1dResult(intpl.position * unit.scale, intpl.intensity)
        result._set_method_called("sigma_clip_ng")
        result._set_method(method)
        result._set_compute_engine(str(method))
        result._set_percentile(percentile)
        result._set_unit(unit)
        result._set_has_mask_applied(has_mask)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_metadata(metadata)
        result._set_sum_signal(intpl.signal)
        result._set_sum_normalization(intpl.normalization)
        result._set_sum_normalization2(intpl.norm_sq)
        result._set_std(intpl.std)
        result._set_sem(intpl.sem)
        result._set_sum_variance(intpl.variance)
        result._set_count(intpl.count)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_error_model(error_model)
        return result

    def sigma_clip_legacy(self, data, npt_rad=1024, npt_azim=512,
                          correctSolidAngle=True, polarization_factor=None,
                          radial_range=None, azimuth_range=None,
                          dark=None, flat=None,
                          method=("full", "histogram", "cython"), unit=units.Q,
                          thres=3, max_iter=5, dummy=None, delta_dummy=None,
                          mask=None, normalization_factor=1.0, metadata=None,
                          safe=True, **kwargs):
        """Perform first a 2D integration and then an iterative sigma-clipping
        filter along each row. See the doc of scipy.stats.sigmaclip for the
        options `thres` and `max_iter`.

        :param data: input image as numpy array
        :param npt_rad: number of radial points (alias: npt)
        :param npt_azim: number of azimuthal points
        :param bool correctSolidAngle: correct for solid angle of each pixel when set
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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param thres: cut-off for n*sigma: discard any values with `|I-<I>| > thres*σ`.
                The threshold can be a 2-tuple with sigma_low and sigma_high.
        :param max_iter: maximum number of iterations
        :param mask: masked out pixels array
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: any other metadata,
        :type metadata: JSON serializable dict
        :param safe: unset to save some checks on sparse matrix shape/content.
        :kwargs: unused, just for signature compatibility when used within Worker.
        :return: Integrate1D like result like

        Nota: The initial 2D-integration requires pixel splitting
        """
        # compatibility layer with sigma_clip_ng
        if "npt" in kwargs:
            npt_rad = kwargs["npt"]
        # We use NaN as dummies
        if dummy is None:
            dummy = numpy.nan
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
                                 normalization_factor=normalization_factor,
                                 safe=safe)
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
            image[mask] = numpy.nan
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
                image[mask] = numpy.nan
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

    _sigma_clip_legacy = sigma_clip_legacy

    def sigma_clip(self, data,
                   npt=1024,
                   correctSolidAngle=True,
                   polarization_factor=None,
                   variance=None,
                   error_model=ErrorModel.NO,
                   radial_range=None,
                   azimuth_range=None,
                   dark=None,
                   flat=None,
                   absorption=None,
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

            ``|I - <I>| < thres * σ(I)``

        This enforces a symmetric, bell-shaped distibution (i.e. gaussian-like)
        and is very good at extracting background or amorphous isotropic scattering
        out of Bragg peaks.

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
        :param ndarray absorption: Detector absorption (image)
        :param ndarray variance: the variance of the signal
        :param str error_model: can be "poisson" to assume a poissonian detector (variance=I) or "azimuthal" to take the std² in each ring (better, more expenive)
        :param unit: unit to be used for integration
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param thres: cut-off for n*sigma: discard any values with (I-<I>)/sigma > thres.
        :param max_iter: maximum number of iterations
        :param mask: masked out pixels array
        :param float normalization_factor: Value of a normalization monitor
        :param metadata: any other metadata,
        :type metadata: JSON serializable dict
        :param safe: set to False to skip some tests
        :return: Integrate1D like result like

        The difference with the previous `sigma_clip_legacy` implementation is that there is no 2D regrouping.
        Pixel splitting should be avoided with this implementation.
        The standard deviation is usually smaller than previously and the signal cleaner.
        It is also slightly faster.

        The case neither `error_model`, nor `variance` is provided, fall-back on a poissonian model.

        """
        for k in kwargs:
            if k == "npt_azim":
                logger.warning("'npt_azim' argument is not used in sigma_clip_ng as not 2D intergration is performed anymore")
            else:
                logger.warning("Got unknown argument %s %s", k, kwargs[k])

        error_model = ErrorModel.parse(error_model)
        if variance is not None:
            if variance.size != data.size: raise RuntimeError("variance array shape does not match data shape")
            error_model = ErrorModel.VARIANCE

        unit = units.to_unit(unit)
        if radial_range:
            radial_range = tuple(radial_range[i] / unit.scale for i in (0, -1))
        if azimuth_range is not None:
            azimuth_range = self.normalize_azimuth_range(azimuth_range)

        method = self._normalize_method(method, dim=1, default=self.DEFAULT_METHOD_1D)
        if method.split != "no":
            logger.warning("Method %s is using a pixel-splitting scheme. sigma_clip_ng should be use WITHOUT PIXEL-SPLITTING! Your results are likely to be wrong!",
                           method)

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
                    elif cython_integr.bins != npt:
                        cython_reset = "number of points changed"
                    elif cython_integr.size != data.size:
                        cython_reset = "input image size changed"
                    elif not nan_equal(cython_integr.empty, self._empty):
                        cython_reset = f"empty value changed {cython_integr.empty}!={self._empty}"
                    elif (mask is not None) and (not cython_integr.check_mask):
                        cython_reset = "mask but CSR was without mask"
                    elif (mask is None) and (cython_integr.check_mask):
                        cython_reset = "no mask but CSR has mask"
                    elif (mask is not None) and (cython_integr.mask_checksum != mask_crc):
                        cython_reset = "mask changed"
                    elif (radial_range is None) and (cython_integr.pos0_range is not None):
                        cython_reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and cython_integr.pos0_range != (min(radial_range), max(radial_range)):
                        cython_reset = "radial_range is defined but not the same as in CSR"
                    elif (azimuth_range is None) and (cython_integr.pos1_range is not None):
                        cython_reset = "azimuth_range not defined and CSR had azimuth_range defined"
                    elif (azimuth_range is not None) and cython_integr.pos1_range != (min(azimuth_range), max(azimuth_range)):
                        cython_reset = "azimuth_range requested and CSR's azimuth_range don't match"
                if cython_reset:
                    logger.info("AI.sigma_clip_ng: Resetting Cython integrator because %s", cython_reset)
                    split = method.split_lower
                    if split == "pseudo":
                        split = "full"
                    try:
                        cython_integr = self.setup_sparse_integrator(data.shape, npt, mask=mask,
                                                       mask_checksum=mask_crc,
                                                       unit=unit, split=split, algo="CSR",
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
                    elif integr.bins != npt:
                        reset = "number of points changed"
                    elif integr.size != data.size:
                        reset = "input image size changed"
                    elif not nan_equal(integr.empty, self._empty):
                        reset = f"empty value changed {integr.empty}!={self._empty}"
                    elif (mask is not None) and (not integr.check_mask):
                        reset = "mask but CSR was without mask"
                    elif (mask is None) and (integr.check_mask):
                        reset = "no mask but CSR has mask"
                    elif (mask is not None) and (integr.mask_checksum != mask_crc):
                        reset = "mask changed"
                    elif (radial_range is None) and (integr.pos0_range is not None):
                        reset = "radial_range was defined in CSR"
                    elif (radial_range is not None) and integr.pos0_range != (min(radial_range), max(radial_range)):
                        reset = "radial_range is defined but not the same as in CSR"
                    elif (azimuth_range is None) and (integr.pos1_range is not None):
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
                          "flat":flat, "solidangle":solidangle, "polarization":polarization, "absorption":absorption,
                          "error_model":error_model, "normalization_factor":normalization_factor,
                          "cutoff":thres, "cycle":max_iter}

                intpl = integr.sigma_clip(data, **kwargs)
        else:
            raise RuntimeError("Not yet implemented. Sorry")
        result = Integrate1dResult(intpl.position * unit.scale, intpl.intensity, intpl.sem)
        result._set_method_called("sigma_clip_ng")
        result._set_method(method)
        result._set_compute_engine(str(method))
        result._set_percentile(thres)
        result._set_unit(unit)
        result._set_has_mask_applied(has_mask)
        result._set_has_dark_correction(has_dark)
        result._set_has_flat_correction(has_flat)
        result._set_metadata(metadata)
        result._set_sum_signal(intpl.signal)
        result._set_sum_normalization(intpl.normalization)
        result._set_sum_normalization2(intpl.norm_sq)
        result._set_std(intpl.std)
        result._set_sem(intpl.sem)
        result._set_sum_variance(intpl.variance)
        result._set_count(intpl.count)
        result._set_polarization_factor(polarization_factor)
        result._set_normalization_factor(normalization_factor)
        result._set_error_model(error_model)
        return result

    sigma_clip_ng = sigma_clip

    def separate(self, data, npt=1024,
                 unit="2th_deg", method=("full", "csr", "cython"),
                 polarization_factor=None,
                 percentile=50, mask=None, restore_mask=True):
        """
        Separate bragg signal from powder/amorphous signal using azimuthal median filering and projected back before subtraction.

        :param data: input image as numpy array
        :param npt: number of radial points
        :param unit: unit to be used for integration
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param polarization_factor: Value of the polarization factor (from -1 to +1), None to disable correction.
        :param percentile: which percentile use for cutting out
        :param mask: masked out pixels array
        :param restore_mask: masked pixels have the same value as input data provided
        :return: SeparateResult which the bragg & amorphous signal

        Note: the filtered 1D spectrum can be retrieved from
        `SeparateResult.radial` and `SeparateResult.intensity` attributes
        """

        filter_result = self.medfilt1d_ng(data, npt=npt,
                                       unit=unit, method=method,
                                       percentile=percentile,
                                       polarization_factor=polarization_factor,
                                       mask=mask)
        # This takes 100ms and is the next to be optimized.
        amorphous = self.calcfrom1d(filter_result.radial, filter_result.intensity,
                                    data.shape, mask=None,
                                    dim1_unit=unit,
                                    correctSolidAngle=True, polarization_factor=polarization_factor)
        bragg = data - amorphous
        if mask is None:
            mask = self.detector.mask
        if restore_mask and mask is not None:
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
        result._set_npt_azim(npt)
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
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param poissonian: If True, add some poisonian noise to the data to make
                           then more realistic
        :param grow_mask: grow mask in polar coordinated to accomodate pixel
            splitting algoritm
        :return: inpainting object which contains the restored image as .data
        """
        from ..ext import inpainting
        dummy = -1
        delta_dummy = 0.9
        method = IntegrationMethod.select_one_available(method, dim=2,
                                                        default=self.DEFAULT_METHOD_2D)
        if mask.shape != self.detector.shape:
            raise RuntimeError("Mask shape does not match detector size")
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

    def guess_max_bins(self, redundancy=1, search_range=None, unit="q_nm^-1", radial_range=None, azimuth_range=None):
        """
        Guess the maximum number of bins, considering the excpected minimum redundancy:

        :param redundancy: minimum number of pixel per bin
        :param search_range: the minimum and maximun number of bins to be considered
        :param unit: the unit to be considered like "2th_deg" or "q_nm^-1"
        :param radial_range: radial range to be considered, depends on unit !
        :param azimuth_range: azimuthal range to be considered
        :return: the minimum bin number providing the provided redundancy
        """
        img = numpy.empty(self.detector.shape, dtype=numpy.float32)
        dia = int(numpy.sqrt(img.shape[0] ** 2 + img.shape[1] ** 2))
        method = self._normalize_method(("no", "histogram", "cython"), dim=1, default=self.DEFAULT_METHOD_1D)
        unit = units.to_unit(unit)
        if search_range is None:
            ref = self.integrate1d(img, dia, method=method, unit=unit,
                                   azimuth_range=azimuth_range, radial_range=radial_range).count.min()
            if ref >= redundancy:
                search_range = (dia, 4 * dia)
            else:
                search_range = (2, dia)

        for i in range(*search_range):
            mini = self.integrate1d(img, i, method=method, unit=unit,
                                  azimuth_range=azimuth_range, radial_range=radial_range).count.min()
            if mini < redundancy:
                return i - 1

    def guess_polarization(self, img, npt_rad=None, npt_azim=360, unit="2th_deg",
                           method=("no", "csr", "cython"), target_rad=None):
        """Guess the polarization factor for the given image

        For this one performs several integration with different polarization factors
        and take the one with the lowest std along the outer-most ring.

        :param img: diffraction image, preferable with beam-stop centered.
        :param npt_rad: number of point in the radial dimension, can be guessed, better avoid oversampling.
        :param npt_azim: number of point in the azimuthal dimension, 1 per degree is usually OK
        :param unit: radial unit for the integration
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation). The  default one is pretty optimal: no splitting, CSR for the speed of the integration
        :param target_rad: position of the outer-most complete ring, can be guessed.
        :return: polarization factor (#, polarization angle)
        """
        if npt_rad is None:
            if self.detector.shape is None:
                self.detector.shape = img.shape
            npt_rad = self.guess_npt_rad()

        res = self.integrate2d_ng(img, npt_rad, npt_azim, unit=unit, method=method)

        if target_rad is None:
            azimuthal_range = (res.count > 0).sum(axis=0)
            azim_min = azimuthal_range.max() * 0.95
            valid_rings = numpy.where(azimuthal_range > azim_min)[0]
            nbpix = res.count.sum(axis=0)[valid_rings]
            bin_idx = valid_rings[numpy.where(nbpix.max() == nbpix)[0][-1]]
        else:
            bin_idx = numpy.argmin(abs(res.radial - target_rad))

        from scipy.optimize import minimize_scalar
        sfun = lambda p:\
            self.integrate2d_ng(img, npt_rad, npt_azim, unit=unit, method=method,
                                polarization_factor=p).intensity[:, bin_idx].std()
        opt = minimize_scalar(sfun, bounds=[-1, 1])
        logger.info(str(opt))
        return opt.x
