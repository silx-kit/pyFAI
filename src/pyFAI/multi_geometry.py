# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
#    Copyright (C)      2016 Synchrotron SOLEIL - L'Orme des Merisiers Saint-Aubin
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

"""Module for treating simultaneously multiple detector configuration
within a single integration"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/01/2024"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import collections.abc
import gc
import logging
logger = logging.getLogger(__name__)
from .integrator.azimuthal import AzimuthalIntegrator
from .integrator.fiber import FiberIntegrator
from .containers import Integrate1dResult
from .containers import Integrate2dResult
from . import units
from .utils.multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import threading
import numpy
from .method_registry import IntegrationMethod
error = None


class MultiGeometry(object):
    """This is an Azimuthal integrator containing multiple geometries,
    for example when the detector is on a goniometer arm
    """

    def __init__(self, ais, unit="2th_deg",
                 radial_range=None, azimuth_range=None,
                 wavelength=None, empty=0.0, chi_disc=180,
                 threadpoolsize=cpu_count()):
        """
        Constructor of the multi-geometry integrator

        :param ais: list of azimuthal integrators
        :param radial_range: common range for integration
        :param azimuthal_range: (2-tuple) common azimuthal range for integration
        :param empty: value for empty pixels
        :param chi_disc: if 0, set the chi_discontinuity at 0, else π
        :param threadpoolsize: By default, use a thread-pool to parallelize histogram/CSC integrator over as many threads as cores,
                               set to False/0 to serialize
        """
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.ais = [ai if isinstance(ai, AzimuthalIntegrator)
                    else AzimuthalIntegrator.sload(ai)
                    for ai in ais]
        self.wavelength = None
        self.threadpool = ThreadPool(min(len(self.ais), threadpoolsize)) if threadpoolsize>0 else None
        if wavelength:
            self.set_wavelength(wavelength)
        if isinstance(unit, (tuple, list)) and len(unit) == 2:
            self.radial_unit = units.to_unit(unit[0])
            self.azimuth_unit = units.to_unit(unit[1])
        else:
            self.radial_unit = units.to_unit(unit)
            self.azimuth_unit = units.CHI_DEG
        self.unit = (self.radial_unit, self.azimuth_unit)
        self.radial_range = radial_range
        self.azimuth_range = azimuth_range
        self.abolute_solid_angle = None
        self.empty = empty
        if chi_disc == 0:
            for ai in self.ais:
                ai.setChiDiscAtZero()
        elif chi_disc == 180:
            for ai in self.ais:
                ai.setChiDiscAtPi()
        else:
            logger.warning("Unable to set the Chi discontinuity at %s", chi_disc)

    def __del__(self):
        if self.threadpool and self.threadpool._state == "RUN":
            self.threadpool.close()

    def __repr__(self, *args, **kwargs):
        return "MultiGeometry integrator with %s geometries on %s radial range (%s) and %s azimuthal range (deg)" % \
            (len(self.ais), self.radial_range, self.unit, self.azimuth_range)

    def _guess_radial_range(self):
        logger.info(f"Calculating the radial range of MultiGeometry...")
        radial = numpy.array([ai.array_from_unit(unit=self.radial_unit) for ai in self.ais])
        return (radial.min(), radial.max())

    def _guess_azimuth_range(self):
        logger.info(f"Calculating the azimuthal range of MultiGeometry...")
        azimuthal = numpy.array([ai.array_from_unit(unit=self.azimuth_unit) for ai in self.ais])
        return (azimuthal.min(), azimuthal.max())

    def integrate1d(self, lst_data, npt=1800,
                    correctSolidAngle=True,
                    lst_variance=None, error_model=None,
                    polarization_factor=None,
                    normalization_factor=None,
                    lst_mask=None, lst_flat=None,
                    method=("full", "histogram", "cython")):
        """Perform 1D azimuthal integration

        :param lst_data: list of numpy array
        :param npt: number of points int the integration
        :param correctSolidAngle: correct for solid angle (all processing are then done in absolute solid angle !)
        :param lst_variance: list of array containing the variance of the data. If not available, no error propagation is done
        :type lst_variance: list of ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param polarization_factor: Apply polarization correction ? is None: not applies. Else provide a value from -1 to +1
        :param normalization_factor: normalization monitors value (list of floats)
        :param all: return a dict with all information in it (deprecated, please refer to the documentation of Integrate1dResult).
        :param lst_mask: numpy.Array or list of numpy.array which mask the lst_data.
        :param lst_flat: numpy.Array or list of numpy.array which flat the lst_data.
        :param method: integration method, a string or a registered method
        :return: 2th/I or a dict with everything depending on "all"
        :rtype: Integrate1dResult, dict
        """
        method = IntegrationMethod.select_one_available(method, dim=1)

        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.ais)
        elif not isinstance(normalization_factor, collections.abc.Iterable):
            normalization_factor = [normalization_factor] * len(self.ais)
        if lst_variance is None:
            lst_variance = [None] * len(self.ais)
        if lst_mask is None:
            lst_mask = [None] * len(self.ais)
        elif isinstance(lst_mask, numpy.ndarray):
            lst_mask = [lst_mask] * len(self.ais)
        if lst_flat is None:
            lst_flat = [None] * len(self.ais)
        elif isinstance(lst_flat, numpy.ndarray):
            lst_flat = [lst_flat] * len(self.ais)
        signal = numpy.zeros(npt, dtype=numpy.float64)
        normalization = numpy.zeros_like(signal)
        count = numpy.zeros_like(signal)
        variance = None
        if self.radial_range is None:
            self.radial_range = self._guess_radial_range()
        if self.azimuth_range is None:
            self.azimuth_range = self._guess_azimuth_range()
        def _integrate(args):
            ai, data, monitor, var, mask, flat = args
            return ai.integrate1d_ng(data, npt=npt,
                                    correctSolidAngle=correctSolidAngle,
                                    variance=var, error_model=error_model,
                                    polarization_factor=polarization_factor,
                                    radial_range=self.radial_range,
                                    azimuth_range=self.azimuth_range,
                                    method=method, unit=self.radial_unit, safe=True,
                                    mask=mask, flat=flat, normalization_factor=monitor)
        if self.threadpool is None:
            results = map(_integrate,
                          zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        else:
            results = self.threadpool.map(_integrate,
                                          zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        for res, ai in zip(results, self.ais):
            sac = (ai.pixel1 * ai.pixel2 / ai.dist ** 2) if correctSolidAngle else 1.0
            count += res.count
            normalization += res.sum_normalization * sac
            signal += res.sum_signal
            if res.sigma is not None:
                if variance is None:
                    variance = res.sum_variance.astype(dtype=numpy.float64)  # explicit copy
                else:
                    variance += res.sum_variance

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(normalization, tiny)
        invalid = count <= 0.0
        I = signal / norm
        I[invalid] = self.empty

        if variance is not None:
            sigma = numpy.sqrt(variance) / norm
            sigma[invalid] = self.empty
            result = Integrate1dResult(res.radial, I, sigma)
        else:
            result = Integrate1dResult(res.radial, I)
        result._set_compute_engine(res.compute_engine)
        result._set_unit(self.radial_unit)
        result._set_sum_signal(signal)
        result._set_sum_normalization(normalization)
        result._set_sum_variance(variance)
        result._set_count(count)
        return result

    def integrate2d(self, lst_data, npt_rad=1800, npt_azim=3600,
                    correctSolidAngle=True,
                    lst_variance=None, error_model=None,
                    polarization_factor=None,
                    normalization_factor=None,
                    lst_mask=None, lst_flat=None,
                    method=("full", "histogram", "cython")):
        """Performs 2D azimuthal integration of multiples frames, one for each geometry

        :param lst_data: list of numpy array
        :param npt: number of points int the integration
        :param correctSolidAngle: correct for solid angle (all processing are then done in absolute solid angle !)
        :param lst_variance: list of array containing the variance of the data. If not available, no error propagation is done
        :type lst_variance: list of ndarray
        :param error_model: When the variance is unknown, an error model can be given: "poisson" (variance = I), "azimuthal" (variance = (I-<I>)^2)
        :type error_model: str
        :param polarization_factor: Apply polarization correction ? is None: not applies. Else provide a value from -1 to +1
        :param normalization_factor: normalization monitors value (list of floats)
        :param all: return a dict with all information in it (deprecated, please refer to the documentation of Integrate2dResult).
        :param lst_mask: numpy.Array or list of numpy.array which mask the lst_data.
        :param lst_flat: numpy.Array or list of numpy.array which flat the lst_data.
        :param method: integration method (or its name)
        :return: I/2th/chi or a dict with everything depending on "all"
        :rtype: Integrate2dResult, dict
        """
        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.ais)
        elif not isinstance(normalization_factor, collections.abc.Iterable):
            normalization_factor = [normalization_factor] * len(self.ais)
        if lst_variance is None:
            lst_variance = [None] * len(self.ais)
        if lst_mask is None:
            lst_mask = [None] * len(self.ais)
        elif isinstance(lst_mask, numpy.ndarray):
            lst_mask = [lst_mask] * len(self.ais)
        if lst_flat is None:
            lst_flat = [None] * len(self.ais)
        elif isinstance(lst_flat, numpy.ndarray):
            lst_flat = [lst_flat] * len(self.ais)

        method = IntegrationMethod.select_one_available(method, dim=2)

        signal = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        count = numpy.zeros_like(signal)
        normalization = numpy.zeros_like(signal)
        variance = None
        if self.radial_range is None:
            self.radial_range = self._guess_radial_range()
        if self.azimuth_range is None:
            self.azimuth_range = self._guess_azimuth_range()
        def _integrate(args):
            ai, data, monitor, var, mask, flat = args
            return ai.integrate2d_ng(data,npt_rad=npt_rad, npt_azim=npt_azim,
                                    correctSolidAngle=correctSolidAngle,
                                    variance=var, error_model=error_model,
                                    polarization_factor=polarization_factor,
                                    radial_range=self.radial_range,
                                    azimuth_range=self.azimuth_range,
                                    method=method, unit=self.unit, safe=True,
                                    mask=mask, flat=flat, normalization_factor=monitor)
        if self.threadpool is None:
            results = map(_integrate,
                          zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        else:
            results = self.threadpool.map(_integrate,
                zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        for res, ai in zip(results, self.ais):
            sac = (ai.pixel1 * ai.pixel2 / ai.dist ** 2) if correctSolidAngle else 1.0
            count += res.count
            signal += res.sum_signal
            normalization += res.sum_normalization * sac
            if res.sigma is not None:
                if variance is None:
                    variance = res.sum_variance.astype(numpy.float64)  # explicit copy !
                else:
                    variance += res.sum_variance

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(normalization, tiny)
        invalid = count <= 0
        I = signal / norm
        I[invalid] = self.empty

        if variance is not None:
            sigma = numpy.sqrt(variance) / norm
            sigma[invalid] = self.empty
            result = Integrate2dResult(I, res.radial, res.azimuthal, sigma)
        else:
            result = Integrate2dResult(I, res.radial, res.azimuthal)
        result._set_sum(signal)
        result._set_compute_engine(res.compute_engine)
        result._set_radial_unit(self.radial_unit)
        result._set_azimuthal_unit(self.azimuth_unit)
        result._set_sum_signal(signal)
        result._set_sum_normalization(normalization)
        result._set_sum_variance(variance)
        result._set_count(count)
        return result

    def set_wavelength(self, value):
        """
        Changes the wavelength of a group of azimuthal integrators
        """
        self.wavelength = float(value)
        for ai in self.ais:
            ai.set_wavelength(self.wavelength)

    def reset(self, collect_garbage=True):
        """Clean up all caches for all integrators, resets the thread-pool as well.

        :param collect_garbage: set to False to prevent garbage collection, faster
        """
        for ai in self.ais:
            ai.reset(collect_garbage=False)
        if self.threadpool:
            try:
                threadpoolsize = self.threadpool._processes
            except Exception as err:
                print(f"{type(err)}: {err}")
                threadpoolsize = 1
            self.threadpool.terminate()
            self.threadpool = ThreadPool(threadpoolsize)
        if collect_garbage:
            gc.collect()

class MultiGeometryFiber(object):
    """This is a Fiber integrator containing multiple geometries,
    for example when the detector is on a goniometer arm
    """

    def __init__(self, fis, unit=("qip_nm^-1", "qoop_nm^-1"),
                 ip_range=None, oop_range=None,
                 incident_angle=None, tilt_angle=None, sample_orientation=None,
                 wavelength=None, empty=0.0, chi_disc=180,
                 threadpoolsize=cpu_count()):
        """
        Constructor of the multi-geometry integrator

        :param ais: list of azimuthal integrators
        :param ip_range: (2-tuple) in-plane range for integration
        :param oop_range: (2-tuple) out-of-plane range for integration
        :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
        :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
        :param int sample_orientation: 1-4, four different orientation of the fiber axis regarding the detector main axis, from 1 to 4 is +90º
        :param empty: value for empty pixels
        :param chi_disc: if 0, set the chi_discontinuity at 0, else π
        :param threadpoolsize: By default, use a thread-pool to parallelize histogram/CSC integrator over as many threads as cores,
                               set to False/0 to serialize
        """
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.fis = [fi if isinstance(fi, FiberIntegrator)
                    else FiberIntegrator.sload(fi)
                    for fi in fis]
        self.wavelength = None
        self.threadpool = ThreadPool(min(len(self.fis), threadpoolsize)) if threadpoolsize>0 else None
        if wavelength:
            self.set_wavelength(wavelength)
        if isinstance(unit, (tuple, list)) and len(unit) == 2:
            self.ip_unit = units.parse_fiber_unit(unit=unit[0],
                                                  incident_angle=incident_angle,
                                                  tilt_angle=tilt_angle,
                                                  sample_orientation=sample_orientation,
                                                  )
            self.oop_unit = units.parse_fiber_unit(unit=unit[1],
                                                   incident_angle=self.ip_unit.incident_angle,
                                                   tilt_angle=self.ip_unit.tilt_angle,
                                                   sample_orientation=self.ip_unit.sample_orientation,
                                                   )
        else:
            self.ip_unit = units.parse_fiber_unit(unit=unit,
                                                  incident_angle=incident_angle,
                                                  tilt_angle=tilt_angle,
                                                  sample_orientation=sample_orientation,
                                                  )
            self.oop_unit = units.parse_fiber_unit(unit="qoop_nm^-1",
                                                   incident_angle=self.ip_unit.incident_angle,
                                                   tilt_angle=self.ip_unit.tilt_angle,
                                                   sample_orientation=self.ip_unit.sample_orientation,
                                                   )

        self.unit = (self.ip_unit, self.oop_unit)
        self.ip_range = ip_range
        self.oop_range = oop_range
        self.abolute_solid_angle = None
        self.empty = empty
        if chi_disc == 0:
            for fi in self.fis:
                fi.setChiDiscAtZero()
        elif chi_disc == 180:
            for fi in self.fis:
                fi.setChiDiscAtPi()
        else:
            logger.warning("Unable to set the Chi discontinuity at %s", chi_disc)

    def __del__(self):
        if self.threadpool and self.threadpool._state == "RUN":
            self.threadpool.close()

    def __repr__(self, *args, **kwargs):
        return "MultiGeometry integrator with %s geometries on %s radial range (%s) and %s azimuthal range (deg)" % \
            (len(self.fis), self.ip_range, self.unit, self.oop_range)

    def _guess_inplane_range(self):
        logger.info(f"Calculating the in-plane range of MultiGeometry...")
        ip = numpy.array([fi.array_from_unit(unit=self.ip_unit) for fi in self.fis])
        return (ip.min(), ip.max())

    def _guess_outofplane_range(self):
        logger.info(f"Calculating the out-of-plane range of MultiGeometry...")
        oop = numpy.array([fi.array_from_unit(unit=self.oop_unit) for fi in self.fis])
        return (oop.min(), oop.max())

    def integrate_fiber(self, lst_data,
                          npt_ip=1000, npt_oop=1000,
                          correctSolidAngle=True,
                          vertical_integration = True,
                          lst_mask=None, dummy=None, delta_dummy=None,
                          lst_variance=None,
                          polarization_factor=None, dark=None, lst_flat=None,
                          method=("no", "histogram", "cython"),
                          normalization_factor=1.0, **kwargs):
        """Performs 1D fiber integration of multiples frames, one for each geometry,
        It wraps the method integrate_fiber of pyFAI.integrator.fiber.FiberIntegrator

        :param lst_data: list of numpy array
        :param int npt_ip: number of points to be used along the in-plane axis
        :param int npt_oop: number of points to be used along the out-of-plane axis
        :param bool vertical_integration: If True, integrates along unit_ip; if False, integrates along unit_oop
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param lst_mask: numpy.Array or list of numpy.array which mask the lst_data.
        :param float dummy: value for dead/masked pixels
        :param float delta_dummy: precision for dummy value
        :param lst_variance: list of array containing the variance of the data. If not available, no error propagation is done
        :type lst_variance: list of ndarray
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
                * 0 for circular polarization or random,
                * None for no correction,
                * True for using the former correction
        :param ndarray dark: dark noise image
        :param lst_flat: numpy.Array or list of numpy.array which flat the lst_data.
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param float normalization_factor: Value of a normalization monitor
        :return: chi bins center positions and regrouped intensity
        :rtype: Integrate1dResult
        """
        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        if vertical_integration and npt_oop is None:
            raise RuntimeError("npt_oop (out-of-plane bins) is needed to do the integration")
        elif not vertical_integration and npt_ip is None:
            raise RuntimeError("npt_ip (in-plane bins) is needed to do the integration")

        if self.ip_range is None:
            self.ip_range = self.ip_range or self._guess_inplane_range()
        if self.oop_range is None:
            self.oop_range = self.oop_range or self._guess_outofplane_range()

        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.fis)
        elif not isinstance(normalization_factor, collections.abc.Iterable):
            normalization_factor = [normalization_factor] * len(self.fis)
        if lst_variance is None:
            lst_variance = [None] * len(self.fis)
        if lst_mask is None:
            lst_mask = [None] * len(self.fis)
        elif isinstance(lst_mask, numpy.ndarray):
            lst_mask = [lst_mask] * len(self.fis)
        if lst_flat is None:
            lst_flat = [None] * len(self.fis)
        elif isinstance(lst_flat, numpy.ndarray):
            lst_flat = [lst_flat] * len(self.fis)

        method = IntegrationMethod.select_one_available(method, dim=1)
        signal = numpy.zeros(npt_oop, dtype=numpy.float64)
        normalization = numpy.zeros_like(signal)
        count = numpy.zeros_like(signal)
        variance = None

        def _integrate(args):
            fi, data, monitor, var, mask, flat = args
            return fi.integrate_fiber(data=data,
                                      npt_oop=npt_oop, unit_oop=self.oop_unit, oop_range=self.oop_range,
                                      npt_ip=npt_ip, unit_ip=self.ip_unit, ip_range=self.ip_range,
                                      vertical_integration=vertical_integration,
                                      correctSolidAngle=correctSolidAngle,
                                      mask=mask, dummy=dummy, delta_dummy=delta_dummy,
                                      polarization_factor=polarization_factor, dark=dark, flat=flat,
                                      method=("no", "histogram", "cython"),
                                      normalization_factor=monitor,
                                      variance=var,
                                      )
        if self.threadpool is None:
            results = map(_integrate,
                          zip(self.fis, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        else:
            results = self.threadpool.map(_integrate,
                                          zip(self.fis, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        for res, fi in zip(results, self.fis):
            sac = (fi.pixel1 * fi.pixel2 / fi.dist ** 2) if correctSolidAngle else 1.0
            count += res.count
            normalization += res.sum_normalization * sac
            signal += res.sum_signal
            if res.sigma is not None:
                if variance is None:
                    variance = res.sum_variance.astype(dtype=numpy.float64)  # explicit copy
                else:
                    variance += res.sum_variance

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(normalization, tiny)
        invalid = count <= 0.0
        I = signal / norm
        I[invalid] = self.empty
        if variance is not None:
            sigma = numpy.sqrt(variance) / norm
            sigma[invalid] = self.empty
            result = Integrate1dResult(res.radial, I, sigma)
        else:
            result = Integrate1dResult(res.radial, I)
        result._set_compute_engine(res.compute_engine)

        if vertical_integration:
            output_unit = self.oop_unit
        else:
            output_unit = self.ip_unit

        result._set_unit(output_unit)
        result._set_sum_signal(signal)
        result._set_sum_normalization(normalization)
        result._set_sum_variance(variance)
        result._set_count(count)
        return result

    integrate_grazing_incidence = integrate_fiber
    integrate1d_grazing_incidence = integrate_grazing_incidence
    integrate1d_fiber = integrate_fiber
    integrate1d = integrate1d_fiber

    def integrate2d_fiber(self, lst_data,
                          npt_ip=1000, npt_oop=1000,
                          correctSolidAngle=True,
                          lst_mask=None, dummy=None, delta_dummy=None,
                          lst_variance=None,
                          polarization_factor=None, dark=None, lst_flat=None,
                          method=("no", "histogram", "cython"),
                          normalization_factor=1.0, **kwargs):
        """Performs 2D azimuthal integration of multiples frames, one for each geometry,
        It wraps the method integrate2d_fiber of pyFAI.integrator.fiber.FiberIntegrator

        :param lst_data: list of numpy array
        :param int npt_ip: number of points to be used along the in-plane axis
        :param pyFAI.units.UnitFiber/str unit_ip: unit to describe the in-plane axis. If not provided, it takes qip_nm^-1
        :param list ip_range: The lower and upper range of the in-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int npt_oop: number of points to be used along the out-of-plane axis
        :param pyFAI.units.UnitFiber/str unit_oop: unit to describe the out-of-plane axis. If not provided, it takes qoop_nm^-1
        :param list oop_range: The lower and upper range of the out-of-plane unit. If not provided, range is simply (data.min(), data.max()). Values outside the range are ignored. Optional.
        :param int sample_orientation: 1-4, four different orientation of the fiber axis regarding the detector main axis, from 1 to 4 is +90º
        :param bool correctSolidAngle: correct for solid angle of each pixel if True
        :param lst_mask: numpy.Array or list of numpy.array which mask the lst_data.
        :param float dummy: value for dead/masked pixels
        :param float delta_dummy: precision for dummy value
        :param lst_variance: list of array containing the variance of the data. If not available, no error propagation is done
        :type lst_variance: list of ndarray
        :param float polarization_factor: polarization factor between -1 (vertical) and +1 (horizontal).
                * 0 for circular polarization or random,
                * None for no correction,
                * True for using the former correction
        :param ndarray dark: dark noise image
        :param lst_flat: numpy.Array or list of numpy.array which flat the lst_data.
        :param IntegrationMethod method: IntegrationMethod instance or 3-tuple with (splitting, algorithm, implementation)
        :param float normalization_factor: Value of a normalization monitor
        :return: regrouped intensity and unit arrays
        :rtype: Integrate2dResult
        """
        if (isinstance(method, (tuple, list)) and method[0] != "no") or (isinstance(method, IntegrationMethod) and method.split != "no"):
            logger.warning(f"Method {method} is using a pixel-splitting scheme. GI integration should be use WITHOUT PIXEL-SPLITTING! The results could be wrong!")

        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.fis)
        elif not isinstance(normalization_factor, collections.abc.Iterable):
            normalization_factor = [normalization_factor] * len(self.fis)
        if lst_variance is None:
            lst_variance = [None] * len(self.fis)
        if lst_mask is None:
            lst_mask = [None] * len(self.fis)
        elif isinstance(lst_mask, numpy.ndarray):
            lst_mask = [lst_mask] * len(self.fis)
        if lst_flat is None:
            lst_flat = [None] * len(self.fis)
        elif isinstance(lst_flat, numpy.ndarray):
            lst_flat = [lst_flat] * len(self.fis)

        method = IntegrationMethod.select_one_available(method, dim=2)
        signal = numpy.zeros((npt_oop, npt_ip), dtype=numpy.float64)
        count = numpy.zeros_like(signal)
        normalization = numpy.zeros_like(signal)
        variance = None

        if self.ip_range is None:
            self.ip_range = self.ip_range or self._guess_inplane_range()
        if self.oop_range is None:
            self.oop_range = self.oop_range or self._guess_outofplane_range()

        def _integrate(args):
            fi, data, monitor, var, mask, flat = args
            return fi.integrate2d_fiber(data,
                                        npt_ip=npt_ip, unit_ip=self.ip_unit, ip_range=self.ip_range,
                                        npt_oop=npt_oop, unit_oop=self.oop_unit, oop_range=self.oop_range,
                                        correctSolidAngle=correctSolidAngle,
                                        variance=var,
                                        polarization_factor=polarization_factor,
                                        method=method, safe=True,
                                        dummy=dummy, delta_dummy=delta_dummy,
                                        mask=mask, flat=flat, dark=dark, normalization_factor=monitor, **kwargs)
        if self.threadpool is None:
            results = map(_integrate,
                          zip(self.fis, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        else:
            results = self.threadpool.map(_integrate,
                zip(self.fis, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat))
        for res, ai in zip(results, self.fis):
            sac = (ai.pixel1 * ai.pixel2 / ai.dist ** 2) if correctSolidAngle else 1.0
            count += res.count
            signal += res.sum_signal
            normalization += res.sum_normalization * sac
            if res.sigma is not None:
                if variance is None:
                    variance = res.sum_variance.astype(numpy.float64)  # explicit copy !
                else:
                    variance += res.sum_variance

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(normalization, tiny)
        invalid = count <= 0
        I = signal / norm
        I[invalid] = self.empty

        if variance is not None:
            sigma = numpy.sqrt(variance) / norm
            sigma[invalid] = self.empty
            result = Integrate2dResult(I, res.radial, res.azimuthal, sigma)
        else:
            result = Integrate2dResult(I, res.radial, res.azimuthal)
        result._set_sum(signal)
        result._set_compute_engine(res.compute_engine)
        result._set_radial_unit(self.ip_unit)
        result._set_azimuthal_unit(self.oop_unit)
        result._set_sum_signal(signal)
        result._set_sum_normalization(normalization)
        result._set_sum_variance(variance)
        result._set_count(count)
        return result

    integrate2d_grazing_incidence = integrate2d_fiber
    integrate2d = integrate2d_fiber
