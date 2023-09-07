# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2021 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "08/03/2023"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import collections.abc
import logging
logger = logging.getLogger(__name__)
from .azimuthalIntegrator import AzimuthalIntegrator
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
                 radial_range=(0, 180), azimuth_range=None,
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
        if azimuth_range is None:
            azimuth_range = (-180, 180) if chi_disc else (0, 360)
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.ais = [ai if isinstance(ai, AzimuthalIntegrator)
                    else AzimuthalIntegrator.sload(ai)
                    for ai in ais]
        self.wavelength = None
        self.threadpool = ThreadPool(min(len(self.ais), threadpoolsize)) if threadpoolsize>0 else None
        if wavelength:
            self.set_wavelength(wavelength)
        self.radial_range = tuple(radial_range[:2])
        self.azimuth_range = tuple(azimuth_range[:2])
        self.unit = units.to_unit(unit)
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
        def _integrate(args):
            ai, data, monitor, var, mask, flat = args
            return ai.integrate1d_ng(data, npt=npt,
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
        result._set_unit(self.unit)
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
        result._set_unit(self.unit)
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
