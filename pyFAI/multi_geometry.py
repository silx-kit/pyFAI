# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import collections
import logging
logger = logging.getLogger(__name__)
from .azimuthalIntegrator import AzimuthalIntegrator
from .containers import Integrate1dResult
from .containers import Integrate2dResult
from . import units
import threading
import numpy
error = None


class MultiGeometry(object):
    """This is an Azimuthal integrator containing multiple geometries,
    for example when the detector is on a goniometer arm
    """

    def __init__(self, ais, unit="2th_deg",
                 radial_range=(0, 180), azimuth_range=(-180, 180),
                 wavelength=None, empty=0.0, chi_disc=180):
        """
        Constructor of the multi-geometry integrator
        :param ais: list of azimuthal integrators
        :param radial_range: common range for integration
        :param azimuthal_range: common range for integration
        :param empty: value for empty pixels
        :param chi_disc: if 0, set the chi_discontinuity at
        """
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.ais = [ai if isinstance(ai, AzimuthalIntegrator)
                    else AzimuthalIntegrator.sload(ai)
                    for ai in ais]
        self.wavelength = None
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

    def __repr__(self, *args, **kwargs):
        return "MultiGeometry integrator with %s geometries on %s radial range (%s) and %s azimuthal range (deg)" % \
            (len(self.ais), self.radial_range, self.unit, self.azimuth_range)

    def integrate1d(self, lst_data, npt=1800,
                    correctSolidAngle=True,
                    lst_variance=None, error_model=None,
                    polarization_factor=None,
                    normalization_factor=None, all=False,
                    lst_mask=None, lst_flat=None):
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
        :return: 2th/I or a dict with everything depending on "all"
        :rtype: Integrate1dResult, dict
        """
        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.ais)
        elif not isinstance(normalization_factor, collections.Iterable):
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
        sum_ = numpy.zeros(npt, dtype=numpy.float64)
        count = numpy.zeros(npt, dtype=numpy.float64)
        sigma2 = None
        for ai, data, monitor, variance, mask, flat in zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat):
            res = ai.integrate1d(data, npt=npt,
                                 correctSolidAngle=correctSolidAngle,
                                 variance=variance, error_model=error_model,
                                 polarization_factor=polarization_factor,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 mask=mask, flat=flat)
            sac = (ai.pixel1 * ai.pixel2 / ai.dist ** 2) if correctSolidAngle else 1.0
            count += res.count * sac
            sum_ += res.sum / monitor
            if res.sigma is not None:
                if sigma2 is None:
                    sigma2 = numpy.zeros(npt, dtype=numpy.float64)
                sigma2 += (res.sigma ** 2) / monitor

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(count, tiny)
        invalid = count <= 0.0
        I = sum_ / norm
        I[invalid] = self.empty

        if sigma2 is not None:
            sigma = numpy.sqrt(sigma2) / norm
            sigma[invalid] = self.empty
            result = Integrate1dResult(res.radial, I, sigma)
        else:

            result = Integrate1dResult(res.radial, I)
        result._set_unit(self.unit)
        result._set_sum(sum_)
        result._set_count(count)

        if all:
            logger.warning("integrate1d(all=True) is deprecated. Please refer to the documentation of Integrate2dResult")
            out = {"I": I,
                   "radial": res.radial,
                   "unit": self.unit,
                   "count": count,
                   "sum": sum_}
            return out

        return result

    def integrate2d(self, lst_data, npt_rad=1800, npt_azim=3600,
                    correctSolidAngle=True,
                    lst_variance=None, error_model=None,
                    polarization_factor=None,
                    normalization_factor=None, all=False, lst_mask=None, lst_flat=None):
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
        :return: I/2th/chi or a dict with everything depending on "all"
        :rtype: Integrate2dResult, dict
        """
        if len(lst_data) == 0:
            raise RuntimeError("List of images cannot be empty")
        if normalization_factor is None:
            normalization_factor = [1.0] * len(self.ais)
        elif not isinstance(normalization_factor, collections.Iterable):
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
        sum_ = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        count = numpy.zeros_like(sum_)
        sigma2 = None
        for ai, data, monitor, variance, mask, flat in zip(self.ais, lst_data, normalization_factor, lst_variance, lst_mask, lst_flat):
            res = ai.integrate2d(data, npt_rad=npt_rad, npt_azim=npt_azim,
                                 correctSolidAngle=correctSolidAngle,
                                 variance=variance, error_model=error_model,
                                 polarization_factor=polarization_factor,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 mask=mask, flat=flat)
            sac = (ai.pixel1 * ai.pixel2 / ai.dist ** 2) if correctSolidAngle else 1.0
            count += res.count * sac
            sum_ += res.sum / monitor
            if res.sigma is not None:
                if sigma2 is None:
                    sigma2 = count = numpy.zeros_like(sum_)
                sigma2 += (res.sigma ** 2) / monitor

        tiny = numpy.finfo("float32").tiny
        norm = numpy.maximum(count, tiny)
        invalid = count <= 0
        I = sum_ / norm
        I[invalid] = self.empty

        if sigma2 is not None:
            sigma = numpy.sqrt(sigma2) / norm
            sigma[invalid] = self.empty
            result = Integrate2dResult(I, res.radial, res.azimuthal, sigma)
        else:
            result = Integrate2dResult(I, res.radial, res.azimuthal)
        result._set_sum(sum_)
        result._set_count(count)
        result._set_unit(self.unit)

        if all:
            logger.warning("integrate1d(all=True) is deprecated. Please refer to the documentation of Integrate2dResult")
            out = {"I": I,
                   "radial": res.radial,
                   "azimuthal": res.azimuthal,
                   "count": count,
                   "sum": sum_,
                   "unit": self.unit}
            return out

        return result

    def set_wavelength(self, value):
        """
        Changes the wavelength of a group of azimuthal integrators
        """
        self.wavelength = float(value)
        for ai in self.ais:
            ai.set_wavelength(self.wavelength)
