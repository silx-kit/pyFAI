#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module containing holder classes, like returned objects."""

__authors__ = ["Valentin Valls", "Jérôme Kieffer"]
__contact__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/12/2025"
__status__ = "development"

import math
import copy
import logging
import warnings
from typing import NamedTuple
from enum import IntEnum
from dataclasses import fields, asdict  # noqa
import numpy
from numpy.typing import ArrayLike
from .utils.dataclasses import dataclass, case_insensitive_dataclass  # noqa
from .utils.decorators import deprecated_warning


logger = logging.getLogger(__name__)


# Few named tuples
class PolarizationArray(NamedTuple):
    array: ArrayLike
    checksum: int


class PolarizationDescription(NamedTuple):
    polarization_factor: float
    axis_offset: float = 0.0


class Integrate1dtpl(NamedTuple):
    """Result of any engines after 1d integration"""

    position: ArrayLike
    intensity: ArrayLike
    sigma: ArrayLike
    signal: ArrayLike
    variance: ArrayLike
    normalization: ArrayLike
    count: ArrayLike
    std: ArrayLike = None
    sem: ArrayLike = None
    norm_sq: ArrayLike = None


class Integrate2dtpl(NamedTuple):
    """Result of any engines after 2d integration"""

    radial: ArrayLike
    azimuthal: ArrayLike
    intensity: ArrayLike
    sigma: ArrayLike
    signal: ArrayLike
    variance: ArrayLike
    normalization: ArrayLike
    count: ArrayLike
    std: ArrayLike = None
    sem: ArrayLike = None
    norm_sq: ArrayLike = None


class ErrorModel(IntEnum):
    NO = 0
    VARIANCE = 1
    POISSON = 2
    AZIMUTHAL = 3
    HYBRID = 4  # used in peak-picking, use azimuthal for sigma-clipping and poisson later on, at the last iteration

    @classmethod
    def parse(cls, value):
        res = cls.NO
        if value is None:
            res = cls.NO
        elif isinstance(value, cls):
            res = value
        elif isinstance(value, str):
            for k, v in cls.__members__.items():
                if k.startswith(value.upper()):
                    res = v
                    break
        elif isinstance(value, int):
            res = cls(value)
        return res

    @property
    def poissonian(self):
        return self._value_ == 2 or self._value_ == 4

    @property
    def do_variance(self):
        return self._value_ != 0

    def as_str(self):
        return self.name.lower()


class _CopyableTuple(tuple):
    "Abstract class that can be copied using the copy module"

    COPYABLE_ATTR = set()  # list of copyable attributes

    def __copy__(self):
        "Helper function for copy.copy()"
        other = self.__class__(*self)
        for attr in self.COPYABLE_ATTR:
            setattr(other, attr, getattr(self, attr))
        return other

    def __deepcopy__(self, memo=None):
        "Helper function for copy.deepcopy()"
        if memo is None:
            memo = {}
        args = []
        for i in self:
            cpy = copy.deepcopy(i, memo)
            memo[id(i)] = cpy
            args.append(cpy)
        other = self.__class__(*args)
        for attr in self.COPYABLE_ATTR:
            org = getattr(self, attr)
            cpy = copy.deepcopy(org, memo)
            memo[id(org)] = cpy
            setattr(other, attr, cpy)
        return other


class IntegrateResult(_CopyableTuple):
    """
    Class defining shared information between Integrate1dResult and Integrate2dResult.
    """

    COPYABLE_ATTR = {
        "_sum_signal",
        "_sum_variance",
        "_sum_normalization",
        "_sum_normalization2",
        "_count",
        "_unit",
        "_has_mask_applied",
        "_has_dark_correction",
        "_has_flat_correction",
        "_has_solidangle_correction",
        "_normalization_factor",
        "_polarization_factor",
        "_metadata",
        "_npt_azim",
        "_percentile",
        "_method",
        "_method_called",
        "_compute_engine",
        "_error_model",
        "_std",
        "_sem",
        "_poni",
        "_weighted_average",
        "_dummy",
    }

    def __init__(self):
        self._sum_signal = None  # sum of signal
        self._sum_variance = None  # sum of variance
        self._sum_normalization = None  # sum of all normalization SA, pol, ...
        self._sum_normalization2 = None  # sum of all normalization squared
        self._count = None  # sum of counts, from signal/norm
        self._unit = None
        self._has_mask_applied = None
        self._has_dark_correction = None
        self._has_flat_correction = None
        self._has_solidangle_correction = None
        self._normalization_factor = None
        self._polarization_factor = None
        self._metadata = None
        self._npt_azim = None
        self._percentile = None
        self._method = None
        self._method_called = None
        self._compute_engine = None
        self._error_model = None
        self._dummy = numpy.nan  # value assigned to masked or empty pixels
        self._std = None  # standard deviation (error for a pixel)
        self._sem = None  # standard error of the mean (error for the mean)
        self._poni = None  # Contains the geometry which was used for the integration
        self._weighted_average = None  # Should be True for weighted average and False for unweighted (legacy)

    def __are_compatible__(self, other, strict:bool=True) -> None | str:
        """Ensure two objects are compatible to make some basic maths together.
        If compatible, return None, else return the reason for the incompatibility

        :param other: another instance of IntegrateResult
        :param strict: ensures also that normalizations are the same, used for basic math but not for union
        :return None if compatible, else the reason with a string.
        """
        if not isinstance(other, self.__class__):
            return f"class incompatible: {self.__class__.__name__} != {other.__class__.__name__}"
        if self.unit != other.unit:
            return f"unit differs: {self.unit} != {other.unit}"
        if strict and not numpy.allclose(self._sum_normalization, other.sum_normalization):
            return "normalization differs"

    def __add__(self, other):
        """Calculate the out-of-place addition of anther IntegrateResult

        :param other: the same datatype as self
        :return: another instance with updated values
        """
        reason = self.__are_compatible__(other)
        if reason:
            raise TypeError(f"Cannot add `IntegrateResult` because of {reason}")
        res = copy.deepcopy(self)
        res._sum_signal = self._sum_signal + other._sum_signal
        if self._sum_variance is None:
            res._sum_variance = other.sum_variance
        elif other.sum_variance is None:
            res._sum_variance = self._sum_variance
        else:
            res._sum_variance = self._sum_variance + other._sum_variance
        return res.__recalculate_means__()

    def __sub__(self, other):
        """Calculate the out-of-place subtraction of anther IntegrateResult

        :param other: the same datatype as self
        :return: another instance with updated values
        """
        reason = self.__are_compatible__(other)
        if reason:
            raise TypeError(f"Cannot subtract `IntegrateResult` because of {reason}")
        res = copy.deepcopy(self)
        res._sum_signal = self._sum_signal - other.sum_signal
        if self._sum_variance is None:
            res._sum_variance = other.sum_variance
        elif other.sum_variance is None:
            res._sum_variance = self._sum_variance
        else:
            res._sum_variance = self._sum_variance + other.sum_variance
        return res.__recalculate_means__()

    def __iadd__(self, other):
        """Calculate the inplace addition of anther IntegrateResult

        :param other: the same datatype as self
        :return: same instance with updated values
        """
        reason = self.__are_compatible__(other)
        if reason:
            raise TypeError(f"Cannot add `IntegrateResult` because of {reason}")
        self._sum_signal += other._sum_signal
        if self._sum_variance is None:
            self._sum_variance = other.sum_variance
        elif other._sum_variance is not None:
            self._sum_variance += other._sum_variance
        return self.__recalculate_means__()

    def __isub__(self, other):
        """Calculate the inplace subtraction of anther IntegrateResult

        :param other: the same datatype as self
        :return: same instance with updated values
        """
        reason = self.__are_compatible__(other)
        if reason:
            raise TypeError(f"Cannot subtract `IntegrateResult` because of {reason}")
        self._sum_signal -= other.sum_signal
        if self._sum_variance is None:
            self._sum_variance = other.sum_variance
        elif other._sum_variance is not None:
            self._sum_variance += other._sum_variance
        return self.__recalculate_means__()

    def __recalculate_means__(self):
        mask = (self.sum_normalization == 0.0)
        nomask = numpy.logical_not(mask)
        numpy.divide(self._sum_signal, self._sum_normalization,
                        out=self.intensity, where=nomask)
        self.intensity[mask] = self.dummy
        if self.sum_variance is not None:
            numpy.sqrt(self._sum_variance, out=self._sem)
            numpy.divide(self._sem, self.sum_normalization,
                         out=self._sem, where=nomask)
            self._sem[mask] = self.dummy

            self._std = numpy.divide(self._sum_variance, self._sum_normalization2,
                                    out=self._sem, where=nomask)
            self._std = numpy.sqrt(self._std, out=self._std)
            self._std[mask] = self.dummy
            self.sigma[...] = self._sem
        return self

    def union(self, other):
        """Calculate the weighted average of two results in a new IntegrateResult

        :param other: the same datatype as self
        :return: another instance of same datatype with the weighted average
        """
        reason = self.__are_compatible__(other, strict=False)
        reason = reason or None if self.error_model == other.error_model else f"differing error models: {self.error_model.name}!={other.error_model.name}"
        if reason:
            raise TypeError(f"Cannot add `IntegrateResult` because of {reason}")
        res = copy.deepcopy(self)
        res._sum_signal = self._sum_signal + other._sum_signal
        res._sum_normalization = self._sum_normalization + other._sum_normalization
        res._count = self._count + other._count
        if self._sum_variance is None or other._sum_variance is None:
            res._sum_variance = None
            res._sum_normalization2 = None
        else:
            res._sum_variance = self._sum_variance + other._sum_variance
            res._sum_normalization2 = self._sum_normalization2 + other._sum_normalization2
        return res.__recalculate_means__()

    @property
    def method(self):
        """return the name of the integration method _actually_ used,
        represented as a 4-tuple (dimension, splitting, algorithm, implementation)
        """
        return self._method

    def _set_method(self, value):
        self._method = value

    @property
    def method_called(self):
        "return the name of the method called"
        return self._method_called

    def _set_method_called(self, value):
        self._method_called = value

    @property
    def compute_engine(self):
        "return the name of the compute engine, like CSR"
        return self._compute_engine

    def _set_compute_engine(self, value):
        self._compute_engine = value

    @property
    def sum(self):
        """Sum of all signal

        :rtype: numpy.ndarray
        """
        return self._sum_signal

    def _set_sum(self, sum_):
        """Set the sum_signal information

        :type count: numpy.ndarray
        """
        self._sum_signal = sum_

    @property
    def sum_signal(self):
        """Sum_signal information

        :rtype: numpy.ndarray
        """
        return self._sum_signal

    def _set_sum_signal(self, sum_):
        """Set the sum_signal information

        :type count: numpy.ndarray
        """
        self._sum_signal = sum_

    @property
    def sum_variance(self):
        """Sum of all variances information

        :rtype: numpy.ndarray
        """
        return self._sum_variance

    def _set_sum_variance(self, sum_):
        """Set the sum of all variance information

        :type count: numpy.ndarray
        """
        self._sum_variance = sum_

    @property
    def sum_normalization(self):
        """Sum of all normalization information

        :rtype: numpy.ndarray
        """
        return self._sum_normalization

    def _set_sum_normalization(self, sum_):
        """Set the sum of all normalization information

        :type count: numpy.ndarray
        """
        self._sum_normalization = sum_

    @property
    def sum_normalization2(self):
        """Sum of all normalization squared information

        :rtype: numpy.ndarray
        """
        return self._sum_normalization2

    def _set_sum_normalization2(self, sum_):
        """Set the sum of all normalization information

        :type count: numpy.ndarray
        """
        self._sum_normalization2 = sum_

    @property
    def count(self):
        """Count information

        :rtype: numpy.ndarray
        """
        return self._count

    def _set_count(self, count):
        """Set the count information

        :type count: numpy.ndarray
        """
        self._count = count

    @property
    def unit(self):
        """Radial unit

        :rtype: string
        """
        return self._unit

    def _set_unit(self, unit):
        """Define the radial unit

        :type unit: str
        """
        self._unit = unit

    @property
    def has_mask_applied(self):
        """True if a mask was applied

        :rtype: bool
        """
        return self._has_mask_applied

    def _set_has_mask_applied(self, has_mask):
        """Define if dark correction was applied

        :type has_mask: bool (or string)
        """
        self._has_mask_applied = has_mask

    @property
    def has_dark_correction(self):
        """True if a dark correction was applied

        :rtype: bool
        """
        return self._has_dark_correction

    def _set_has_dark_correction(self, has_dark_correction):
        """Define if dark correction was applied

        :type has_dark_correction: bool
        """
        self._has_dark_correction = has_dark_correction

    @property
    def has_flat_correction(self):
        """True if a flat correction was applied

        :rtype: bool
        """
        return self._has_flat_correction

    def _set_has_flat_correction(self, has_flat_correction):
        """Define if flat correction was applied

        :type has_flat_correction: bool
        """
        self._has_flat_correction = has_flat_correction

    @property
    def has_solidangle_correction(self):
        """True if a flat correction was applied

        :rtype: bool
        """
        return self._has_solidangle_correction

    def _set_has_solidangle_correction(self, has_solidangle_correction):
        """Define if flat correction was applied

        :type has_solidangle_correction: bool
        """
        self._has_solidangle_correction = has_solidangle_correction

    @property
    def normalization_factor(self):
        """The normalisation factor used

        :rtype: float
        """
        return self._normalization_factor

    def _set_normalization_factor(self, normalization_factor):
        """Define the used normalisation factor

        :type normalization_factor: float
        """
        self._normalization_factor = normalization_factor

    @property
    def polarization_factor(self):
        """The polarization factor used

        :rtype: float
        """
        return self._polarization_factor

    def _set_polarization_factor(self, polarization_factor):
        """Define the used polarization factor

        :type polarization_factor: float
        """
        self._polarization_factor = polarization_factor

    @property
    def metadata(self):
        """Metadata associated with the input frame

        :rtype: JSON serializable dict object
        """
        return self._metadata

    def _set_metadata(self, metadata):
        """Define the metadata associated with the input frame

        :type metadata: JSON serializable dict object
        """
        self._metadata = metadata

    @property
    def percentile(self):
        "for median filter along the azimuth, position of the centile retrieved"
        return self._percentile

    def _set_percentile(self, value):
        self._percentile = value

    @property
    def npt_azim(self):
        "for median filter along the azimuth, number of azimuthal bin initially used"
        return self._npt_azim

    def _set_npt_azim(self, value):
        self._npt_azim = value

    def _set_std(self, value):
        self._std = value

    @property
    def std(self):
        return self._std

    def _set_sem(self, value):
        self._sem = value

    @property
    def sem(self):
        return self._sem

    @property
    def error_model(self):
        return self._error_model

    def _set_error_model(self, value):
        self._error_model = value

    @property
    def poni(self):
        "content of the PONI-file"
        return self._poni

    def _set_poni(self, value):
        self._poni = value

    @property
    def weighted_average(self):
        """Average have been done:
        * if True with the weighted mean (-ng)
        * if False with the unweighted mean (-legacy)
        """
        return self._weighted_average

    def _set_weighted_average(self, value):
        """Average have been done:
        * if True with the weighted mean (-ng)
        * if False with the unweighted mean (-legacy)
        """
        self._weighted_average = bool(value)

    @property
    def dummy(self):
        return self._dummy

    def _set_dummy(self, value):
        """Set the value for masked/invalid/empty bins"""
        self._dummy = value


class Integrate1dResult(IntegrateResult):
    """
    Result of an 1D integration. Provide a tuple access as a simple way to reach main attributes.
    Default result, extra results, and some integration parameters are available from attributes.

    For compatibility with older API, the object can be read as a tuple in different ways:

    .. code-block:: python

        result = ai.integrate1d(...)
        if result.sigma is None:
            radial, intensity = result
        else:
            radial, intensity, sigma = result
    """

    COPYABLE_ATTR = IntegrateResult.COPYABLE_ATTR.union({"_unit"})

    def __new__(self, radial, intensity, sigma=None):
        if sigma is None:
            t = radial, intensity
        else:
            t = radial, intensity, sigma
        return IntegrateResult.__new__(Integrate1dResult, t)

    def __init__(self, radial, intensity, sigma=None):
        super(Integrate1dResult, self).__init__()

    @property
    def radial(self):
        """
        Radial positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def intensity(self):
        """
        Regrouped intensity

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def sigma(self):
        """
        Error array if it was requested

        :rtype: numpy.ndarray, None
        """
        if len(self) == 2:
            return None
        return self[2]

    def __are_compatible__(self, other, strict:bool=True):
        """Ensure two objects are compatible to make some basic maths together"""
        reason = super().__are_compatible__(other, strict=strict)
        if reason:
            return reason
        if not numpy.allclose(self.radial, other.radial):
            return "radial differs"

    def calc_spottiness(self, weighted:bool=False) -> float:
        """Calculate the spottiness of a powder diffraction pattern:
        Inspired by doi:10.1107/S1600576713029713
        Requires the azimuthal error propagation.

        :param weighted: Weight the spottiness by the intensity of each ring.
        :return: a value that increases with the spottiness
        """
        if self.error_model != ErrorModel.AZIMUTHAL:
            logger.warning("Error model must be azimuthal to calculate spottiness !")

        intensity = numpy.maximum(0.0, self.intensity) if weighted else numpy.ones_like(self.intensity)
        sum_variance = numpy.zeros_like(self.sum_signal) if self.sum_variance is None else numpy.maximum(0.0, self.sum_variance)
        sum_signal = numpy.maximum(1.0, self.sum_signal)
        proportion = sum_variance/sum_signal**2
        return math.sqrt((proportion*intensity).sum(dtype=numpy.float64)/intensity.sum(dtype=numpy.float64))


class Integrate2dResult(IntegrateResult):
    """
    Result of an 2D integration. Provide a tuple access as a simple way to reach main attributes.
    Default result, extra results, and some integration parameters are available from attributes.

    For compatibility with older API, the object can be read as a tuple in different ways:

    .. code-block:: python

        result = ai.integrate2d(...)
        if result.sigma is None:
            intensity, radial, azimuthal = result
        else:
            intensity, radial, azimuthal, sigma = result
    """

    COPYABLE_ATTR = IntegrateResult.COPYABLE_ATTR.union(
        {"_radial_unit", "_azimuthal_unit"}
    )

    def __new__(self, intensity, radial, azimuthal, sigma=None):
        if sigma is None:
            t = intensity, radial, azimuthal
        else:
            t = intensity, radial, azimuthal, sigma
        return IntegrateResult.__new__(Integrate2dResult, t)

    def __init__(self, intensity, radial, azimuthal, sigma=None):
        super(Integrate2dResult, self).__init__()
        self._radial_unit = None
        self._azimuthal_unit = None

    def __are_compatible__(self, other, strict:bool=True):
        """Ensure two objects are compatible to make some basic maths together"""
        reason = super().__are_compatible__(other, strict=strict)
        if reason:
            return reason
        if not numpy.allclose(self.radial, other.radial):
            return "radial differs"
        if not numpy.allclose(self.azimuthal, other.azimuthal):
            return "azimuthal differs"

    @property
    def intensity(self):
        """
        Azimuthaly regrouped intensity

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def radial(self):
        """
        Radial positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def azimuthal(self):
        """
        Azimuthal positions (chi)

        :rtype: numpy.ndarray
        """
        return self[2]

    @property
    def sigma(self):
        """
        Error array if it was requested

        :rtype: numpy.ndarray, None
        """
        if len(self) == 3:
            return None
        return self[3]

    @property
    def unit(self):
        """Radial unit

        :rtype: Unit or 2-tuple of Unit
        """
        if self._azimuthal_unit is None:
            return self._radial_unit
        else:
            return self._radial_unit, self._azimuthal_unit

    def _set_unit(self, unit):
        """Define the radial unit

        :type unit: str
        """
        deprecated_warning(
            "Function",
            "_set_unit",
            replacement="_set_radial_unit/_set_azimuthal_unit",
            since_version="2023.09",
            only_once=True,
        )
        if isinstance(unit, (tuple, list)) and len(unit) == 2:
            self._radial_unit, self._azimuthal_unit = unit
        else:
            self._radial_unit = unit

    @property
    def radial_unit(self):
        """Radial unit

        :rtype: string
        """
        return self._radial_unit

    def _set_radial_unit(self, unit):
        """Define the radial unit

        :type unit: str
        """
        self._radial_unit = unit

    @property
    def azimuthal_unit(self):
        """Radial unit

        :rtype: string
        """
        return self._azimuthal_unit

    def _set_azimuthal_unit(self, unit):
        """Define the radial unit

        :type unit: str
        """
        self._azimuthal_unit = unit


    def rebin1d(self) -> Integrate1dResult:
        """Function that rebins an Integrate2dResult into a Integrate1dResult

        :return: Integrate1dResult
        """
        bins_rad = self.radial
        sum_signal = self.sum_signal.sum(axis=0)
        sum_normalization = self.sum_normalization.sum(axis=0)
        intensity = sum_signal / sum_normalization
        if self.sum_variance is not None:
            sum_variance = self.sum_variance.sum(axis=0)
            sem = numpy.sqrt(sum_variance) / sum_normalization
            result = Integrate1dResult(bins_rad, intensity, sem)
            result._set_sum_normalization2(self.sum_normalization2.sum(axis=0))
            result._set_sum_variance(sum_variance)
            result._set_std(numpy.sqrt(sum_variance / result.sum_normalization2))
            result._set_std(sem)
        else:
            result = Integrate1dResult(bins_rad, intensity)

        result._set_sum_signal(sum_signal)
        result._set_sum_normalization(sum_normalization)

        result._set_method_called("integrate1d")
        result._set_compute_engine(self.compute_engine)
        result._set_method(self.method)
        result._set_unit(self.radial_unit)
        # result._set_azimuthal_unit(self.azimuth_unit)
        result._set_count(self.count.sum(axis=0))
        # result._set_sum(sum_)
        result._set_has_dark_correction(self.has_dark_correction)
        result._set_has_flat_correction(self.has_flat_correction)
        result._set_has_mask_applied(self.has_mask_applied)
        result._set_polarization_factor(self.polarization_factor)
        result._set_normalization_factor(self.normalization_factor)
        result._set_metadata(self.metadata)
        return result


class SeparateResult(_CopyableTuple):
    """
    Class containing the result of AzimuthalIntegrator.separate which separates the

    * Amorphous isotropic signal (from a median filter or a sigma-clip)
    * Bragg peaks (signal > amorphous)
    * Shadow areas (signal < amorphous)
    """

    COPYABLE_ATTR = {
        "_radial",
        "_intensity",
        "_sigma",
        "_sum_signal",
        "_sum_variance",
        "_sum_normalization",
        "_count",
        "_unit",
        "_has_mask_applied",
        "_has_dark_correction",
        "_has_flat_correction",
        "_normalization_factor",
        "_polarization_factor",
        "_metadata",
        "_npt_rad",
        "_npt_azim",
        "_percentile",
        "_method",
        "_method_called",
        "_compute_engine",
        "_shadow",
    }

    def __new__(self, bragg, amorphous):
        return tuple.__new__(SeparateResult, (bragg, amorphous))

    def __init__(self, bragg, amorphous):
        # tuple.__init__(self, (bragg, amorphous))
        self._radial = None
        self._intensity = None
        self._sigma = None
        self._sum_signal = None  # sum of signal
        self._sum_variance = None  # sum of variance
        self._sum_normalization = None  # sum of all normalization SA, pol, ...
        self._count = None  # sum of counts, from signal/norm
        self._unit = None
        self._has_mask_applied = None
        self._has_dark_correction = None
        self._has_flat_correction = None
        self._normalization_factor = None
        self._polarization_factor = None
        self._metadata = None
        self._npt_rad = None
        self._npt_azim = None
        self._percentile = None
        self._method = None
        self._method_called = None
        self._compute_engine = None
        self._shadow = None

    @property
    def bragg(self):
        """
        Contains the bragg peaks

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def amorphous(self):
        """
        Contains the amorphous (isotropic) signal

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def shadow(self):
        """
        Contains the shadowed (weak) signal part

        :rtype: numpy.ndarray
        """
        return self._shadow

    @property
    def radial(self):
        """
        Radial positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self._radial

    @property
    def intensity(self):
        """
        Regrouped intensity

        :rtype: numpy.ndarray
        """
        return self._intensity

    @property
    def sigma(self):
        """
        Error array if it was requested

        :rtype: numpy.ndarray, None
        """
        return self._sigma

    @property
    def method(self):
        """return the name of the integration method _actually_ used,
        represented as a 4-tuple (dimension, splitting, algorithm, implementation)
        """
        return self._method

    def _set_method(self, value):
        self._method = value

    @property
    def method_called(self):
        "return the name of the method called"
        return self._method_called

    def _set_method_called(self, value):
        self._method_called = value

    @property
    def compute_engine(self):
        "return the name of the compute engine, like CSR"
        return self._compute_engine

    def _set_compute_engine(self, value):
        self._compute_engine = value

    @property
    def sum(self):
        """Sum of all signal

        :rtype: numpy.ndarray
        """
        return self._sum_signal

    def _set_sum(self, sum_):
        """Set the sum_signal information

        :type count: numpy.ndarray
        """
        self._sum_signal = sum_

    @property
    def sum_signal(self):
        """Sum_signal information

        :rtype: numpy.ndarray
        """
        return self._sum_signal

    def _set_sum_signal(self, sum_):
        """Set the sum_signal information

        :type count: numpy.ndarray
        """
        self._sum_signal = sum_

    @property
    def sum_variance(self):
        """Sum of all variances information

        :rtype: numpy.ndarray
        """
        return self._sum_variance

    def _set_sum_variance(self, sum_):
        """Set the sum of all variance information

        :type count: numpy.ndarray
        """
        self._sum_variance = sum_

    @property
    def sum_normalization(self):
        """Sum of all normalization information

        :rtype: numpy.ndarray
        """
        return self._sum_normalization

    def _set_sum_normalization(self, sum_):
        """Set the sum of all normalization information

        :type count: numpy.ndarray
        """
        self._sum_normalization = sum_

    @property
    def count(self):
        """Count information

        :rtype: numpy.ndarray
        """
        return self._count

    def _set_count(self, count):
        """Set the count information

        :type count: numpy.ndarray
        """
        self._count = count

    @property
    def unit(self):
        """Radial unit

        :rtype: string
        """
        return self._unit

    def _set_unit(self, unit):
        """Define the radial unit

        :type unit: str
        """
        self._unit = unit

    @property
    def has_mask_applied(self):
        """True if a mask was applied

        :rtype: bool
        """
        return self._has_mask_applied

    def _set_has_mask_applied(self, has_mask):
        """Define if dark correction was applied

        :type has_mask: bool (or string)
        """
        self._has_mask_applied = has_mask

    @property
    def has_dark_correction(self):
        """True if a dark correction was applied

        :rtype: bool
        """
        return self._has_dark_correction

    def _set_has_dark_correction(self, has_dark_correction):
        """Define if dark correction was applied

        :type has_dark_correction: bool
        """
        self._has_dark_correction = has_dark_correction

    @property
    def has_flat_correction(self):
        """True if a flat correction was applied

        :rtype: bool
        """
        return self._has_flat_correction

    def _set_has_flat_correction(self, has_flat_correction):
        """Define if flat correction was applied

        :type has_flat_correction: bool
        """
        self._has_flat_correction = has_flat_correction

    @property
    def normalization_factor(self):
        """The normalisation factor used

        :rtype: float
        """
        return self._normalization_factor

    def _set_normalization_factor(self, normalization_factor):
        """Define the used normalisation factor

        :type normalization_factor: float
        """
        self._normalization_factor = normalization_factor

    @property
    def polarization_factor(self):
        """The polarization factor used

        :rtype: float
        """
        return self._polarization_factor

    def _set_polarization_factor(self, polarization_factor):
        """Define the used polarization factor

        :type polarization_factor: float
        """
        self._polarization_factor = polarization_factor

    @property
    def metadata(self):
        """Metadata associated with the input frame

        :rtype: JSON serializable dict object
        """
        return self._metadata

    def _set_metadata(self, metadata):
        """Define the metadata associated with the input frame

        :type metadata: JSON serializable dict object
        """
        self._metadata = metadata

    @property
    def percentile(self):
        "for median filter along the azimuth, position of the centile retrieved"
        return self._percentile

    def _set_percentile(self, value):
        self._percentile = value

    @property
    def npt_azim(self):
        "for median filter along the azimuth, number of azimuthal bin initially used"
        return self._npt_azim

    def _set_npt_azim(self, value):
        self._npt_azim = value


class SparseFrame(_CopyableTuple):
    """Result of the sparsification of a diffraction frame"""

    COPYABLE_ATTR = {
        "_shape",
        "_dtype",
        "_mask",
        "_radius",
        "_dummy",
        "_background_avg",
        "_background_std",
        "_unit",
        "_has_dark_correction",
        "_has_flat_correction",
        "_normalization_factor",
        "_polarization_factor",
        "_metadata",
        "_percentile",
        "_method",
        "_method_called",
        "_compute_engine",
        "_cutoff_clip",
        "_cutoff_pick",
        "_cutoff_peak",
        "_background_cycle",
        "_noise",
        "_radial_range",
        "_error_model",
        "_peaks",
        "_peak_patch_size",
        "_peak_connected",
    }

    def __new__(self, index, intensity):
        return tuple.__new__(SparseFrame, (index, intensity))

    def __init__(self, index, intensity):
        self._shape = None
        self._dtype = None
        self._mask = None
        self._dummy = None
        self._radius = None
        self._background_avg = None
        self._background_std = None
        self._unit = None
        self._has_dark_correction = None
        self._has_flat_correction = None
        self._normalization_factor = None
        self._polarization_factor = None
        self._metadata = None
        self._percentile = None
        self._method = None
        self._method_called = None
        self._compute_engine = None
        self._cutoff_clip = None
        self._cutoff_pick = None
        self._cutoff_peak = None
        self._background_cycle = None
        self._noise = None
        self._radial_range = None
        self._error_model = None
        self._peaks = None
        self._peak_patch_size = None
        self._peak_connected = None

    @property
    def index(self):
        """
        Contains the index position of bragg peaks

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def intensity(self):
        """
        Contains the intensity of bragg peaks

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def mask(self):
        """
        Contains the mask used (encodes for the shape of the image as well)

        :rtype: numpy.ndarray
        """
        return self._mask

    @property
    def x(self):
        if self._shape is None:
            return self[0]
        else:
            return self[0] % self._shape[-1]

    @property
    def y(self):
        if self._shape is None:
            return 0
        else:
            return self[0] // self._shape[-1]

    @property
    def noise(self):
        return self._noise

    @property
    def radius(self):
        return self._radius

    @property
    def background_avg(self):
        return self._background_avg

    @property
    def background_std(self):
        return self._background_std

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dummy(self):
        return self._dummy

    @property
    def peaks(self):
        return self._peaks

    @property
    def cutoff_clip(self):
        return self._cutoff_clip

    @property
    def cutoff_pick(self):
        return self._cutoff_pick

    cutoff = cutoff_pick

    @property
    def cutoff_peak(self):
        return self._cutoff_peak

    @property
    def error_model(self):
        return self._error_model

    @property
    def peak_patch_size(self):
        return self._peak_patch_size

    @property
    def peak_connected(self):
        return self._peak_connected

    @property
    def unit(self):
        return self._unit


def rebin1d(res2d: Integrate2dResult) -> Integrate1dResult:
    """Function that rebins an Integrate2dResult into a Integrate1dResult

    :param res2d: Integrate2dResult instance obtained from ai.integrate2d
    :return: Integrate1dResult
    """
    deprecated_warning(type_='Function',
                       name="rebin1d",
                       reason="use method `rebin1d` of Integrate2dResult",
                       since_version="2025.11",
                       only_once=True)
    return res2d.rebin1d()


def symmetrize(res2d: Integrate2dResult) -> Integrate2dResult:
    """Function that symmetrize an Integrate2dResult, i.e. merge data with those 180° apart in azimuthal space

    :param res2d: Integrate2dResult instance obtained from ai.integrate2d
    :return: Integrate1dResult
    """

    def _symmetrize_single_array(ary):
        half = ary.shape[0] // 2
        tmp = numpy.zeros_like(ary)
        tmp[:half] = ary[half:]
        tmp[half:] = ary[:half]
        tmp += ary
        return tmp

    bins_rad = res2d.radial
    bins_azim = res2d.azimuthal
    delta = (bins_azim.max() - bins_azim.min()) / res2d.azimuthal_unit.scale
    if delta < 6.2:
        logger.error("Expected an azimuthal integration over 2π ! Expect wrong results")
    if bins_azim.size % 2:
        logger.error(
            "Expected an azimuthal integration an even number of bins azimuthally! Expect wrong results"
        )

    sum_signal = _symmetrize_single_array(res2d.sum_signal)
    sum_normalization = _symmetrize_single_array(res2d.sum_normalization)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        intensity = sum_signal / sum_normalization

    if res2d.sum_variance is not None:
        sum_variance = _symmetrize_single_array(res2d.sum_variance)
        sum_normalization2 = _symmetrize_single_array(res2d.sum_normalization2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sem = numpy.sqrt(sum_variance) / sum_normalization
            std = numpy.sqrt(sum_variance / sum_normalization2)
        mask = sum_normalization2 <= 0.0
        sem[mask] = res2d.dummy
        std[mask] = res2d.dummy

        result = Integrate2dResult(intensity, bins_rad, bins_azim, sem)
        result._set_sum_normalization2(sum_normalization2)
        result._set_sum_variance(sum_variance)
    else:
        result = Integrate2dResult(intensity, bins_rad, bins_azim)
        mask = sum_normalization <= 0.0

    intensity[mask] = res2d.dummy
    result._set_sum_signal(sum_signal)
    result._set_sum_normalization(sum_normalization)
    result._set_method_called(f"{res2d.method_called}|symmetrize")
    result._set_compute_engine(res2d.compute_engine)
    result._set_method(res2d.method)
    result._set_radial_unit(res2d.radial_unit)
    result._set_count(_symmetrize_single_array(res2d.count))
    result._set_azimuthal_unit(res2d.azimuthal_unit)
    result._set_has_dark_correction(res2d.has_dark_correction)
    result._set_has_flat_correction(res2d.has_flat_correction)
    result._set_has_mask_applied(res2d.has_mask_applied)
    result._set_polarization_factor(res2d.polarization_factor)
    result._set_normalization_factor(res2d.normalization_factor)
    result._set_metadata(res2d.metadata)
    result._set_dummy(res2d.dummy)
    return result


class Integrate1dFiberResult(IntegrateResult):
    def __new__(self, integrated, intensity, sigma=None):
        if sigma is None:
            t = integrated, intensity
        else:
            t = integrated, intensity, sigma
        return IntegrateResult.__new__(Integrate1dFiberResult, t)

    def __init__(self, integrated, intensity, sigma=None):
        super(Integrate1dFiberResult, self).__init__()
        self._vertical_integration = None

    @property
    def integrated(self):
        """
        Integrated positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def radial(self):
        logger.warning(
            "radial does not apply to a fiber/grazing-incidence result, use integrated instead"
        )
        return self[0]

    @property
    def intensity(self):
        """
        Regrouped intensity

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def sigma(self):
        """
        Error array if it was requested

        :rtype: numpy.ndarray, None
        """
        if len(self) == 2:
            return None
        return self[2]

    @property
    def vertical_integration(self):
        """Vertical integration

        :rtype: bool
        """
        if self._vertical_integration is None:
            logger.warning("Direction of 1D integration is unknown.")
        return self._vertical_integration

    def _set_vertical_integration(self, vertical_integration):
        """Define the direction of the 1D fiber/grazing-incidence integration

        :type unit: bool
        """
        self._vertical_integration = vertical_integration

class Integrate2dFiberResult(IntegrateResult):
    """
    Result of an 2D integration for fiber/grazing-incidence scattering.
    Provide a tuple access as a simple way to reach main attributes.
    Default result, extra results, and some integration parameters are available from attributes.
    Analog to azimuthal integrate containers but: Radial -> in-plane, Azimuthal -> out-of-plane
    """

    def __new__(self, intensity, inplane, outofplane, sigma=None):
        if sigma is None:
            t = intensity, inplane, outofplane
        else:
            t = intensity, inplane, outofplane, sigma
        return IntegrateResult.__new__(Integrate2dFiberResult, t)

    def __init__(self, intensity, inplane, outofplane, sigma=None):
        super(Integrate2dFiberResult, self).__init__()
        self._oop_unit = None
        self._ip_unit = None

    @property
    def intensity(self):
        """
        Regrouped intensity

        :rtype: numpy.ndarray
        """
        return self[0]

    @property
    def inplane(self):
        """
        In-plane positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self[1]

    @property
    def outofplane(self):
        """
        Out-of-plane positions (q/2theta/r)

        :rtype: numpy.ndarray
        """
        return self[2]

    @property
    def sigma(self):
        """
        Error array if it was requested

        :rtype: numpy.ndarray, None
        """
        if len(self) == 3:
            return None
        return self[3]

    @property
    def unit(self):
        """
        :rtype: 2-tuple of Unit
        """
        return self._ip_unit, self._oop_unit

    @property
    def radial(self):
        logger.warning(
            "Radial does not apply to a fiber/grazing-incidence result, use inplane instead"
        )
        return self.inplane

    @property
    def azimuthal(self):
        logger.warning(
            "Azimuthal does not apply to a fiber/grazing-incidence result, use outofplane instead"
        )
        return self.outofplane

    @property
    def ip_unit(self):
        """In-plane scattering unit

        :rtype: string
        """
        return self._ip_unit

    def _set_ip_unit(self, unit):
        """Define the in-plane scattering unit

        :type unit: str
        """
        self._ip_unit = unit

    def _set_radial_unit(self, unit):
        logger.warning(
            "Radial units does not apply to a fiber/grazing-incidence result, use ip_unit instead"
        )
        self._set_ip_unit(unit)

    @property
    def oop_unit(self):
        """Out-of-plane scattering unit

        :rtype: string
        """
        return self._oop_unit

    def _set_oop_unit(self, unit):
        """Define the out-of-plane scattering unit

        :type unit: str
        """
        self._oop_unit = unit

    def _set_azimuthal_unit(self, unit):
        logger.warning(
            "Azimuthal units does not apply to a fiber/grazing-incidence result, use oop_unit instead"
        )
        self._set_oop_unit(unit)


class Miller(NamedTuple):
    """This represents the Miller index of a family of lattice plans"""

    h: int  # noqa: E741
    k: int  # noqa: E741
    l: int  # noqa: E741

    def __repr__(self):
        return f"Miller({self.h}, {self.k}, {self.l})"

    def __str__(self):
        return f"({self.h} {self.k} {self.l})"

    @classmethod
    def parse(cls, text: str):
        words = text.split(" ")
        newlst = []
        for word in words:
            newlst.extend(word.split(","))
        words = newlst
        newlst = []
        for word in words:
            newlst.extend(word.split(";"))
        words = newlst
        ints = []
        for word in words:
            stripped = word.strip(" ()")
            if not stripped:
                continue
            try:
                value = int(stripped)
            except ValueError:
                logger.warning(f"Unable to parse int in {stripped}")
            else:
                ints.append(value)
        if len(ints) != 3:
            raise RuntimeError(
                f"Miller indices expects exactly 3 integers to define a family of plans, got `{text}`."
            )
        return cls(*ints)


@dataclass
class Reflection:
    "Represent a family of Miller plans"

    dspacing: float = None
    intensity: float = None
    hkl: tuple = tuple()  # or better Miller namedtuple
    multiplicity: int = None

    @property
    def is_weak(self):
        """Return True if the intensity is weak"""
        if self.intensity is not None:
            return self.intensity == 0.0


class FixedParameters(set):
    """
    Like a set, made for FixedParameters in geometry refinement
    """

    def add_or_discard(self, key, value=True):
        """
        Add a value to a set if value, else discard it.

        :param key: element to add or discard from set
        :type value: boolean. If None do nothing !
        :return: None
        """
        if value is None:
            return
        if value:
            self.add(key)
        else:
            self.discard(key)

    def __repr__(self):
        return f"Fixed parameters: {', '.join(self)}."

    def __iadd__(self, other):
        for i in other:
            self.add(i)
        return self

    def __add__(self, other):
        """Enables the addition of a list"""
        new = self.__class__(self)
        new.__iadd__(other)
        return new
