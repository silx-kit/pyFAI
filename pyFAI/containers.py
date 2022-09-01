#!/usr/bin/env python3
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2020 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Valentin Valls"
__contact__ = "valentin.valls@esrf.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/09/2022"
__status__ = "development"

from collections import namedtuple
from enum import IntEnum

Integrate1dtpl = namedtuple("Integrate1dtpl", "position intensity sigma signal variance normalization count std sem norm_sq", defaults=(None,) * 3)
Integrate2dtpl = namedtuple("Integrate2dtpl", "radial azimuthal intensity sigma signal variance normalization count std sem norm_sq", defaults=(None,) * 3)


class ErrorModel(IntEnum):
    NO = 0
    VARIANCE = 1
    POISSON = 2
    AZIMUTHAL = 3
    HYBRID = 4  # used in peak-pickin, use azimuthal for sigma-clipping and poisson later on

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
        return self._value_ == 2

    @property
    def do_variance(self):
        return self._value_ != 0


class IntegrateResult(tuple):
    """
    Class defining shared information between Integrate1dResult and Integrate2dResult.
    """

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
        self._std = None  # standard deviation (error for a pixel)
        self._sem = None  # standard error of the mean (error for the mean)
        self._poni = None  # Contains the geometry which was used for the integration

    @property
    def method(self):
        """return the name of the integration method _actually_ used, 
        represented as a 4-tuple (dimention, splitting, algorithm, implementation)
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
        """content of the PONI-file
        """
        return self._poni

    def _set_poni(self, value):
        self._poni = value


class Integrate1dResult(IntegrateResult):
    """
    Result of an 1D integration. Provide a tuple access as a simple way to reach main attrbutes.
    Default result, extra results, and some interagtion parameters are available from attributes.

    For compatibility with older API, the object can be read as a tuple in different ways:

    .. code-block:: python

        result = ai.integrate1d(...)
        if result.sigma is None:
            radial, I = result
        else:
            radial, I, sigma = result
    """

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


class Integrate2dResult(IntegrateResult):
    """
    Result of an 2D integration. Provide a tuple access as a simple way to reach main attrbutes.
    Default result, extra results, and some interagtion parameters are available from attributes.

    For compatibility with older API, the object can be read as a tuple in different ways:

    .. code-block:: python

        result = ai.integrate2d(...)
        if result.sigma is None:
            I, radial, azimuthal = result
        else:
            I, radial, azimuthal, sigma = result
    """

    def __new__(self, intensity, radial, azimuthal, sigma=None):
        if sigma is None:
            t = intensity, radial, azimuthal
        else:
            t = intensity, radial, azimuthal, sigma
        return IntegrateResult.__new__(Integrate2dResult, t)

    def __init__(self, intensity, radial, azimuthal, sigma=None):
        super(Integrate2dResult, self).__init__()

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


class SeparateResult(tuple):
    """
    Class containing the result of AzimuthalIntegrator.separte which separates the
    
    * Amorphous isotropic signal (from a median filter or a sigma-clip)
    * Bragg peaks (signal > amorphous)
    * Shadow areas (signal < amorphous)  
    """

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
        represented as a 4-tuple (dimention, splitting, algorithm, implementation)
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


class SparseFrame(tuple):
    """Result of the sparsification of a diffraction frame"""

    def __new__(self, index, intensity):
        return tuple.__new__(SparseFrame, (index, intensity))

    def __init__(self, index, intensity):
        self._shape = None
        self._dtype = None
        self._mask = None
        self._dummy = None
        self._radial = None
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
