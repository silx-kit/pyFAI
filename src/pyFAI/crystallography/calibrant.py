#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Calibrant

A module containing classical calibrant class

"""

from __future__ import annotations

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/07/2025"
__status__ = "production"

import os
import logging
import numpy
import itertools
from typing import Optional, List
from collections.abc import Iterable
from math import sin, asin, cos, sqrt, pi, ceil, log, isfinite
import threading
from ..utils import get_calibration_dir
from ..utils.decorators import deprecated
from .. import units
from .cell import Cell
from .space_groups import ReflectionCondition
from .resolution import _ResolutionFunction, Caglioti, Constant
from ..containers import Integrate1dResult

logger = logging.getLogger(__name__)
epsilon = 1.0e-6  # for floating point comparison


class Calibrant:
    """
    A calibrant is a named reference compound where the d-spacing are known.

    The d-spacing (interplanar distances) are expressed in Angstrom (in the file).

    If the access is don't from a file, the IO are delayed. If it is not desired
    one could explicitly access to :meth:`load_file`.

    .. code-block:: python

        c = Calibrant()
        c.load_file("my_calibrant.D")

    :param filename: A filename containing the description (usually with .D extension).
                     The access to the file description is delayed until the information
                     is needed.
    :param dspacing: A list of d spacing in Angstrom.
    :param wavelength: A wavelength in meter
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        dspacing: Optional[List[float]] = None,
        wavelength: Optional[float] = None,
        **kwargs):
        if "dSpacing" in kwargs:
            dspacing = kwargs["dSpacing"]
            logger.warning("Usage of `dSpacing` keyword argument in `Calibrant` constructor is deprecated and has been replaced with `dspacing` (PEP8) since 2025.07")
        self._filename = filename
        self._wavelength = wavelength
        self._sem = threading.Semaphore()
        self._2th = []
        if filename is not None:
            self._dspacing = None
        elif dspacing is None:
            self._dspacing = []
        else:
            self._dspacing = list(dspacing)
        self._out_dspacing = []
        if self._dspacing and self._wavelength:
            self._calc_2th()

    @classmethod
    def from_cell(cls, cell):
        """Alternative constructor from a cell-object

        :param cell: Instance of Cell
        :return: Calibrant instance
        """
        return cell.to_calibrant()

    def __eq__(self, other: object) -> bool:
        """
        Test the equality with another object

        It only takes into account the wavelength and dspacing, not the
        filename. dspacing are lazy-loaded when needed.

        :param other: Another calibrant
        """
        if other is None:
            return False
        if not isinstance(other, Calibrant):
            return False
        if self._wavelength != other._wavelength:
            return False
        if self.dspacing != other.dspacing:  # enforce lazy-loading
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        Test the non-equality with another object

        It only takes into account the wavelength and dspacing, not the
        filename.

        :param other: Another object
        """
        return not (self == other)

    def __hash__(self) -> int:
        """
        Returns the hash of the object.

        It only takes into account the wavelength and dspacing, not the
        filename.
        """
        h = hash(self._wavelength)
        for d in self.dspacing:
            h = h ^ hash(d)
        return h

    def __copy__(self) -> Calibrant:
        """
        Copy a calibrant.
        """
        self._initialize()
        return Calibrant(
            filename=self._filename,
            dspacing=self._dspacing + self._out_dspacing,
            wavelength=self._wavelength,
        )

    def __repr__(self) -> str:
        if self._filename:
            name = self._filename
            if name.startswith("pyfai:"):
                name = name[6:]
        else:
            name = "undefined"
        name += " Calibrant "
        if len(self.dspacing):
            name += "with %i reflections " % len(self._dspacing)
        if self._wavelength:
            name += "at wavelength %s" % self._wavelength
        return name

    @property
    def name(self) -> str:
        """Returns a short name describing the calibrant.

        It's the name of the file or the resource.
        """
        f = self._filename
        if f is None:
            return "Undefined"
        if f.startswith("pyfai:"):
            return f[6:]
        return os.path.splitext(os.path.basename(f))[0]

    def get_filename(self) -> str:
        return self._filename

    filename = property(get_filename)

    def load_file(self, filename: str):
        """
        Load a calibrant.from file.

        :param filename: The filename containing the calibrant description.
        """
        with self._sem:
            self._load_file(filename)

    @staticmethod
    def _get_abs_path(filename: str) -> str:
        """Returns the absolute location of the calibrant."""
        if filename is None:
            return None
        if filename.startswith("pyfai:"):
            name = filename[6:]
            basedir = get_calibration_dir()
            return os.path.join(basedir, f"{name}.D")
        return os.path.abspath(filename)

    def _load_file(self, filename: Optional[str] = None):
        if filename:
            self._filename = filename

        path = self._get_abs_path(self._filename)
        if not os.path.isfile(path):
            logger.error("No such calibrant file: %s", path)
            return
        self._dspacing = numpy.unique(numpy.loadtxt(path))
        self._dspacing = list(self._dspacing[-1::-1])  # reverse order
        # self._dspacing.sort(reverse=True)
        if self._wavelength:
            self._calc_2th()

    def _initialize(self):
        """Initialize the object if expected."""
        if self._dspacing is None:
            if self._filename:
                self._load_file()
            else:
                self._dspacing = []

    def count_registered_dspacing(self) -> int:
        """Count of registered dspacing positions."""
        self._initialize()
        return len(self._dspacing) + len(self._out_dspacing)

    count_registered_dSpacing = deprecated(count_registered_dspacing,
                                           reason="PEP8",
                                           replacement="count_registered_dspacing",
                                           since_version="2025.07")

    def save_dspacing(self, filename: Optional[str] = None):
        """
        Save the d-spacing into a file.

        :param filename: name of the file
        :return: None
        """
        self._initialize()
        if (filename is None) and (self._filename is not None):
            if self._filename.startswith("pyfai:"):
                raise ValueError(
                    "A calibrant resource from pyFAI can't be overwritten)"
                )
            filename = self._filename
        else:
            return
        with open(filename) as f:
            f.write("# %s Calibrant" % filename)
            for i in self.dSpacing:
                f.write("%s\n" % i)

    save_dSpacing = deprecated(save_dspacing,
                            reason="PEP8",
                            replacement="save_dspacing",
                            since_version="2025.07")  # PEP8

    @property
    def dspacing(self) -> List[float]:
        self._initialize()
        return self._dspacing

    @dspacing.setter
    def dspacing(self, lst: List[float]):
        self._dspacing = list(lst)
        self._out_dspacing = []
        self._filename = "Modified"
        if self._wavelength:
            self._calc_2th()

    def append_dspacing(self, value: float):
        """Insert a d position at the right position of the dspacing list"""
        self._initialize()
        with self._sem:
            delta = [abs(value - v) / v for v in self._dspacing if v is not None]
            if not delta or min(delta) > epsilon:
                self._dspacing.append(value)
                self._dspacing.sort(reverse=True)
                self._calc_2th()
    append_dSpacing = deprecated(append_dspacing,
                                reason="PEP8",
                                replacement="append_dspacing",
                                since_version="2025.07")  # PEP8

    def append_2th(self, value: float):
        """Insert a 2th position at the right position of the dSpacing list"""
        with self._sem:
            self._initialize()
            if value not in self._2th:
                self._2th.append(value)
                self._2th.sort()
                self._calc_dSpacing()

    def setWavelength_change2th(self, value: Optional[float] = None):
        """
        Set a new wavelength.
        """
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning(
                        "This is an unlikely wavelength (in meter): %s",
                        self._wavelength,
                    )
                self._calc_2th()

    def setWavelength_changeDs(self, value: Optional[float] = None):
        """
        Set a new wavelength and only update the dSpacing list.

        This is probably not a good idea, but who knows!
        """
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning(
                        "This is an unlikely wavelength (in meter): %s",
                        self._wavelength,
                    )
                self._calc_dSpacing()

    @property
    def wavelength(self) -> Optional[float]:
        """
        Returns the used wavelength.
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: Optional[float] = None):
        """
        Set a new wavelength .
        """
        updated = False
        with self._sem:
            if self._wavelength is None:
                if value:
                    self._wavelength = float(value)
                    if (self._wavelength < 1e-15) or (self._wavelength > 1e-6):
                        logger.warning(
                            "This is an unlikely wavelength (in meter): %s",
                            self._wavelength,
                        )
                    updated = True
            elif abs(self._wavelength - value) / self._wavelength > epsilon:
                logger.warning(
                    "Forbidden to change the wavelength once it is fixed !!!!"
                )
                logger.warning(
                    "%s != %s, delta= %s",
                    self._wavelength,
                    value,
                    self._wavelength - value,
                )
        if updated:
            self._calc_2th()

    set_wavelength = deprecated(wavelength.fset,
                                reason="use property",
                                since_version="2025.07")

    def _calc_2th(self):
        """Calculate the 2theta positions for all peaks"""
        self._initialize()
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        tths = []
        dSpacing = self._dspacing[:] + self._out_dspacing  # explicit copy
        try:
            for ds in dSpacing:
                tth = 2.0 * asin(5.0e9 * self._wavelength / ds)
                tths.append(tth)
        except ValueError:
            size = len(tths)
            # remove dspacing outside of 0..180
            self._dspacing = dSpacing[:size]
            self._out_dspacing = dSpacing[size:]
        else:
            self._dspacing = dSpacing
            self._out_dspacing = []
        self._2th = tths

    def _calc_dspacing(self):
        """Replace the dspacing values by those calculated from the 2theta array"""
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._dspacing = [
            5.0e9 * self._wavelength / sin(tth / 2.0) for tth in self._2th
        ]
    _calc_dSpacing = deprecated(_calc_dspacing,
                                reason="PEP8",
                                since_version="2025.07")

    def get_2th(self) -> List[float]:
        """Returns the 2theta positions for all peaks (cached)"""
        if not self._2th:
            self._initialize()
            if not self._dspacing:
                logger.error("Not d-spacing for calibrant: %s", self)
            with self._sem:
                if not self._2th:
                    self._calc_2th()
        return self._2th

    def get_2th_index(self, angle: float, delta: Optional[float] = None) -> int:
        """Returns the index in the 2theta angle index.

        :param angle: expected angle in radians
        :param delta: precision on angle
        :return: 0-based index or None
        """
        if angle in self._2th:
            return self._2th.index(angle)
        if delta:
            d2th = abs(numpy.array(self._2th) - angle)
            i = d2th.argmin()
            if d2th[i] < delta:
                return i
        return None

    def get_max_wavelength(self, index: Optional[int] = None):
        """Calculate the maximum wavelength assuming the ring at index is visible.

        Bragg's law says: $\\lambda = 2d sin(\\theta)$
        So at 180° $\\lambda = 2d$

        :param index: Ring number, otherwise assumes all rings are visible
        :return: the maximum visible wavelength
        """
        dspacing = self._dspacing[:] + self._out_dspacing  # get all rings
        if index is None:
            index = len(dspacing) - 1
        if index >= len(dspacing):
            raise IndexError(
                "There are not than many (%s) rings indices in this calibrant" % (index)
            )
        return dspacing[index] * 2e-10

    def get_peaks(self, unit: units.Units|str = units.TTH_DEG):
        """Calculate the peak position as this unit.

        :return: numpy array (unlike other methods which return lists)
        """
        unit = units.to_unit(unit)
        scale = unit.scale
        name = unit.name
        size = len(self.get_2th())
        if name.startswith("2th"):
            values = numpy.array(self.get_2th())
        elif name.startswith("q"):
            values = 20.0 * pi / numpy.array(self.dspacing[:size])
        else:
            raise ValueError("Only *2theta* and *q* units are supported for now")

        return values * scale

    def fake_xrpdp(self,
                   nbpt: int=1000,
                   tth_range: tuple = (0,120),
                   background: float = 0.0,
                   Imax: float=1.0,
                   resolution: float = 0.1,
                   unit: units.Unit|str = units.TTH_DEG,
                    ):
        """Generate a fake powder diffraction pattern from this calibrant

        :param nbpt: number of point in the powder pattern
        :param tth_range: diffraction angle 2theta, unit as specified in unit parameter, deg by default.
        :param background: value or array (gonna be interpolated)
        :param Imax: intensity of the scattering signal
        :param resolution: pic width δ(°) or resolution function
        :param unit: can be a string or an instance
        :return: Integrate1dResult with unit in 2th_deg
        """
        unit = units.to_unit(unit)
        if unit.space != "2th":
            raise RuntimeError("XRPD have to be generated in `2theta` space")

        tth_range_min = min(tth_range)
        tth_range_max = max(tth_range)
        tth_user = numpy.linspace(tth_range_min, tth_range_max, nbpt)
        tth_rad = tth_user / unit.scale

        # background can be an array
        if isinstance(background, Iterable):
            background = numpy.interp(tth_deg,
                numpy.linspace(tth_range_min, tth_range_max, len(background)),
                background)
        else:  # or a constant
            background = numpy.zeros_like(tth_user) + background

        tth_peak = numpy.array(self.get_2th())
        if not isinstance(resolution, _ResolutionFunction):
            resolution = Constant(resolution, unit=unit)
        dtth2_peaks = resolution.sigma2(tth_peak)

        tth_min = tth_range_min / unit.scale
        tth_max = tth_range_max / unit.scale

        # TODO: deal with peak intensities ... after #2562 is addressed
        intensities = numpy.ones_like(tth_peak)

        # Normalization of peak intensities
        intensities /= numpy.nanmax(intensities)

        signal = numpy.zeros_like(tth_rad)
        for peak, sigma2, intensity in zip(tth_peak, dtth2_peaks, intensities):
            if not isfinite(intensity):
                continue
            sigma = sqrt(sigma2)
            if peak < (tth_min - 3 * sigma):
                continue
            elif peak > (tth_max + 3 * sigma):
                break

            # Gaussian profile
            signal += intensity * numpy.exp(-(tth_rad - peak) ** 2 / (2.0 * sigma2))

        signal *= Imax
        signal += background
        result = Integrate1dResult(tth_user, signal)
        result._set_unit(unit)
        return result

    def fake_calibration_image(
        self,
        ai,
        shape: tuple|None = None,
        Imax: float =1.0,
        Imin : float|numpy.ndarray=0.0,
        resolution: _ResolutionFunction|float =0.1,
        **kwargs) -> numpy.ndarray:
        """
        Generates a fake calibration image from an azimuthal integrator.

        :param ai: azimuthal integrator
        :param Imax: maximum intensity of rings
        :param Imin: minimum intensity of the signal (background)
        :param resolution: either the FWHM (static, in degree) or a `pyFAI.crystallography.resolution._ResolutionFunction` class instance
        Deprecated options:
        :param U, V, W: width of the peak from Caglioti's law (FWHM^2 = Utan(th)^2 + Vtan(th) + W) --> deprecated
        :return: an image
        """
        # Handle deprecated attributes ...
        if "U" in kwargs or "V" in kwargs or "W" in kwargs:
            logger.warning("The usage of (U,V,W) as parameters of `fake_calibration_image` is deprecated since 2025.07. Please use `resolution.Caglioti` instead")
            resolution = Caglioti(kwargs.get("U", 0), kwargs.get("V", 0), kwargs.get("W", 0))

        if not isinstance(resolution, _ResolutionFunction):
            resolution = Constant(resolution)

        if shape is None:
            if ai.detector.shape:
                shape = ai.detector.shape
            elif ai.detector.max_shape:
                shape = ai.detector.max_shape
        if shape is None:
            raise RuntimeError("No shape available")
        if (self.wavelength is None) and (ai._wavelength is not None):
            self.wavelength = ai.wavelength
        elif (self.wavelength is None) and (ai._wavelength is None):
            raise RuntimeError("Wavelength needed to calculate 2theta position")
        elif ((self.wavelength is not None)
                and (ai._wavelength is not None)
                and abs(self.wavelength - ai.wavelength) > 1e-15):
            logger.warning(
                "Mismatch between wavelength for calibrant (%s) and azimutal integrator (%s)",
                self.wavelength,
                ai.wavelength,
            )
        tth = ai.array_from_unit(shape=shape, typ="center", unit=units.TTH_DEG, scale=True)
        tth_min = tth.min()
        tth_max = tth.max()
        dim = int(numpy.sqrt(shape[0] * shape[0] + shape[1] * shape[1]))
        integrated = self.fake_xrpdp(dim, (tth_min, tth_max),
                                    background=Imin,
                                    Imax=Imax,
                                    resolution=resolution,
                                    unit=units.TTH_DEG)

        res = ai.calcfrom1d(
            integrated.radial,
            integrated.intensity,
            shape=shape,
            mask=ai.mask,
            dim1_unit=integrated.unit,
            correctSolidAngle=True)
        return res

    def __getnewargs_ex__(self):
        return (self._filename, self._dspacing, self._wavelength), {}

    def __getstate__(self):
        state_blacklist = ("_sem",)
        state = self.__dict__.copy()
        for key in state_blacklist:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        for statekey, statevalue in state.items():
            setattr(self, statekey, statevalue)
        self._sem = threading.Semaphore()

    @property
    @deprecated(reason="PEP8", replacement="dspacing", since_version="2025.07")
    def dSpacing(self):
        return self.dspacing

    @dSpacing.setter
    @deprecated(reason="PEP8", replacement="dspacing", since_version="2025.07")
    def dSpacing(self, value):
        self.dspacing.fset(value)

    get_dSpacing = deprecated(dspacing.fget, reason="property", replacement="dspacing", since_version="2025.07")
