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
__date__ = "02/07/2025"
__status__ = "production"

import os
import logging
import numpy
import itertools
from typing import Optional, List
from math import sin, asin, cos, sqrt, pi, ceil
import threading
from ..utils import get_calibration_dir
from ..utils.decorators import deprecated
from .. import units
from .cell import Cell
from .space_groups import ReflectionCondition

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
    :param dSpacing: A list of d spacing in Angstrom.
    :param wavelength: A wavelength in meter
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        dSpacing: Optional[List[float]] = None,
        wavelength: Optional[float] = None,
    ):
        self._filename = filename
        self._wavelength = wavelength
        self._sem = threading.Semaphore()
        self._2th = []
        if filename is not None:
            self._dSpacing = None
        elif dSpacing is None:
            self._dSpacing = []
        else:
            self._dSpacing = list(dSpacing)
        self._out_dSpacing = []
        if self._dSpacing and self._wavelength:
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

        It only takes into account the wavelength and dSpacing, not the
        filename.

        :param other: Another object
        """
        if other is None:
            return False
        if not isinstance(other, Calibrant):
            return False
        if self._wavelength != other._wavelength:
            return False
        if self.dSpacing != other.dSpacing:
            return False
        return True

    def __ne__(self, other: object) -> bool:
        """
        Test the non-equality with another object

        It only takes into account the wavelength and dSpacing, not the
        filename.

        :param other: Another object
        """
        return not (self == other)

    def __hash__(self) -> int:
        """
        Returns the hash of the object.

        It only takes into account the wavelength and dSpacing, not the
        filename.
        """
        h = hash(self._wavelength)
        for d in self.dSpacing:
            h = h ^ hash(d)
        return h

    def __copy__(self) -> Calibrant:
        """
        Copy a calibrant.
        """
        self._initialize()
        return Calibrant(
            filename=self._filename,
            dSpacing=self._dSpacing + self._out_dSpacing,
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
        if len(self.dSpacing):
            name += "with %i reflections " % len(self._dSpacing)
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

    def _get_abs_path(self, filename: str) -> str:
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
        self._dSpacing = numpy.unique(numpy.loadtxt(path))
        self._dSpacing = list(self._dSpacing[-1::-1])  # reverse order
        # self._dSpacing.sort(reverse=True)
        if self._wavelength:
            self._calc_2th()

    def _initialize(self):
        """Initialize the object if expected."""
        if self._dSpacing is None:
            if self._filename:
                self._load_file()
            else:
                self._dSpacing = []

    def count_registered_dSpacing(self) -> int:
        """Count of registered dSpacing positions."""
        self._initialize()
        return len(self._dSpacing) + len(self._out_dSpacing)

    def save_dSpacing(self, filename: Optional[str] = None):
        """
        Save the d-spacing to a file.
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

    def get_dSpacing(self) -> List[float]:
        self._initialize()
        return self._dSpacing

    def set_dSpacing(self, lst: List[float]):
        self._dSpacing = list(lst)
        self._out_dSpacing = []
        self._filename = "Modified"
        if self._wavelength:
            self._calc_2th()

    dSpacing = property(get_dSpacing, set_dSpacing)

    def append_dSpacing(self, value: float):
        """Insert a d position at the right position of the dSpacing list"""
        self._initialize()
        with self._sem:
            delta = [abs(value - v) / v for v in self._dSpacing if v is not None]
            if not delta or min(delta) > epsilon:
                self._dSpacing.append(value)
                self._dSpacing.sort(reverse=True)
                self._calc_2th()

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

    def set_wavelength(self, value: Optional[float] = None):
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

    def get_wavelength(self) -> Optional[float]:
        """
        Returns the used wavelength.
        """
        return self._wavelength

    wavelength = property(get_wavelength, set_wavelength)

    def _calc_2th(self):
        """Calculate the 2theta positions for all peaks"""
        self._initialize()
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        tths = []
        dSpacing = self._dSpacing[:] + self._out_dSpacing  # explicit copy
        try:
            for ds in dSpacing:
                tth = 2.0 * asin(5.0e9 * self._wavelength / ds)
                tths.append(tth)
        except ValueError:
            size = len(tths)
            # remove dSpacing outside of 0..180
            self._dSpacing = dSpacing[:size]
            self._out_dSpacing = dSpacing[size:]
        else:
            self._dSpacing = dSpacing
            self._out_dSpacing = []
        self._2th = tths

    def _calc_dSpacing(self):
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._dSpacing = [
            5.0e9 * self._wavelength / sin(tth / 2.0) for tth in self._2th
        ]

    def get_2th(self) -> List[float]:
        """Returns the 2theta positions for all peaks (cached)"""
        if not self._2th:
            self._initialize()
            if not self._dSpacing:
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
        dSpacing = self._dSpacing[:] + self._out_dSpacing  # get all rings
        if index is None:
            index = len(dSpacing) - 1
        if index >= len(dSpacing):
            raise IndexError(
                "There are not than many (%s) rings indices in this calibrant" % (index)
            )
        return dSpacing[index] * 2e-10

    def get_peaks(self, unit: str = "2th_deg"):
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
            values = 20.0 * pi / numpy.array(self.get_dSpacing()[:size])
        else:
            raise ValueError("Only 2\theta and *q* units are supported for now")

        return values * scale

    def fake_calibration_image(
        self,
        ai,
        shape=None,
        Imax=1.0,
        Imin=0.0,
        U=0,
        V=0,
        W=0.0001,
    ) -> numpy.ndarray:
        """
        Generates a fake calibration image from an azimuthal integrator.
        :param ai: azimuthal integrator
        :param Imax: maximum intensity of rings
        :param Imin: minimum intensity of the signal (background)
        :param U, V, W: width of the peak from Caglioti's law (FWHM^2 = Utan(th)^2 + Vtan(th) + W)
        """
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
        elif (
            (self.wavelength is not None)
            and (ai._wavelength is not None)
            and abs(self.wavelength - ai.wavelength) > 1e-15
        ):
            logger.warning(
                "Mismatch between wavelength for calibrant (%s) and azimutal integrator (%s)",
                self.wavelength,
                ai.wavelength,
            )
        tth = ai.twoThetaArray(shape)
        tth_min = tth.min()
        tth_max = tth.max()
        dim = int(numpy.sqrt(shape[0] * shape[0] + shape[1] * shape[1]))
        tth_1d = numpy.linspace(tth_min, tth_max, dim)
        tanth = numpy.tan(tth_1d / 2.0)
        fwhm2 = U * tanth**2 + V * tanth + W
        sigma2 = fwhm2 / (8.0 * numpy.log(2.0))
        signal = numpy.zeros_like(sigma2)
        sigma_min = (sigma2.min()) ** 0.5
        sigma_max = (sigma2.max()) ** 0.5
        for t in self.get_2th():
            if t < (tth_min - 3 * sigma_min):
                continue
            elif t > (tth_max + 3 * sigma_max):
                break
            else:
                signal += Imax * numpy.exp(-((tth_1d - t) ** 2) / (2.0 * sigma2))
        signal = (Imax - Imin) * signal + Imin

        res = ai.calcfrom1d(
            tth_1d,
            signal,
            shape=shape,
            mask=ai.mask,
            dim1_unit="2th_rad",
            correctSolidAngle=True,
        )

        return res

    def __getnewargs_ex__(self):
        return (self._filename, self._dSpacing, self._wavelength), {}

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
