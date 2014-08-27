#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Calibrant

A module containing classical calibrant and also tools to generate d-spacing.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/02/2014"
__status__ = "production"

import os
import logging
import numpy
from math import sin, asin
import threading
logger = logging.getLogger("pyFAI.calibrant")
epsilon = 1.0e-6 # for floating point comparison

class Calibrant(object):
    """
    A calibrant is a reference compound where the d-spacing (interplanar distances)
    are known. They are expressed in Angstrom (in the file)
    """
    def __init__(self, filename=None, dSpacing=None, wavelength=None):
        object.__init__(self)
        self._filename = filename
        self._wavelength = wavelength
        self._sem = threading.Semaphore()
        self._2th = []
        if dSpacing is None:
            self._dSpacing = []
        else:
            self._dSpacing = list(dSpacing)
        if self._dSpacing and self._wavelength:
            self._calc_2th()

    def __repr__(self):
        name = "undefined"
        if self._filename:
            name = os.path.splitext(os.path.basename(self._filename))[0]
        name += " Calibrant "
        if len(self._dSpacing):
            name += "with %i reflections " % len(self._dSpacing)
        if self._wavelength:
            name += "at wavelength %s" % self._wavelength
        return name

    def load_file(self, filename=None):
        with self._sem:
            if filename:
                self._filename = filename
            if not os.path.isfile(self._filename):
                logger.error("No such calibrant file: %s" % self._filename)
                return
            self._filename = os.path.abspath(self._filename)
            self._dSpacing = numpy.unique(numpy.loadtxt(self._filename))
            self._dSpacing = list(self._dSpacing[-1::-1]) #reverse order
#            self._dSpacing.sort(reverse=True)
            if self._wavelength:
                self._calc_2th()

    def save_dSpacing(self, filename=None):
        """
        save the d-spacing to a file

        """
        if filename == None and self._filename is not None:
            filename = self._filename
        else:
            return
        with open(filename) as f:
            f.write("# %s Calibrant" % filename)
            for i in self.dSpacing:
                f.write("%s\n" % i)

    def get_dSpacing(self):
        if not self._dSpacing and self._filename:
            self.load_file()
        return self._dSpacing

    def set_dSpacing(self, lst):
        self._dSpacing = list(lst)
        self._filename = "Modified"
        if self._wavelength:
            self._calc_2th()
    dSpacing = property(get_dSpacing, set_dSpacing)

    def append_dSpacing(self, value):
        with self._sem:
            delta = [abs(value - v) / v for v in self._dSpacing if v is not None]
            if not delta or min(delta) > epsilon:
                self._dSpacing.append(value)
                self._dSpacing.sort(reverse=True)
                self._calc_2th()
    def append_2th(self, value):
        with self._sem:
            if value not in self._2th:
                self._2th.append(value)
                self._2th.sort()
                self._calc_dSpacing()

    def setWavelength_change2th(self, value=None):
        with self._sem:
            if value:
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                self._calc_2th()

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value :
                self._wavelength = float(value)
                if self._wavelength < 1e-15 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                self._calc_dSpacing()
                self._ring = [self.dSpacing.index(i) for i in d]

    def set_wavelength(self, value=None):
        updated = False
        with self._sem:
            if self._wavelength is None:
                if value:
                    self._wavelength = float(value)
                    if (self._wavelength < 1e-15) or (self._wavelength > 1e-6):
                        logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                    updated = True
            elif abs(self._wavelength - value) / self._wavelength > epsilon:
                logger.warning("Forbidden to change the wavelength once it is fixed !!!!")
                logger.warning("%s != %s, delta= %s" % (self._wavelength, value, self._wavelength - value))
        if updated:
            self._calc_2th()

    def get_wavelength(self):
        return self._wavelength
    wavelength = property(get_wavelength, set_wavelength)

    def _calc_2th(self):
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._2th = []
        for ds in self.dSpacing:
            try:
                tth = 2.0 * asin(5.0e9 * self._wavelength / ds)
            except ValueError:
                tth = None
            self._2th.append(tth)

    def _calc_dSpacing(self):
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._dSpacing = [5.0e9 * self._wavelength / sin(tth / 2.0) for tth in self._2th]

    def get_2th(self):
        if not self._2th:
            ds = self.dSpacing #forces the file reading if not done
            with self._sem:
                if not self._2th:
                    self._calc_2th()
        return self._2th

    def get_2th_index(self, angle):
        """
        return the index in the 2theta angle index
        """
        idx = None
        if angle:
            idx = self._2th.find(angle)
        if idx == -1:
            idx = None
        return idx

    def fake_calibration_image(self, ai, shape=None, Imax=1.0, U=0, V=0, W=0.0001):
        """
        Generates a fake calibration image from an azimuthal integrator

        @param ai: azimuthal integrator
        @param Imax: maximum intensity of rings
        @param U, V, W: width of the peak (FWHM = Utan(th)^2 + Vtan(th) + W)

        """
        if shape is None:
            if ai.detector.shape:
                shape = ai.detector.shape
            elif ai.detector.max_shape:
                 shape = ai.detector.max_shape
        if shape is None:
            raise RuntimeError("No shape available")
        tth = ai.twoThetaArray(shape)
        tth_min = tth.min()
        tth_max = tth.max()
        dim = int(numpy.sqrt(shape[0] * shape[0] + shape[1] * shape[1]))
        tth_1d = numpy.linspace(tth_min, tth_max, dim)
        tanth = numpy.tan(tth_1d / 2.0)
        fwhm = U * tanth ** 2 + V * tanth + W
        sigma2 = 8.0 * numpy.log(2.0) * fwhm * fwhm
        signal = numpy.zeros_like(sigma2)
        for t in self.get_2th():
            if t >= tth_max:
                break
            else:
                signal += Imax * numpy.exp(-(tth_1d - t) ** 2 / (2 * sigma2))
        res = ai.calcfrom1d(tth_1d, signal, shape=shape, mask=ai.mask,
                   dim1_unit='2th_rad', correctSolidAngle=True)
        return res


CALIBRANT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
if os.path.isdir(CALIBRANT_DIR):
    ALL_CALIBRANTS = dict([(os.path.splitext(i)[0], Calibrant(os.path.join(CALIBRANT_DIR, i)))
                           for i in os.listdir(CALIBRANT_DIR)
                           if i.endswith(".D")])
else:
    ALL_CALIBRANTS = {}
