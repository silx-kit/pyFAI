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
__date__ = "06/02/2014"
__status__ = "development"

import os
import logging
import numpy
from math import sin, asin
import threading
logger = logging.getLogger("pyFAI.calibrant")

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
            self._dSpacing = list(numpy.loadtxt(self._filename))
            self._dSpacing.sort(reverse=True)
            if self._wavelength:
                self._calc_2th()

    def save_dSpacing(self, filename=None):
        """
        save the d-spacing to a file

        """
        if filename==None and self._filename is not None:
            filename = self._filename
        else:
            return
        with open(filename) as f:
            f.write("# %s Calibrant" % filename)
            for i in self.dSpacing:
                f.write("%s%s" % (i, os.linesep))

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
            if value not in self._dSpacing:
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
                if self._wavelength < 0 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                self._calc_2th()

    def setWavelength_changeDs(self, value=None):
        """
        This is probably not a good idea, but who knows !
        """
        with self._sem:
            if value :
                self._wavelength = float(value)
                if self._wavelength < 0 or self._wavelength > 1e-6:
                    logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                self._calc_dSpacing()
                self._ring = [self.dSpacing.index(i) for i in d]

    def set_wavelength(self, value=None):
        with self._sem:
            if self._wavelength is None:
                if value:
                    self._wavelength = float(value)
                    if self._wavelength < 0 or self._wavelength > 1e-6:
                        logger.warning("This is an unlikely wavelength (in meter): %s" % self._wavelength)
                    self._calc_2th()
            elif self._wavelength != value:
                logger.warning("Forbidden to change the wavelength once it is fixed !!!!")
                logger.warning("%s != %s" % (self._wavelength, value))
#                import traceback
#                traceback.print_stack()

    def get_wavelength(self):
        return self._wavelength
    wavelength = property(get_wavelength, set_wavelength)

    def _calc_2th(self):
        if self._wavelength is None:
            logger.error("Cannot calculate 2theta angle without knowing wavelength")
            return
        self._2th = []
        for ds in self._dSpacing:
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


CALIBRANT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
if os.path.isdir(CALIBRANT_DIR):
    ALL_CALIBRANTS = dict([(os.path.splitext(i)[0], Calibrant(os.path.join(CALIBRANT_DIR, i)))
                           for i in os.listdir(CALIBRANT_DIR)
                           if i.endswith(".D")])
else:
    ALL_CALIBRANTS = {}
