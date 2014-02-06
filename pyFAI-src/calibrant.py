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
logger = logging.getLogger("pyFAI.calibrant")

class Calibrant(object):
    """
    A calibrant is a reference compound where the d-spacing (interplanar distances)
    are known. They are expressed in Angstrom (in the file)
    """
    def __init__(self, filename=None):
        object.__init__(self)
        self._filename = filename
        self._dSpacing = None

    def __repr__(self):
        name = "undefined"
        if self._filename:
            name = os.path.splitext(os.path.basename(self._filename))[0]
        name += " Calibrant "
        if self._dSpacing:
            name += "with %i reflections" % len(self._dSpacing)
        return name

    def load_file(self, filename=None):
        if filename:
            self._filename = filename
        if not os.path.isfile(self._filename):
            logger.error("No such calibrant file: %s" % self._filename)
            return
        self._filename = os.path.abspath(self._filename)
        self._dSpacing = list(numpy.loadtxt(self._filename))
        self._dSpacing.sort(reverse=True)

    @property
    def dSpacing(self):
        if not self._dSpacing and self._filename:
            self.load_file()
        return self._dSpacing

CALIBRANT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration")
if os.path.isdir(CALIBRANT_DIR):
    ALL_CALIBRANTS = dict([(os.path.splitext(i)[0], Calibrant(os.path.join(CALIBRANT_DIR, i)))
                           for i in os.listdir(CALIBRANT_DIR)
                           if i.endswith(".D")])
else:
    ALL_CALIBRANTS = {}
