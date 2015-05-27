#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
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
from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/05/2015"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import os
import logging
logger = logging.getLogger("pyFAI.multi_geometry")
from math import pi
from .azimuthalIntegrator import AzimuthalIntegrator
from . import units
from . import utils
from .utils import StringTypes, deprecated, EPS32
import fabio
import threading
import numpy
from numpy import rad2deg
error = None
from math import pi

class MultiGeometry(object):
    """
    This is an Azimuthal integrator containing multiple geometries (when 
    the detector is on a goniometer arm)
     
    """

    def __init__(self, ais, unit="2th_deg",
                 radial_range=(0, 180), azimuth_range=(-180, 180),
                 wavelength=None, empty=0.0):
        """
        Constructor of the multi-geometry integrator
        @param ais: list of azimuthal integrators
        @param radial_range: common range for integration
        @param azimuthal_range: common range for integration
        @param empty: value for empty pixels
        """
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.ais = [ ai if isinstance(ai, AzimuthalIntegrator) else AzimuthalIntegrator.sload(ai) for ai in ais]
        if wavelength:
            self.wavelength = float(wavelength)
        self.radial_range = tuple(radial_range[:2])
        self.azimuth_range = tuple(azimuth_range[:2])
        self.unit = units.to_unit(unit)
        self.abolute_solid_angle = None
        self.empty = empty

    def __repr__(self, *args, **kwargs):
        return "MultiGeometry integrator with %s geometries on %s radial range (%s) and %s azimuthal range (deg)" % \
            (len(self.ais), self.radial_range, self.unit, self.azimuth_range)

    def integrate1d(self, lst_data, npt=1800, monitors=None, all=False):
        """
        Perform 1D azimuthal integration
        
        @param lst_data: list of numpy array 
        @param npt: number of points int the integration
        @param monitors:
        """
        if monitors is None:
            monitors = [1.0] * len(self.ais)
        sum = numpy.zeros(npt, dtype=numpy.float64)
        count = numpy.zeros(npt, dtype=numpy.float64)
        for ai, data, monitor in zip(self.ais, lst_data, monitors):
            res = ai.integrate1d(data, npt=npt,
                                 correctSolidAngle=True,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 all=True)
            count += res["count"]
            sum += res["sum"] * (ai.pixel1 * ai.pixel2 / monitor / ai.dist ** 2)

        I = sum / numpy.maximum(count, 1)
        I[count == 0.0] = self.empty

        if all:
            out = {"I":I,
                 "radial": res["radial"],
                 "count": count,
                 "sum": sum}
#             if sigma is not None:
#                 res["sigma"] = sigma
        else:
#             if sigma is not None:
#                 res = I, res["radial"], res["azimuthal"], sigma
#             else:
                out = res["radial"], I
        return out

    def integrate2d(self, lst_data, npt_rad=1800, npt_azim=3600, monitors=None, all=False):
        """
        Perform 1D azimuthal integration of multiples frames, one for each geometry
        
        @param lst_data: list of numpy array 
        @param npt_rad: integration range
        @param monitors: 
        """
        if monitors is None:
            monitors = [1.0] * len(self.ais)
        sum = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        count = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        for ai, data, monitor in zip(self.ais, lst_data, monitors):
            res = ai.integrate2d(data, npt_rad=npt_rad, npt_azim=npt_azim,
                                 correctSolidAngle=True,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 all=True)
            count += res["count"]
            sum += res["sum"] * (ai.pixel1 * ai.pixel2 / monitor / ai.dist ** 2)

        I = sum / numpy.maximum(count, 1)
        I[count == 0.0] = self.empty

        if all:
            out = {"I": I,
                 "radial": res["radial"],
                 "azimuthal": res["azimuthal"],
                 "count": count,
                 "sum":  sum}
#             if sigma is not None:
#                 res["sigma"] = sigma
        else:
#             if sigma is not None:
#                 res = I, res["radial"], res["azimuthal"], sigma
#             else:
                out = I, res["radial"], res["azimuthal"]
        return out
