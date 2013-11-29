#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author: Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
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
__authors__ = ["Picca Frédéric-Emmanuel", "Jérôme Kieffer"]
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/12/2012"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger("pyFAI.unit")
from numpy import pi
import types
hc = 12.398419292004204

class Enum(dict):
    """
    Simple class half way between a dict and a class, behaving as an enum
    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError

    def __repr__(self, *args, **kwargs):
        if "REPR" in self:
            return self["REPR"]
        else:
            return dict.__repr__(self, *args, **kwargs)

UNDEFINED = Enum(REPR='?')

TTH_DEG = TTH = Enum(REPR="2th_deg",
                     corner="cornerArray",
                     center="twoThetaArray",
                     delta="delta2Theta",
                     scale=180.0 / pi)

TTH_RAD = Enum(REPR="2th_rad",
               corner="cornerArray",
               center="twoThetaArray",
               delta="delta2Theta",
               scale=1.0)

Q = Q_NM = Enum(REPR="q_nm^-1",
                center="qArray",
                corner="cornerQArray",
                delta="deltaQ",
                scale=1.0)

Q_A = Enum(REPR="q_A^-1",
           center="qArray",
           corner="cornerQArray",
           delta="deltaQ",
           scale=0.1)

R = R_MM = Enum(REPR="r_mm",
                center="rArray",
                corner="cornerRArray",
                delta="deltaR",
                scale=1000.0)

RADIAL_UNITS = (TTH_DEG, TTH_RAD, Q_NM, Q_A, R_MM)

def to_unit(obj):
    rad_unit = None
    if type(obj) in types.StringTypes:
        for one_unit in RADIAL_UNITS:
            if one_unit.REPR == obj:
                rad_unit = one_unit
                break
    elif obj.__class__.__name__.split(".")[-1] == "Enum":
        rad_unit = obj
    if rad_unit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s. Valid units are 2th_deg, 2th_rad, q_nm^-1, q_A^-1 and r_mm" % (obj, type(obj)))
    return rad_unit
