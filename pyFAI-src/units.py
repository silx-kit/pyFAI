#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
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
__date__ = "26/10/2015"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger("pyFAI.unit")
from numpy import pi
try:
    import six
except (ImportError, Exception):
    from .third_party import six

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
    # ensures hashability
    def __hash__(self):
        return self.__repr__().__hash__()


UNDEFINED = Enum(REPR='?')

TTH_DEG = TTH = Enum(REPR="2th_deg",
                     corner="cornerArray",
                     center="twoThetaArray",
                     delta="delta2Theta",
                     scale=180.0 / pi,
                     label=r"Scattering angle $2\theta$ ($^{o}$)")

TTH_RAD = Enum(REPR="2th_rad",
               corner="cornerArray",
               center="twoThetaArray",
               delta="delta2Theta",
               scale=1.0,
               label=r"Scattering angle $2\theta$ ($rad$)")

Q = Q_NM = Enum(REPR="q_nm^-1",
                center="qArray",
                corner="cornerQArray",
                delta="deltaQ",
                scale=1.0,
                label=r"Scattering vector $q$ ($nm$^{-1}$)")

Q_A = Enum(REPR="q_A^-1",
           center="qArray",
           corner="cornerQArray",
           delta="deltaQ",
           scale=0.1,
           label=r"Scattering vector $q$ ($\AA ^{-1}$)")

RecD2_A = Enum(REPR="d*2_A^-2",
               center="rd2Array",
               corner="cornerRd2Array",
               delta="deltaRd2",
               scale=0.01,
               label=r"Reciprocal spacing squared $d^{*2}$ ($\AA ^{-2}$)")

RecD2_NM = Enum(REPR="d*2_nm^-2",
                center="rd2Array",
                corner="cornerRd2Array",
                delta="deltaRd2",
                scale=1.0,
                label=r"Reciprocal spacing squared $d^{*2}$ ($nm^{-2}$)")

R = R_MM = Enum(REPR="r_mm",
                center="rArray",
                corner="cornerRArray",
                delta="deltaR",
                scale=1000.0,
                label=r"Radius $r$ ($mm$)")


RADIAL_UNITS = (TTH_DEG, TTH_RAD, Q_NM, Q_A, R_MM, RecD2_A, RecD2_NM)


def to_unit(obj):
    rad_unit = None
    if isinstance(obj, six.string_types):
        for one_unit in RADIAL_UNITS:
            if one_unit.REPR == obj:
                rad_unit = one_unit
                break
    elif isinstance(obj, Enum):
        rad_unit = obj
    if rad_unit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s. Valid units are %s" % (obj, type(obj), ", ".join([i.REPR for i in RADIAL_UNITS])))
    return rad_unit
