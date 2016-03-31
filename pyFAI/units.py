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
__date__ = "31/03/2016"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger("pyFAI.unit")
import numpy
from numpy import pi

try:
    from .third_party import six
except (ImportError, Exception):
    import six

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


def eq_r(x, y, z=None, wavelength=None):
    """Calculates the radius

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: Vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    """
    return numpy.sqrt(x * x + y * y)


def eq_2th(x, y, z, wavelength=None):
    """Calculates the 2theta aperture of the cone

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: Vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    """
    return numpy.arctan2(eq_r(x, y), z)


def eq_q(x, y, z, wavelength):
    """Calculates the modulus of the scattering vector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: Vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    """
    return 4.0e-9 * numpy.pi * numpy.sin(eq_2th(x, y, z) / 2.0) / wavelength


def eq_rd2(x, y, z, wavelength):
    """Calculates the reciprocal spacing squared

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: Vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    """
    return (eq_q(x, y, z, wavelength) / (2.0 * numpy.pi)) ** 2


def eq_logq(x, y, z, wavelength):
    """Calculates the log (decimal) of the scattering vector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: Vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    """
    return numpy.log10(eq_q(x, y, z, wavelength) * 1e9)  # convert in meter


TTH_DEG = TTH = Enum(REPR="2th_deg",
#                      corner="cornerArray",
                     center="twoThetaArray",
                     delta="delta2Theta",
                     scale=180.0 / numpy.pi,
                     label=r"Scattering angle $2\theta$ ($^{o}$)",
                     equation=eq_2th)

TTH_RAD = Enum(REPR="2th_rad",
#                corner="cornerArray",
               center="twoThetaArray",
               delta="delta2Theta",
               scale=1.0,
               label=r"Scattering angle $2\theta$ ($rad$)",
               equation=eq_2th)

Q = Q_NM = Enum(REPR="q_nm^-1",
                center="qArray",
#                 corner="cornerQArray",
                delta="deltaQ",
                scale=1.0,
                label=r"Scattering vector $q$ ($nm^{-1}$)",
                equation=eq_q)

Q_A = Enum(REPR="q_A^-1",
           center="qArray",
#            corner="cornerQArray",
           delta="deltaQ",
           scale=0.1,
           label=r"Scattering vector $q$ ($\AA ^{-1}$)",
           equation=eq_q)

RecD2_A = Enum(REPR="d*2_A^-2",
               center="rd2Array",
#                corner="cornerRd2Array",
               delta="deltaRd2",
               scale=0.01,
               label=r"Reciprocal spacing squared $d^{*2}$ ($\AA ^{-2}$)",
               equation=eq_rd2)

RecD2_NM = Enum(REPR="d*2_nm^-2",
                center="rd2Array",
#                 corner="cornerRd2Array",
                delta="deltaRd2",
                scale=1.0,
                label=r"Reciprocal spacing squared $d^{*2}$ ($nm^{-2}$)",
                equation=eq_rd2)

R = R_MM = Enum(REPR="r_mm",
                center="rArray",
#                 corner="cornerRArray",
                delta="deltaR",
                scale=1000.0,
                label=r"Radius $r$ ($mm$)",
                equation=eq_r)

R_M = Enum(REPR="r_m",
                center="rArray",
#                 corner="cornerRArray",
                delta="deltaR",
                scale=1.0,
                label=r"Radius $r$ ($m$)",
                equation=eq_r)

LogQ = Enum(REPR="log(q)_m",
            # center="rArray",
            # corner="cornerRArray",
            # delta="deltaR",
            scale=1.0,
            label=r"log10($q$)",
            equation=eq_logq)


RADIAL_UNITS = (TTH_DEG, TTH_RAD, Q_NM, Q_A, R_MM, R_M, RecD2_A, RecD2_NM, LogQ)

l_m = Enum(REPR="m",
           scale=1.,
           label=r"length $l$ ($m$)")
l_mm = Enum(REPR="mm",
            scale=1e3,
            label=r"length $l$ ($mm$)")
l_cm = Enum(REPR="cm",
            scale=1e2,
            label=r"length $l$ ($cm$)")
l_um = Enum(REPR="micron",
            scale=1e6,
            label=r"length $l$ ($\mu m$)")
l_nm = Enum(REPR="nm",
            scale=1e9,
            label=r"length $l$ ($nm$)")
l_A = Enum(REPR="A",
           scale=1e10,
           label=r"length $l$ ($\AA$)")
LENGTH_UNITS = (l_m, l_mm, l_cm, l_um, l_nm, l_A)

A_deg = Enum(REPR="deg",
             scale=180.0 / pi,
             label=r"angle $\alpha$ ($^{o}$)")

A_rad = Enum(REPR="rad",
             scale=1.0,
             label=r"angle $\alpha$ ($rad$)")

ANGLE_UNITS = (A_deg, A_rad)


def to_unit(obj, type_=RADIAL_UNITS):
    rad_unit = None
    if isinstance(obj, six.string_types):
        for one_unit in type_:
            if one_unit.REPR == obj:
                rad_unit = one_unit
                break
    elif isinstance(obj, Enum):
        rad_unit = obj
    if rad_unit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s. "
                     "Valid units are %s" % (obj, type(obj), ", ".join([i.REPR for i in type_])))
    return rad_unit
