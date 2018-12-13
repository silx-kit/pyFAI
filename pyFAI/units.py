#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author: Picca Frédéric-Emmanuel <picca@synchrotron-soleil.fr>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Manages the different units

Nota for developers: this module is used a singleton to store all units in a
unique manner. This explains the number of top-level variables on the one
hand and their CAPITALIZATION on the other.
"""


from __future__ import division, print_function


__authors__ = ["Picca Frédéric-Emmanuel", "Jérôme Kieffer"]
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "13/12/2018"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import numpy
from numpy import pi
import scipy.constants

from .third_party import six

################################################################################
# A few physical constants
################################################################################

hc = CONST_hc = scipy.constants.c * scipy.constants.h / scipy.constants.e * 1e7
"""Product of h the Planck constant, and c the speed of light in vacuum
in Angstrom.KeV. It is approximativly equal to 12.398419292004204."""

CONST_q = 1.602176565e-19
"""One electron-volt is equal to 1.602176565⋅10-19 joules"""


class Unit(object):
    """Represents a unit.

    It has at least a name and a scale (in SI-unit)
    """
    def __init__(self, name, scale=1, label=None, equation=None,
                 center=None, corner=None, delta=None, short_name=None, unit_symbol=None):
        """Constructor of a unit.

        :param str name: name of the unit
        :param float scale: scale of th unit to go to SI
        :param string label: label for nice representation in matplotlib,
                                can use latex representation
        :param func equation: equation to calculate the value from coordinates
                                 (x,y,z) in detector space.
                                 Parameters of the function are x, y, z, lambda
        :param str center: name of the fast-path function
        :param str unit_symbol: Symbol used to display values of this unit
        """
        self.name = name
        self.scale = scale
        self.label = label if label is not None else name
        self.corner = corner
        self.center = center
        self.delta = delta
        self.equation = equation
        self.short_name = short_name
        self.unit_symbol = unit_symbol

    def get(self, key):
        """Mimic the dictionary interface

        :param (str) key: key wanted
        :return: self.key
        """
        res = None
        if key in dir(self):
            res = self.__getattribute__(key)
        return res

    def __repr__(self):
        return self.name

    # ensures hashability
    def __hash__(self):
        return self.name.__hash__()


RADIAL_UNITS = {}


def register_radial_unit(name, scale=1, label=None, equation=None,
                         center=None, corner=None, delta=None, short_name=None, unit_symbol=None):
    RADIAL_UNITS[name] = Unit(name, scale, label, equation, center, corner, delta, short_name, unit_symbol)


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


register_radial_unit("r_mm",
                     center="rArray",
                     delta="deltaR",
                     scale=1000.0,
                     label=r"Radius $r$ ($mm$)",
                     equation=eq_r,
                     short_name="r",
                     unit_symbol="mm")

register_radial_unit("r_m",
                     center="rArray",
                     delta="deltaR",
                     scale=1.0,
                     label=r"Radius $r$ ($m$)",
                     equation=eq_r,
                     short_name="r",
                     unit_symbol="m")

register_radial_unit("2th_deg", scale=180.0 / numpy.pi,
                     center="twoThetaArray",
                     delta="delta2Theta",
                     label=r"Scattering angle $2\theta$ ($^{o}$)",
                     equation=eq_2th,
                     short_name=r"2\theta",
                     unit_symbol="deg")

register_radial_unit("2th_rad",
                     center="twoThetaArray",
                     delta="delta2Theta",
                     scale=1.0,
                     label=r"Scattering angle $2\theta$ ($rad$)",
                     equation=eq_2th,
                     short_name=r"2\theta",
                     unit_symbol="rad")

register_radial_unit("q_nm^-1",
                     center="qArray",
                     delta="deltaQ",
                     scale=1.0,
                     label=r"Scattering vector $q$ ($nm^{-1}$)",
                     equation=eq_q,
                     short_name="q",
                     unit_symbol="nm^{-1}")

register_radial_unit("q_A^-1",
                     center="qArray",
                     delta="deltaQ",
                     scale=0.1,
                     label=r"Scattering vector $q$ ($\AA^{-1}$)",
                     equation=eq_q,
                     short_name="q",
                     unit_symbol="\AA^{-1}")

register_radial_unit("d*2_A^-2",
                     center="rd2Array",
                     delta="deltaRd2",
                     scale=0.01,
                     label=r"Reciprocal spacing squared $d^{*2}$ ($\AA^{-2}$)",
                     equation=lambda x, y, z, wavelength: (eq_q(x, y, z, wavelength) / (2.0 * numpy.pi)) ** 2,
                     short_name="d^{*2}",
                     unit_symbol="\AA^{-2}")

register_radial_unit("d*2_nm^-2",
                     center="rd2Array",
                     delta="deltaRd2",
                     scale=1.0,
                     label=r"Reciprocal spacing squared $d^{*2}$ ($nm^{-2}$)",
                     equation=lambda x, y, z, wavelength: (eq_q(x, y, z, wavelength) / (2.0 * numpy.pi)) ** 2,
                     short_name="d^{*2}",
                     unit_symbol="nm^{-2}")

register_radial_unit("log10(q.m)_None",
                     scale=1.0,
                     label=r"log10($q$.m)",
                     equation=lambda x, y, z, wavelength: numpy.log10(1e9 * eq_q(x, y, z, wavelength)),
                     short_name="log10(q.m)",
                     unit_symbol="?")

register_radial_unit("log(q.nm)_None",
                     scale=1.0,
                     label=r"log($q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.log(eq_q(x, y, z, wavelength)),
                     short_name="log(q.nm)",
                     unit_symbol="?")

register_radial_unit("log(1+q.nm)_None",
                     scale=1.0,
                     label=r"log(1+$q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.log1p(eq_q(x, y, z, wavelength)),
                     short_name="log(1+q.nm)",
                     unit_symbol="?")

register_radial_unit("log(1+q.A)_None",
                     scale=1.0,
                     label=r"log(1+$q$.\AA)",
                     equation=lambda x, y, z, wavelength: numpy.log1p(0.1 * eq_q(x, y, z, wavelength)),
                     short_name=r"log(1+q.\AA)",
                     unit_symbol="?")

register_radial_unit("arcsinh(q.nm)_None",
                     scale=1.0,
                     label=r"arcsinh($q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.arcsinh(eq_q(x, y, z, wavelength)),
                     short_name="arcsinh(q.nm)",
                     unit_symbol="?")

register_radial_unit("arcsinh(q.A)_None",
                     scale=1.0,
                     label=r"arcsinh($q$.\AA)",
                     equation=lambda x, y, z, wavelength: numpy.arcsinh(0.1 * eq_q(x, y, z, wavelength)),
                     short_name=r"arcsinh(q.\AA)",
                     unit_symbol="?")


LENGTH_UNITS = {"m": Unit("m", scale=1., label=r"length $l$ ($m$)"),
                "mm": Unit("mm", scale=1e3, label=r"length $l$ ($mm$)"),
                "cm": Unit("cm", scale=1e2, label=r"length $l$ ($cm$)"),
                "micron": Unit("micron", scale=1e6, label=r"length $l$ ($\mu m$)"),
                "nm": Unit("nm", scale=1e9, label=r"length $l$ ($nm$)"),
                "A": Unit("A", scale=1e10, label=r"length $l$ ($\AA$)"),
                }


ANGLE_UNITS = {"deg": Unit("deg", scale=180.0 / pi, label=r"angle $\alpha$ ($^{o}$)"),
               "rad": Unit("rad", scale=1.0, label=r"angle $\alpha$ ($rad$)"),
               }

AZIMUTHAL_UNITS = {"chi_rad": Unit("chi_rad", scale=1.0, label=r"Azimuthal angle $\chi$ ($rad$)"),
                   "chi_deg": Unit("chi_deg", scale=180 / pi, label=r"Azimuthal angle $\chi$ ($^{o}$)")}


def to_unit(obj, type_=None):
    if type_ is None:
        type_ = RADIAL_UNITS
    rad_unit = None
    if isinstance(obj, six.string_types):
        rad_unit = type_.get(obj)
    elif isinstance(obj, Unit):
        rad_unit = obj
    if rad_unit is None:
        logger.error("Unable to recognize this type unit '%s' of type %s. "
                     "Valid units are %s" % (obj, type(obj), ", ".join([i for i in type_])))
    return rad_unit


# To ensure the compatibility with former code:
Q = Q_NM = RADIAL_UNITS["q_nm^-1"]
Q_A = RADIAL_UNITS["q_A^-1"]
TTH_RAD = RADIAL_UNITS["2th_rad"]
TTH_DEG = TTH = RADIAL_UNITS["2th_deg"]
R = R_MM = RADIAL_UNITS["r_mm"]
R_M = RADIAL_UNITS["r_m"]
RecD2_NM = RADIAL_UNITS["d*2_nm^-2"]
l_m = LENGTH_UNITS["m"]
A_rad = ANGLE_UNITS["rad"]
CHI_DEG = AZIMUTHAL_UNITS["chi_deg"]
CHI_RAD = AZIMUTHAL_UNITS["chi_rad"]
