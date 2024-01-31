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

__authors__ = ["Picca Frédéric-Emmanuel", "Jérôme Kieffer"]
__contact__ = "picca@synchrotron-soleil.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/10/2023"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
import numpy
from numpy import pi
import scipy.constants
try:
    import numexpr
except (ImportError, ModuleNotFoundError):
    numexpr = None

################################################################################
# A few physical constants
################################################################################

CONST_hc = hc = scipy.constants.c * scipy.constants.h / scipy.constants.e * 1e7
"""Product of h the Planck constant, and c the speed of light in vacuum
in Angstrom.KeV. It is approximatively equal to:

- pyFAI reference: 12.398419292004204
- scipy v1.3.1:   12.398419739640717
- scipy-1.4.0rc1: 12.398419843320026
"""

CONST_q = scipy.constants.e
"""One electron-volt is equal to 1.602176634⋅10-19 joules"""


class Unit(object):
    """Represents a unit.

    It has at least a name and a scale (in SI-unit)
    """

    def __init__(self, name, scale=1, label=None, equation=None, formula=None,
                 center=None, corner=None, delta=None, short_name=None, unit_symbol=None,
                 positive=True, period=None):
        """Constructor of a unit.

        :param str name: name of the unit
        :param float scale: scale of the unit to go to SI
        :param str label: label for nice representation in matplotlib,
                                can use latex representation
        :param func equation: equation to calculate the value from coordinates
                                 (x,y,z) in detector space.
                                 Parameters of the function are `x`, `y`, `z`, `wavelength`
        :param str formula: string with the mathematical formula.
                       Valid variable names are `x`, `y`, `z`, `λ` and the constant `π`
        :param str center: name of the fast-path function
        :param str unit_symbol: symbol used to display values of this unit
        :param bool positive: this value can only be positive
        :param period: None or the periodicity of the unit (angles are periodic)
        """
        self.name = name
        self.space = name.split("_")[0]  # used to idenfify compatible spaces.
        self.scale = scale
        self.label = label if label is not None else name
        self.corner = corner
        self.center = center
        self.delta = delta
        self._equation = equation
        self.formula = formula
        if (numexpr is not None) and isinstance(formula, str):
            signature = [(key, numpy.float64) for key in "xyzλπ" if key in formula]
            ne_formula = numexpr.NumExpr(formula, signature)

            def ne_equation(x, y, z=None, wavelength=None, ne_formula=ne_formula):
                π = numpy.pi
                λ = wavelength
                ldict = locals()
                args = tuple(ldict[i] for i in ne_formula.input_names)
                return ne_formula(*args)

            self.equation = ne_equation
        else:
            self.equation = self._equation
        self.short_name = short_name
        self.unit_symbol = unit_symbol
        self.positive = positive
        self.period = period

    def get(self, key):
        """Mimics the dictionary interface

        :param str key: key wanted
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
AZIMUTHAL_UNITS = {}
ANY_UNITS = {}


def register_radial_unit(name, scale=1, label=None, equation=None, formula=None,
                         center=None, corner=None, delta=None, short_name=None,
                         unit_symbol=None, positive=True, period=None):
    RADIAL_UNITS[name] = Unit(name, scale, label, equation, formula, center,
                              corner, delta, short_name, unit_symbol, positive, period)
    ANY_UNITS.update(RADIAL_UNITS)


def register_azimuthal_unit(name, scale=1, label=None, equation=None, formula=None,
                         center=None, corner=None, delta=None, short_name=None,
                         unit_symbol=None, positive=False, period=None):
    AZIMUTHAL_UNITS[name] = Unit(name, scale, label, equation, formula, center,
                                 corner, delta, short_name, unit_symbol, positive, period)
    ANY_UNITS.update(AZIMUTHAL_UNITS)


def eq_r(x, y, z=None, wavelength=None):
    """Calculates the radius in meter

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: radius in meter
    """
    return numpy.sqrt(x * x + y * y)


def eq_2th(x, y, z, wavelength=None):
    """Calculates the 2theta aperture of the cone

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: opening angle 2θ in radian
    """
    return numpy.arctan2(eq_r(x, y), z)


def eq_q(x, y, z, wavelength):
    """Calculates the modulus of the scattering vector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: modulus of the scattering vector q in inverse nm
    """
    return 4.0e-9 * numpy.pi * numpy.sin(eq_2th(x, y, z) / 2.0) / wavelength


def eq_exitangle(x, y, z, wavelength=None, incident_angle=0.0):
    """Calculates the vertical exit scattering angle (relative to incident angle), used for grazing incidence

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: modulus of the scattering vector q in inverse nm
    """
    return numpy.arctan2(y, z) - incident_angle


def eq_qhorz(hpos, vpos, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the horizontal direction in the sample frame (for grazing-incidence geometries), towards the center of the ring

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the horizontal direction in inverse nm
    """
    exit_angle = eq_exitangle(x=hpos, y=vpos, z=z, incident_angle=incident_angle)
    c1 = numpy.cos(exit_angle) * numpy.sin(numpy.arctan2(hpos, z)) * numpy.cos(tilt_angle)
    c2 = numpy.sin(exit_angle) * numpy.sin(tilt_angle)
    c3 = numpy.sin(incident_angle) * numpy.sin(tilt_angle)
    return 2.0e-9 * numpy.pi * (c1 - c2 - c3)/ wavelength


def eq_qvert(hpos, vpos, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the vertical direction in the sample frame (for grazing-incidence geometries), to the roof

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the vertical direction in inverse nm
    """
    exit_angle = eq_exitangle(x=hpos, y=vpos, z=z, incident_angle=incident_angle)
    c1 = numpy.cos(exit_angle) * numpy.sin(numpy.arctan2(hpos, z)) * numpy.sin(tilt_angle)
    c2 = numpy.sin(exit_angle) * numpy.cos(tilt_angle)
    c3 = numpy.sin(incident_angle) * numpy.cos(tilt_angle)
    return 2.0e-9 * numpy.pi * (c1 + c2 + c3) / wavelength


def eq_qbeam(hpos, vpos, z, wavelength, incident_angle=0.0):
    """Calculates the component of the scattering vector along the beam propagation direction in the sample frame (for grazing-incidence geometries)

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :return: component of the scattering vector along the beam propagation direction in inverse nm
    """
    exit_angle = eq_exitangle(x=hpos, y=vpos, z=z, incident_angle=incident_angle)
    c1 = numpy.cos(exit_angle) * numpy.cos(numpy.arctan2(hpos, z))
    c2 = numpy.cos(incident_angle)
    return 2.0e-9 * numpy.pi * (c1 - c2) / wavelength


def eq_qxgi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the horizontal direction in the sample frame (for grazing-incidence geometries), towards the center of the ring

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the horizontal direction in inverse nm
    """
    return eq_qhorz(hpos=x, vpos=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qxgi_rot90(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the horizontal direction in the sample frame (for grazing-incidence geometries), towards the center of the ring
    Use if the horizontal axis of the lab frame is the vertical axis of the detector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the horizontal direction in inverse nm
    """
    return eq_qhorz(hpos=y, vpos=x, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qygi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the vertical direction in the sample frame (for grazing-incidence geometries), to the roof

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the vertical direction in inverse nm
    """
    return eq_qvert(hpos=x, vpos=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qygi_rot90(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector along the vertical direction in the sample frame (for grazing-incidence geometries), to the roof
    Use if the horizontal axis of the lab frame is the vertical axis of the detector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the vertical direction in inverse nm
    """
    return eq_qvert(hpos=y, vpos=x, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qzgi(x, y, z, wavelength, incident_angle=0.0):
    """Calculates the component of the scattering vector along the beam propagation direction in the sample frame (for grazing-incidence geometries)

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :return: component of the scattering vector along the beam propagation direction in inverse nm
    """
    return eq_qbeam(hpos=x, vpos=y, z=z, wavelength=wavelength, incident_angle=incident_angle)


def eq_qzgi_rot90(x, y, z, wavelength, incident_angle=0.0):
    """Calculates the component of the scattering vector along the beam propagation direction in the sample frame (for grazing-incidence geometries)
    Use if the horizontal axis of the lab frame is the vertical axis of the detector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :return: component of the scattering vector along the beam propagation direction in inverse nm
    """
    return eq_qbeam(hpos=y, vpos=x, z=z, wavelength=wavelength, incident_angle=incident_angle)


def eq_qip(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector in the plane YZ in the sample frame (for grazing-incidence geometries)

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    qxgi = eq_qxgi(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)
    qzgi = eq_qzgi(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle)
    return numpy.sqrt(qxgi ** 2 + qzgi ** 2) * numpy.sign(qxgi)


def eq_qip_rot90(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector in the plane XZ in the sample frame (for grazing-incidence geometries)
    Use if the horizontal axis of the lab frame is the vertical axis of the detector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    qygi = eq_qxgi_rot90(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)
    qzgi = eq_qzgi_rot90(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle)
    return numpy.sqrt(qygi ** 2 + qzgi ** 2) * numpy.sign(qygi)


def eq_qoop(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector in the vertical direction in the sample frame (for grazing-incidence geometries)

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    return eq_qygi(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qoop_rot90(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0):
    """Calculates the component of the scattering vector in the vertical direction in the sample frame (for grazing-incidence geometries)
    Use if the horizontal axis of the lab frame is the vertical axis of the detector

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    return eq_qygi_rot90(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)

formula_r = "sqrt(x * x + y * y)"
formula_2th = "arctan2(sqrt(x * x + y * y), z)"
formula_chi = "arctan2(y, x)"
formula_q = "4.0e-9*π/λ*sin(0.5*arctan2(sqrt(x * x + y * y), z))"
formula_d = "0.5*λ/sin(0.5*arctan2(sqrt(x * x + y * y), z))"
formula_d2 = "(2.0e-9/λ*sin(0.5*arctan2(sqrt(x * x + y * y), z)))**2"
formula_qx = "4.0e-9*π/λ*sin(arctan2(x, z)/2.0)"
formula_qy = "4.0e-9*π/λ*sin(arctan2(y, z)/2.0)"

register_radial_unit("r_mm",
                     center="rArray",
                     delta="deltaR",
                     scale=1000.0,
                     label=r"Radius $r$ ($mm$)",
                     equation=eq_r,
                     formula=formula_r,
                     short_name="r",
                     unit_symbol="mm")

register_radial_unit("r_m",
                     center="rArray",
                     delta="deltaR",
                     scale=1.0,
                     label=r"Radius $r$ ($m$)",
                     equation=eq_r,
                     formula=formula_r,
                     short_name="r",
                     unit_symbol="m")

register_radial_unit("2th_deg", scale=180.0 / numpy.pi,
                     center="twoThetaArray",
                     delta="delta2Theta",
                     label=r"Scattering angle $2\theta$ ($^{o}$)",
                     equation=eq_2th,
                     formula=formula_2th,
                     short_name=r"2\theta",
                     unit_symbol="deg")

register_radial_unit("2th_rad",
                     center="twoThetaArray",
                     delta="delta2Theta",
                     scale=1.0,
                     label=r"Scattering angle $2\theta$ ($rad$)",
                     equation=eq_2th,
                     formula=formula_2th,
                     short_name=r"2\theta",
                     unit_symbol="rad")

register_radial_unit("q_nm^-1",
                     center="qArray",
                     delta="deltaQ",
                     scale=1.0,
                     label=r"Scattering vector $q$ ($nm^{-1}$)",
                     equation=eq_q,
                     formula=formula_q,
                     short_name="q",
                     unit_symbol="nm^{-1}")

register_radial_unit("q_A^-1",
                     center="qArray",
                     delta="deltaQ",
                     scale=0.1,
                     label=r"Scattering vector $q$ ($\AA^{-1}$)",
                     equation=eq_q,
                     formula=formula_q,
                     short_name="q",
                     unit_symbol=r"\AA^{-1}")

register_radial_unit("d_m",
                     scale=1,
                     label=r"d-spacing $d$ ($m$)",
                     equation=lambda x, y, z, wavelength: ((2.0 * numpy.pi) / (1e9 * eq_q(x, y, z, wavelength))),
                     formula=formula_d,
                     short_name="d",
                     unit_symbol=r"m")

register_radial_unit("d_nm",
                     scale=1e9,
                     label=r"d-spacing $d$ ($nm$)",
                     equation=lambda x, y, z, wavelength: ((2.0 * numpy.pi) / (1e9 * eq_q(x, y, z, wavelength))),
                     formula=formula_d,
                     short_name="d",
                     unit_symbol=r"nm")

register_radial_unit("d_A",
                     scale=1e10,
                     label=r"d-spacing $d$ ($\AA$)",
                     equation=lambda x, y, z, wavelength: ((2.0 * numpy.pi) / (1e9 * eq_q(x, y, z, wavelength))),
                     formula=formula_d,
                     short_name="d",
                     unit_symbol=r"\AA")

register_radial_unit("d*2_A^-2",
                     center="rd2Array",
                     delta="deltaRd2",
                     scale=0.01,
                     label=r"Recip. spacing sq. $d^{*2}$ ($\AA^{-2}$)",
                     equation=lambda x, y, z, wavelength: (eq_q(x, y, z, wavelength) / (2.0 * numpy.pi)) ** 2,
                     formula=formula_d2,
                     short_name="d^{*2}",
                     unit_symbol=r"\AA^{-2}")

register_radial_unit("d*2_nm^-2",
                     center="rd2Array",
                     delta="deltaRd2",
                     scale=1.0,
                     label=r"Recip. spacing sq. $d^{*2}$ ($nm^{-2}$)",
                     equation=lambda x, y, z, wavelength: (eq_q(x, y, z, wavelength) / (2.0 * numpy.pi)) ** 2,
                     formula=formula_d2,
                     short_name="d^{*2}",
                     unit_symbol="nm^{-2}")

register_radial_unit("log10(q.m)_None",
                     scale=1.0,
                     label=r"log10($q$.m)",
                     equation=lambda x, y, z, wavelength: numpy.log10(1e9 * eq_q(x, y, z, wavelength)),
                     formula="log10(4*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name="log10(q.m)",
                     unit_symbol="?",
                     positive=False)

register_radial_unit("log(q.nm)_None",
                     scale=1.0,
                     label=r"log($q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.log(eq_q(x, y, z, wavelength)),
                     formula="log(4e-9*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name="log(q.nm)",
                     unit_symbol="?",
                     positive=False)

register_radial_unit("log(1+q.nm)_None",
                     scale=1.0,
                     label=r"log(1+$q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.log1p(eq_q(x, y, z, wavelength)),
                     formula="log1p(4e-9*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name="log(1+q.nm)",
                     unit_symbol="?",
                     positive=True)

register_radial_unit("log(1+q.A)_None",
                     scale=1.0,
                     label=r"log(1+$q$.$\AA$)",
                     equation=lambda x, y, z, wavelength: numpy.log1p(0.1 * eq_q(x, y, z, wavelength)),
                     formula="log1p(4e-10*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name=r"log(1+q.\AA)",
                     unit_symbol="?",
                     positive=True)

register_radial_unit("arcsinh(q.nm)_None",
                     scale=1.0,
                     label=r"arcsinh($q$.nm)",
                     equation=lambda x, y, z, wavelength: numpy.arcsinh(eq_q(x, y, z, wavelength)),
                     formula="arcsinh(4e-9*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name="arcsinh(q.nm)",
                     unit_symbol="?",
                     positive=True)

register_radial_unit("arcsinh(q.A)_None",
                     scale=1.0,
                     label=r"arcsinh($q$.$\AA$)",
                     equation=lambda x, y, z, wavelength: numpy.arcsinh(0.1 * eq_q(x, y, z, wavelength)),
                     formula="arcsinh(4e-10*π/λ*sin(arctan2(sqrt(x * x + y * y), z)/2.0))",
                     short_name=r"arcsinh(q.\AA)",
                     unit_symbol="?",
                     positive=True)

register_radial_unit("qx_nm^-1",
                     scale=1.0,
                     label=r"Rectilinear scattering vector $q_x$ ($nm^{-1}$)",
                     formula=formula_qx,
                     short_name="qx",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qy_nm^-1",
                     scale=1.0,
                     label=r"Rectilinear scattering vector $q_y$ ($nm^{-1}$)",
                     formula=formula_qy,
                     short_name="qy",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("exitangle_rad",
                     scale=1.0,
                     label=r"Exit scattering angle (rad)",
                     equation=eq_exitangle,
                     short_name="exitangle",
                     unit_symbol="rad",
                     positive=False)

register_radial_unit("qxgi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_x$ ($nm^{-1}$)",
                     equation=eq_qxgi,
                     short_name="qxgi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qxgirot90_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_x$ ($nm^{-1}$)",
                     equation=eq_qxgi_rot90,
                     short_name="qxgirot90",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qygi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_y$ ($nm^{-1}$)",
                     equation=eq_qygi,
                     short_name="qygi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qygirot90_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_y$ ($nm^{-1}$)",
                     equation=eq_qygi_rot90,
                     short_name="qygirot90",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qzgi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_z$ ($nm^{-1}$)",
                     equation=eq_qzgi,
                     short_name="qzgi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qzgirot90_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_z$ ($nm^{-1}$)",
                     equation=eq_qzgi_rot90,
                     short_name="qzgirot90",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qip_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{IP}$ ($nm^{-1}$)",
                     equation=eq_qip,
                     short_name="qip",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qiprot90_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{IP}$ ($nm^{-1}$)",
                     equation=eq_qip_rot90,
                     short_name="qiprot90",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qoop_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{OOP}$ ($nm^{-1}$)",
                     equation=eq_qoop,
                     short_name="qoop",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qooprot90_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{OOP}$ ($nm^{-1}$)",
                     equation=eq_qoop_rot90,
                     short_name="qooprot90",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_unit("qxgi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_x$ ($A^{-1}$)",
                     equation=eq_qxgi,
                     short_name="qxgi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qxgirot90_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_x$ ($A^{-1}$)",
                     equation=eq_qxgi_rot90,
                     short_name="qxgirot90",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qygi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_y$ ($A^{-1}$)",
                     equation=eq_qygi,
                     short_name="qygi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qygirot90_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_y$ ($A^{-1}$)",
                     equation=eq_qygi_rot90,
                     short_name="qygirot90",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qzgi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_z$ ($A^{-1}$)",
                     equation=eq_qzgi,
                     short_name="qzgi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qzgirot90_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_z$ ($A^{-1}$)",
                     equation=eq_qzgi_rot90,
                     short_name="qzgirot90",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qip_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{IP}$ ($A^{-1}$)",
                     equation=eq_qip,
                     short_name="qip",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qiprot90_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{IP}$ ($A^{-1}$)",
                     equation=eq_qip_rot90,
                     short_name="qiprot90",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qoop_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{OOP}$ ($A^{-1}$)",
                     equation=eq_qoop,
                     short_name="qoop",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_unit("qooprot90_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{OOP}$ ($A^{-1}$)",
                     equation=eq_qoop_rot90,
                     short_name="qooprot90",
                     unit_symbol="A^{-1}",
                     positive=False)


LENGTH_UNITS = {"m": Unit("m", scale=1., label=r"length $l$ ($m$)", positive=False),
                "mm": Unit("mm", scale=1e3, label=r"length $l$ ($mm$)", positive=False),
                "cm": Unit("cm", scale=1e2, label=r"length $l$ ($cm$)", positive=False),
                "micron": Unit("micron", scale=1e6, label=r"length $l$ ($\mu m$)", positive=False),
                "nm": Unit("nm", scale=1e9, label=r"length $l$ ($nm$)", positive=False),
                "A": Unit("A", scale=1e10, label=r"length $l$ ($\AA$)", positive=False),
                }

ANGLE_UNITS = {"deg": Unit("deg", scale=180.0 / pi, label=r"angle $\alpha$ ($^{o}$)", positive=False, period=360),
               "rad": Unit("rad", scale=1.0, label=r"angle $\alpha$ ($rad$)", positive=False, period=2 * numpy.pi),
               }

register_azimuthal_unit("chi_rad",
                        scale=1.0,
                        label=r"Azimuthal angle $\chi$ ($rad$)",
                        formula=formula_chi,
                        positive=False,
                        period=2.*pi)
register_azimuthal_unit("chi_deg",
                        scale=180. / pi,
                        label=r"Azimuthal angle $\chi$ ($^{o}$)",
                        formula=formula_chi,
                        positive=False,
                        period=360)

AZIMUTHAL_UNITS["qx_nm^-1"] = RADIAL_UNITS["qx_nm^-1"]
AZIMUTHAL_UNITS["qy_nm^-1"] = RADIAL_UNITS["qy_nm^-1"]
AZIMUTHAL_UNITS["qxgi_nm^-1"] = RADIAL_UNITS["qxgi_nm^-1"]
AZIMUTHAL_UNITS["qygi_nm^-1"] = RADIAL_UNITS["qygi_nm^-1"]
AZIMUTHAL_UNITS["qzgi_nm^-1"] = RADIAL_UNITS["qzgi_nm^-1"]
AZIMUTHAL_UNITS["qip_nm^-1"] = RADIAL_UNITS["qip_nm^-1"]
AZIMUTHAL_UNITS["qoop_nm^-1"] = RADIAL_UNITS["qoop_nm^-1"]
AZIMUTHAL_UNITS["qiprot90_nm^-1"] = RADIAL_UNITS["qiprot90_nm^-1"]
AZIMUTHAL_UNITS["qooprot90_nm^-1"] = RADIAL_UNITS["qooprot90_nm^-1"]
AZIMUTHAL_UNITS["qip_A^-1"] = RADIAL_UNITS["qip_A^-1"]
AZIMUTHAL_UNITS["qoop_A^-1"] = RADIAL_UNITS["qoop_A^-1"]
AZIMUTHAL_UNITS["qiprot90_A^-1"] = RADIAL_UNITS["qiprot90_A^-1"]
AZIMUTHAL_UNITS["qooprot90_A^-1"] = RADIAL_UNITS["qooprot90_A^-1"]

def to_unit(obj, type_=None):
    """Convert to Unit object

    :param obj: can be a unit or a string like "2th_deg"
    :param type_: family of units like AZIMUTHAL_UNITS or RADIAL_UNITS
    :return: Unit instance
    """
    rad_unit = None
    if type_ is None:
        type_ = ANY_UNITS
    if isinstance(obj, (str,)):
        rad_unit = type_.get(obj)
    elif isinstance(obj, Unit):
        rad_unit = obj
    # elif isinstance(obj, (list, tuple)) and len(obj) == 2:
    #     rad_unit = tuple(to_unit(i) for i in obj)
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
D_A = RADIAL_UNITS["d_A"]
D_NM = RADIAL_UNITS["d_nm"]
D_M = RADIAL_UNITS["d_m"]
RecD2_NM = RADIAL_UNITS["d*2_nm^-2"]
RecD2_A = RADIAL_UNITS["d*2_A^-2"]
l_m = LENGTH_UNITS["m"]
A_rad = ANGLE_UNITS["rad"]
CHI_DEG = AZIMUTHAL_UNITS["chi_deg"]
CHI_RAD = AZIMUTHAL_UNITS["chi_rad"]
