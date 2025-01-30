#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "24/12/2024"
__status__ = "production"
__docformat__ = 'restructuredtext'

import copy
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
        self.space = "_".join(self.name.split("_")[:-1])  # used to idenfify compatible spaces.
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

class UnitFiber(Unit):
    """Represents a unit + two rotation axis. To be used in a Grazing-Incidence or Fiber Diffraction/Scattering experiment.

    Fiber parameters:
    :param float incident_angle: pitch angle; projection angle of the beam in the sample. Its rotation axis is the horizontal axis of the lab system.
    :param float tilt angle: roll angle; its rotation axis is the beam axis. Tilting of the horizon for grazing incidence in thin films.
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)

    It has at least a name and a scale (in SI-unit)
    """
    def __init__(self, name, scale=1, label=None, equation=None, formula=None,
                 incident_angle=0.0, tilt_angle=0.0, sample_orientation=1,
                 center=None, corner=None, delta=None, short_name=None, unit_symbol=None,
                 positive=True, period=None):
        super().__init__(
            name=name,
            scale=scale,
            label=label,
            equation=equation,
            # formula=formula,
            center=center,
            corner=corner,
            delta=delta,
            short_name=short_name,
            unit_symbol=unit_symbol,
            positive=positive,
            period=period,
        )
        self.formula = formula
        self.formula_so1 = formula
        self._incident_angle = incident_angle
        self._tilt_angle = tilt_angle
        self._sample_orientation = sample_orientation
        self._update_ne_equation()

    def _update_ne_equation(self):
        """Updates the string-equation for the current used sample orientation following the EXIF orientation values
        https://sirv.com/help/articles/rotate-photos-to-be-upright/

        Sample orientations
        1 - No changes are applied to the image
        2 - Image is mirrored (flipped horizontally)
        3 - Image is rotated 180 degrees
        4 - Image is rotated 180 degrees and mirrored
        5 - Image is mirrored and rotated 90 degrees counter clockwise
        6 - Image is rotated 90 degrees counter clockwise
        7 - Image is mirrored and rotated 90 degrees clockwise
        8 - Image is rotated 90 degrees clockwise
        """
        if (numexpr is not None) and isinstance(self.formula, str):
            signature = [(key, numpy.float64) for key in "xyzλπηχ" if key in self.formula]

            formula_ = self.formula
            if self._sample_orientation == 1:
                ...
            elif self._sample_orientation == 2:
                formula_ = self.formula_so1.replace('x', '(-x)')
            elif self._sample_orientation == 3:
                formula_ = self.formula_so1
                formula_ = formula_.replace('x', 'ψ').replace('y', 'ξ')
                formula_ = formula_.replace('ψ', '(-x)').replace('ξ', '(-y)')
            elif self._sample_orientation == 4:
                formula_ = self.formula_so1.replace('y', '(-y)')
            elif self._sample_orientation == 5:
                formula_ = self.formula_so1
                formula_ = formula_.replace('x', 'ψ').replace('y', 'ξ')
                formula_ = formula_.replace('ψ', '(-y)').replace('ξ', '(-x)')
            elif self._sample_orientation == 6:
                formula_ = self.formula_so1
                formula_ = formula_.replace('x', 'ψ').replace('y', 'ξ')
                formula_ = formula_.replace('ψ', '(-y)').replace('ξ', '(x)')
            elif self._sample_orientation == 7:
                formula_ = self.formula_so1
                formula_ = formula_.replace('x', 'ψ').replace('y', 'ξ')
                formula_ = formula_.replace('ψ', '(y)').replace('ξ', '(x)')
            elif self._sample_orientation == 8:
                formula_ = self.formula_so1
                formula_ = formula_.replace('x', 'ψ').replace('y', 'ξ')
                formula_ = formula_.replace('ψ', '(y)').replace('ξ', '(-x)')
            self.formula = formula_
            ne_formula = numexpr.NumExpr(self.formula, signature)

            def ne_equation(x, y, z=None, wavelength=None,
                            incident_angle=self._incident_angle,
                            tilt_angle=self._tilt_angle,
                            sample_orientation=self._sample_orientation,
                            ne_formula=ne_formula):
                π = numpy.pi
                λ = wavelength
                η = self._incident_angle
                χ = self._tilt_angle
                ldict = locals()
                args = tuple(ldict[i] for i in ne_formula.input_names)
                return ne_formula(*args)

            self.equation = ne_equation
        else:
            self.equation = self._equation

    def __repr__(self):
        return f"""
{self.name}
Incident_angle={self.incident_angle}\u00b0
Tilt_angle={self.tilt_angle}\u00b0
Sample orientation={self.sample_orientation}
"""

    @property
    def incident_angle(self):
        return self._incident_angle

    @property
    def tilt_angle(self):
        return self._tilt_angle

    @property
    def sample_orientation(self):
        return self._sample_orientation

    def set_incident_angle(self, value:float):
        self._incident_angle = value
        self._update_ne_equation()

    def set_tilt_angle(self, value:float):
        self._tilt_angle = value
        self._update_ne_equation()

    def set_sample_orientation(self, value: int):
        self._sample_orientation = value
        self._update_ne_equation()


RADIAL_UNITS = {}
AZIMUTHAL_UNITS = {}
ANY_UNITS = {}

def register_radial_unit(name, scale=1, label=None, equation=None, formula=None,
                         center=None, corner=None, delta=None, short_name=None,
                         unit_symbol=None, positive=True, period=None):
    RADIAL_UNITS[name] = Unit(name, scale, label, equation, formula, center,
                              corner, delta, short_name, unit_symbol, positive, period)
    ANY_UNITS.update(RADIAL_UNITS)

def register_radial_fiber_unit(name, scale=1, label=None, equation=None, formula=None,
                               incident_angle=0.0, tilt_angle=0.0, sample_orientation=1,
                               center=None, corner=None, delta=None, short_name=None,
                               unit_symbol=None, positive=True, period=None):
    RADIAL_UNITS[name] = UnitFiber(name=name,
                                   scale=scale,
                                   label=label,
                                   equation=equation,
                                   formula=formula,
                                   incident_angle=incident_angle,
                                   tilt_angle=tilt_angle,
                                   sample_orientation=sample_orientation,
                                   center=center,
                                   corner=corner,
                                   delta=delta,
                                   short_name=short_name,
                                   unit_symbol=unit_symbol,
                                   positive=positive,
                                   period=period,
    )
    ANY_UNITS.update(RADIAL_UNITS)

def register_azimuthal_unit(name, scale=1, label=None, equation=None, formula=None,
                         center=None, corner=None, delta=None, short_name=None,
                         unit_symbol=None, positive=False, period=None):
    AZIMUTHAL_UNITS[name] = Unit(name, scale, label, equation, formula, center,
                                 corner, delta, short_name, unit_symbol, positive, period)
    ANY_UNITS.update(AZIMUTHAL_UNITS)

def register_azimuthal_fiber_unit(name, scale=1, label=None, equation=None, formula=None,
                                  incident_angle=0.0, tilt_angle=0.0, sample_orientation=1,
                                  center=None, corner=None, delta=None, short_name=None,
                                  unit_symbol=None, positive=False, period=None):
    AZIMUTHAL_UNITS[name] = UnitFiber(name=name, scale=scale, label=label,
                                      equation=equation, formula=formula,
                                      incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation,
                                      center=center, corner=corner, delta=delta,
                                      short_name=short_name, unit_symbol=unit_symbol,
                                      positive=positive, period=period,
    )
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


def eq_scattering_angle_vertical(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the vertical scattering angle (relative to direct beam axis), used for GI/Fiber diffraction

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: vertical exit angle in radians
    """
    return numpy.arctan2(y, numpy.sqrt(z ** 2 + x ** 2))


def eq_scattering_angle_horz(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the horizontal scattering angle (relative to direct beam axis), used for GI/Fiber diffraction

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: horizontal exit angle in radians
    """
    return numpy.arctan2(x, z)


def eq_exit_angle_vert(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the vertical exit angle in radians relative to the horizon (for thin films), used for GI/Fiber diffraction

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param wavelength: in meter
    :return: vertical exit angle in radians
    """
    rot_incident_angle = numpy.array([[1,0,0],
                                      [0,numpy.cos(incident_angle), numpy.sin(-incident_angle)],
                                      [0, numpy.sin(incident_angle), numpy.cos(incident_angle)]],
    )
    rot_tilt_angle = numpy.array([[numpy.cos(tilt_angle), numpy.sin(-tilt_angle), 0],
                                  [numpy.sin(tilt_angle), numpy.cos(tilt_angle), 0],
                                  [0, 0, 1]],
    )
    rotated_xyz = numpy.tensordot(rot_incident_angle, numpy.stack((x,y,z)), axes=1)
    xp, yp, zp = numpy.tensordot(rot_tilt_angle, rotated_xyz, axes=1)
    return numpy.arctan2(yp, numpy.sqrt(zp ** 2 + xp ** 2))


def eq_exit_angle_horz(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the horizontal exit angle in radians relative to the horizon (for thin films), used for GI/Fiber diffraction

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param wavelength: in meter
    :return: horizontal exit angle in radians
    """
    rot_incident_angle = numpy.array([[1,0,0],
                              [0,numpy.cos(incident_angle), numpy.sin(-incident_angle)],
                              [0, numpy.sin(incident_angle), numpy.cos(incident_angle)]],
    )
    rot_tilt_angle = numpy.array([[numpy.cos(tilt_angle), numpy.sin(-tilt_angle), 0],
                              [numpy.sin(tilt_angle), numpy.cos(tilt_angle), 0],
                              [0, 0, 1]],
    )
    rotated_xyz = numpy.tensordot(rot_incident_angle, numpy.stack((x,y,z)), axes=1)
    xp, yp, zp = numpy.tensordot(rot_tilt_angle, rotated_xyz, axes=1)
    return numpy.arctan2(xp, zp)


eq_exitangle = eq_exit_angle_vert


def q_lab_horz(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the horizontal component (y) of the scattering vector in the laboratory frame

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: horizontal scattering vector in inverse nm
    """
    scattering_angle_vertical = eq_scattering_angle_vertical(x=x, y=y, z=z, wavelength=wavelength)
    scattering_angle_horz = eq_scattering_angle_horz(x=x, y=y, z=z, wavelength=wavelength)
    return 2.0e-9 / wavelength * numpy.pi * numpy.cos(scattering_angle_vertical) * numpy.sin(scattering_angle_horz)


def q_lab_vert(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the vertical component (z) of the scattering vector in the laboratory frame

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: vertical scattering vector in inverse nm
    """
    scattering_angle_vertical = eq_scattering_angle_vertical(x=x, y=y, z=z, wavelength=wavelength)
    return 2.0e-9 / wavelength * numpy.pi * numpy.sin(scattering_angle_vertical)


def q_lab_beam(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the beam component (x) of the scattering vector in the laboratory frame

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: beam scattering vector in inverse nm
    """
    scattering_angle_vertical = eq_scattering_angle_vertical(x=x, y=y, z=z, wavelength=wavelength)
    scattering_angle_horz = eq_scattering_angle_horz(x=x, y=y, z=z, wavelength=wavelength)
    return 2.0e-9 / wavelength * numpy.pi * (numpy.cos(scattering_angle_vertical) * numpy.cos(scattering_angle_horz) - 1)


def q_lab(x, y, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the scattering vector in the laboratory frame (for GI/Fiber diffraction): no sample rotations are applied

    :param hpos: horizontal position, towards the center of the ring, from sample position
    :param vpos: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :return: scattering vector in the laboratory frame reference in inverse nm
    """
    return numpy.stack((q_lab_beam(x=x, y=y, z=z, wavelength=wavelength),
                        q_lab_horz(x=x, y=y, z=z, wavelength=wavelength),
                        q_lab_vert(x=x, y=y, z=z, wavelength=wavelength),
    ))


def rotation_tilt_angle(tilt_angle=0.0):
    """Calculates the rotation matrix along the x axis, (beam axis); represents the tilt angle rotation

    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: 3x3 rotation matrix along the beam axis
    """
    return numpy.array([[1 ,0 ,0],
                        [0, numpy.cos(tilt_angle), numpy.sin(tilt_angle)],
                        [0, numpy.sin((-1) * tilt_angle), numpy.cos(tilt_angle)],
                        ])


def rotation_incident_angle(incident_angle=0.0):
    """Calculates the rotation matrix along the y axis, (horizontal axis); represents the incident angle rotation

    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :return: 3x3 rotation matrix along the horizontal axis
    """
    return numpy.array([[numpy.cos(incident_angle), 0, numpy.sin(incident_angle)],
                        [0, 1, 0],
                        [numpy.sin((-1) * incident_angle), 0, numpy.cos(incident_angle)],
                        ])


def eq_qbeam(hpos, vpos, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the beam propagation direction in the sample frame (for GI/Fiber diffraction)
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param hpos: horizontal position, towards the center of the ring, from sample position
    :param vpos: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the beam propagation direction in inverse nm
    """
    return numpy.tensordot(numpy.dot(rotation_tilt_angle(tilt_angle=tilt_angle),
                                     rotation_incident_angle(incident_angle=incident_angle),
                                     )[0,:],
                           q_lab(x=hpos, y=vpos, z=z, wavelength=wavelength),
                           axes=(0,0),
    )


def eq_qhorz(hpos, vpos, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the horizontal direction in the sample frame (for GI/Fiber diffraction), towards the center of the ring
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the horizontal direction in inverse nm
    """
    return numpy.tensordot(numpy.dot(rotation_tilt_angle(tilt_angle=tilt_angle),
                                     rotation_incident_angle(incident_angle=incident_angle),
                                     )[1,:],
                           q_lab(x=hpos, y=vpos, z=z, wavelength=wavelength),
                           axes=(0,0),
    )


def eq_qvert(hpos, vpos, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the vertical direction in the sample frame (for GI/Fiber diffraction), to the roof
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param hpos: horizontal position, towards the center of the ring, from sample position
    :param vpos: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: component of the scattering vector along the vertical direction in inverse nm
    """
    return numpy.tensordot(numpy.dot(rotation_tilt_angle(tilt_angle=tilt_angle),
                                     rotation_incident_angle(incident_angle=incident_angle),
                                     )[2,:],
                           q_lab(x=hpos, y=vpos, z=z, wavelength=wavelength),
                           axes=(0,0),
    )


def q_sample(hpos, vpos, z, wavelength=None, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the scattering vector in the sample frame (for GI/Fiber diffraction) after incident angle and tilt angle rotations

    :param hpos: horizontal position, towards the center of the ring, from sample position
    :param vpos: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :return: scattering vector in the laboratory frame reference in inverse nm
    """
    return numpy.tensordot(numpy.dot(rotation_tilt_angle(tilt_angle=tilt_angle),
                                     rotation_incident_angle(incident_angle=incident_angle),
                                     ),
                           q_lab(x=hpos, y=vpos, z=z, wavelength=wavelength),
                           axes=(1,0),
    )


def rotate_sample_orientation(x, y, sample_orientation=1):
    """Rotates/Flips the axis x and y following the EXIF orientation values:
    https://sirv.com/help/articles/rotate-photos-to-be-upright/

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param int sample_orientation: 1-8, orientation of the fiber axis regarding the detector main axis

    Sample orientations
    1 - No changes are applied to the image
    2 - Image is mirrored (flipped horizontally)
    3 - Image is rotated 180 degrees
    4 - Image is rotated 180 degrees and mirrored
    5 - Image is mirrored and rotated 90 degrees counter clockwise
    6 - Image is rotated 90 degrees counter clockwise
    7 - Image is mirrored and rotated 90 degrees clockwise
    8 - Image is rotated 90 degrees clockwise
    """
    if sample_orientation == 1:
        hpos = x; vpos = y
    elif sample_orientation == 2:
        hpos = -x; vpos = y
    elif sample_orientation == 3:
        hpos = -x ; vpos = -y
    elif sample_orientation == 4:
        hpos = x ; vpos = -y
    elif sample_orientation == 5:
        hpos = -y ; vpos = -x
    elif sample_orientation == 6:
        hpos = -y; vpos = x
    elif sample_orientation == 7:
        hpos = y ; vpos = x
    elif sample_orientation == 8:
        hpos = y ; vpos = -x
    return hpos, vpos


def eq_qhorz_gi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the horizontal direction in the sample frame (for GI/Fiber diffraction), towards the center of the ring
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector along the horizontal direction in inverse nm
    """
    hpos, vpos = rotate_sample_orientation(x=x, y=y, sample_orientation=sample_orientation)
    return eq_qhorz(hpos=hpos, vpos=vpos, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qvert_gi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the vertical direction in the sample frame (for GI/Fiber diffraction), to the roof
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector along the vertical direction in inverse nm
    """
    hpos, vpos = rotate_sample_orientation(x=x, y=y, sample_orientation=sample_orientation)
    return eq_qvert(hpos=hpos, vpos=vpos, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qbeam_gi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector along the beam propagation direction in the sample frame (for GI/Fiber diffraction)
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector along the beam propagation direction in inverse nm
    """
    hpos, vpos = rotate_sample_orientation(x=x, y=y, sample_orientation=sample_orientation)
    return eq_qbeam(hpos=hpos, vpos=vpos, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)


def eq_qip(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector in the plane YZ in the sample frame (for GI/Fiber diffraction)
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    hpos, vpos = rotate_sample_orientation(x=x, y=y, sample_orientation=sample_orientation)
    q_sample_ = q_sample(hpos=hpos, vpos=vpos, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle)
    qsample_beam, qsample_horz = q_sample_[0], q_sample_[1]

    return numpy.sqrt(qsample_beam ** 2 + qsample_horz ** 2) * numpy.sign(qsample_horz)


def eq_qoop(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the component of the scattering vector in the vertical direction in the sample frame (for GI/Fiber diffraction)
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    return eq_qvert_gi(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)

def eq_q_total(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the total component of the scattering vector joining qip and qoop (for GI/Fiber diffraction)
        First, rotates the lab sample reference around the beam axis a tilt_angle value in radians,
         then rotates again around the horizontal axis using an incident angle value in radians

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    hpos, vpos = rotate_sample_orientation(x=x, y=y, sample_orientation=sample_orientation)
    return 4.0e-9 * numpy.pi * numpy.sin(eq_2th(hpos, vpos, z) / 2.0) / wavelength

def eq_chi_gi(x, y, z, wavelength, incident_angle=0.0, tilt_angle=0.0, sample_orientation=1):
    """Calculates the polar angle from the vertical axis (fiber or thin-film main axis)

    :param x: horizontal position, towards the center of the ring, from sample position
    :param y: vertical position, to the roof, from sample position
    :param z: distance from sample along the beam
    :param wavelength: in meter
    :param incident_angle: tilting of the sample towards the beam (analog to rot2): in radians
    :param tilt_angle: tilting of the sample orthogonal to the beam direction (analog to rot3): in radians
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    :return: component of the scattering vector in the plane YZ, in inverse nm
    """
    qoop = eq_qoop(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
    qip = eq_qip(x=x, y=y, z=z, wavelength=wavelength, incident_angle=incident_angle, tilt_angle=tilt_angle, sample_orientation=sample_orientation)
    return numpy.arctan2(qip, qoop)

formula_r = "sqrt(x * x + y * y)"
formula_2th = f"arctan2({formula_r}, z)"
formula_chi = "arctan2(y, x)"
formula_q = f"4.0e-9*π/λ*sin(0.5*{formula_2th})"
formula_d = f"0.5*λ/sin(0.5*{formula_2th})"
formula_d2 = f"(2.0e-9/λ*sin(0.5*{formula_2th}))**2"
formula_qx = f"4.0e-9*π/λ*sin(arctan2(x, z)/2.0)"  # TODO: wrong, fix me
formula_qy = f"4.0e-9*π/λ*sin(arctan2(y, z)/2.0)"  # TODO: wrong, fix me

formula_scattering_angle_vert = "arctan2(y, sqrt(z*z+x*x))"
formula_scattering_angle_horz = "arctan2(x,z)"
formula_x_rot_iangle = "x"
formula_y_rot_iangle = "(y*cos(η)-z*sin(η))"
formula_z_rot_iangle = "(y*sin(η)+z*cos(η))"
formula_x_rot_tangle = f"({formula_x_rot_iangle} * cos(χ) - {formula_y_rot_iangle} * sin(χ))"
formula_y_rot_tangle = f"({formula_x_rot_iangle} * sin(χ) + {formula_y_rot_iangle} * cos(χ))"
formula_z_rot_tangle = formula_z_rot_iangle
formula_exit_angle_vert = f"arctan2({formula_y_rot_tangle}, sqrt({formula_z_rot_tangle} * {formula_z_rot_tangle} + {formula_x_rot_tangle} * {formula_x_rot_tangle}))"
formula_exit_angle_horz = f"arctan2({formula_x_rot_tangle}, ({formula_z_rot_tangle}))"
formula_exit_angle = formula_scattering_angle_vert

formula_qbeam_lab = f"2.0e-9/λ*π*(cos({formula_scattering_angle_vert})*cos({formula_scattering_angle_horz}) - 1)"
formula_qhorz_lab = f"2.0e-9/λ*π*cos({formula_scattering_angle_vert})*sin({formula_scattering_angle_horz})"
formula_qvert_lab = f"2.0e-9/λ*π*sin({formula_scattering_angle_vert})"
formula_qbeam_rot = f"cos(η)*({formula_qbeam_lab})+sin(η)*({formula_qvert_lab})"
formula_qhorz_rot = f"cos(χ)*({formula_qhorz_lab})-sin(χ)*sin(η)*({formula_qbeam_lab})+sin(χ)*cos(η)*({formula_qvert_lab})"
formula_qvert_rot = f"-sin(χ)*({formula_qhorz_lab})-cos(χ)*sin(η)*({formula_qbeam_lab})+cos(χ)*cos(η)*({formula_qvert_lab})"
formula_qip = f"sqrt(({formula_qbeam_rot})**2+({formula_qhorz_rot})**2)*((({formula_qhorz_rot} > 0) * 2) - 1)"
formula_qoop = formula_qvert_rot
formula_qtot = formula_q
formula_chi_gi = f"arctan2(({formula_qip}), ({formula_qoop}))"

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

register_radial_fiber_unit("scattering_angle_vert_rad",
                     scale=1.0,
                     label=r"Vertical scattering angle (rad)",
                     formula=formula_scattering_angle_vert,
                     equation=eq_scattering_angle_vertical,
                     short_name="scatangle_vert",
                     unit_symbol="rad",
                     positive=False)

register_radial_fiber_unit("scattering_angle_horz_rad",
                     scale=1.0,
                     label=r"Horizontal scattering angle (rad)",
                     formula=formula_scattering_angle_horz,
                     equation=eq_scattering_angle_horz,
                     short_name="scatangle_horz",
                     unit_symbol="rad",
                     positive=False)

register_radial_fiber_unit("exit_angle_vert_rad",
                     scale=1.0,
                     label=r"Vertical exit angle (rad)",
                     formula=formula_exit_angle_vert,
                     equation=eq_exit_angle_vert,
                     short_name="exitangle_vert_rad",
                     unit_symbol="rad",
                     positive=False)

register_radial_fiber_unit("exit_angle_horz_rad",
                     scale=1.0,
                     label=r"Horizontal exit angle (rad)",
                     formula=formula_exit_angle_horz,
                     equation=eq_exit_angle_horz,
                     short_name="exitangle_horz_rad",
                     unit_symbol="rad",
                     positive=False)

register_radial_fiber_unit("exit_angle_vert_deg",
                     scale=180.0 / numpy.pi,
                     label=r"Vertical exit angle (deg)",
                     formula=formula_exit_angle_vert,
                     equation=eq_exit_angle_vert,
                     short_name="exitangle_vert",
                     unit_symbol="deg",
                     positive=False)

register_radial_fiber_unit("exit_angle_horz_deg",
                     scale=180.0 / numpy.pi,
                     label=r"Horizontal exit angle (deg)",
                     formula=formula_exit_angle_horz,
                     equation=eq_exit_angle_horz,
                     short_name="exitangle_horz",
                     unit_symbol="deg",
                     positive=False)

register_radial_fiber_unit("qxgi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_x$ ($nm^{-1}$)",
                     formula=formula_qhorz_rot,
                     equation=eq_qhorz_gi,
                     short_name="qxgi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qygi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_y$ ($nm^{-1}$)",
                     formula=formula_qvert_rot,
                     equation=eq_qvert_gi,
                     short_name="qygi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qzgi_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_z$ ($nm^{-1}$)",
                     formula=formula_qbeam_rot,
                     equation=eq_qbeam_gi,
                     short_name="qzgi",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qip_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{IP}$ ($nm^{-1}$)",
                     formula=formula_qip,
                     equation=eq_qip,
                     short_name="qip",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qoop_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{OOP}$ ($nm^{-1}$)",
                     formula=formula_qoop,
                     equation=eq_qoop,
                     short_name="qoop",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qtot_nm^-1",
                     scale=1.0,
                     label=r"Scattering vector $q_{xyz}$ ($nm^{-1}$)",
                     formula=formula_qtot,
                     equation=eq_q_total,
                     short_name="q",
                     unit_symbol="nm^{-1}",
                     positive=False)

register_radial_fiber_unit("qxgi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_x$ ($A^{-1}$)",
                     equation=eq_qhorz_gi,
                     short_name="qxgi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_fiber_unit("qygi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_y$ ($A^{-1}$)",
                     equation=eq_qvert_gi,
                     short_name="qygi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_fiber_unit("qzgi_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_z$ ($A^{-1}$)",
                     equation=eq_qbeam_gi,
                     short_name="qzgi",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_fiber_unit("qip_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{IP}$ ($A^{-1}$)",
                     formula=formula_qip,
                     equation=eq_qip,
                     short_name="qip",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_fiber_unit("qoop_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{OOP}$ ($A^{-1}$)",
                     formula=formula_qoop,
                     equation=eq_qoop,
                     short_name="qoop",
                     unit_symbol="A^{-1}",
                     positive=False)

register_radial_fiber_unit("qtot_A^-1",
                     scale=0.1,
                     label=r"Scattering vector $q_{xyz}$ ($A^{-1}$)",
                     equation=eq_q_total,
                     short_name="q",
                     unit_symbol="A^{-1}",
                     positive=False)

LENGTH_UNITS = {"m": Unit("m", scale=1., label=r"length $l$ ($m$)", positive=False),
                "cm": Unit("cm", scale=1e2, label=r"length $l$ ($cm$)", positive=False),
                "mm": Unit("mm", scale=1e3, label=r"length $l$ ($mm$)", positive=False),
                "micron": Unit("micron", scale=1e6, label=r"length $l$ ($\mu m$)", positive=False),
                "nm": Unit("nm", scale=1e9, label=r"length $l$ ($nm$)", positive=False),
                "A": Unit("A", scale=1e10, label=r"length $l$ ($\AA$)", positive=False),
                }
LENGTH_UNITS["µm"] = LENGTH_UNITS["micron"]

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
register_azimuthal_fiber_unit(name="chigi_rad",
                              scale=1.0,
                              label=r"Polar angle $\chi$ ($rad$)",
                              formula=formula_chi_gi,
                              equation=eq_chi_gi,
                              positive=False,
                              period=2.*pi)
register_azimuthal_fiber_unit(name="chigi_deg",
                              scale=180. / pi,
                              label=r"Polar angle $\chi$ ($^{o}$)",
                              formula=formula_chi_gi,
                              equation=eq_chi_gi,
                              positive=False,
                              period=360)

AZIMUTHAL_UNITS["qx_nm^-1"] = RADIAL_UNITS["qx_nm^-1"]
AZIMUTHAL_UNITS["qy_nm^-1"] = RADIAL_UNITS["qy_nm^-1"]
AZIMUTHAL_UNITS["qxgi_nm^-1"] = RADIAL_UNITS["qxgi_nm^-1"]
AZIMUTHAL_UNITS["qygi_nm^-1"] = RADIAL_UNITS["qygi_nm^-1"]
AZIMUTHAL_UNITS["qzgi_nm^-1"] = RADIAL_UNITS["qzgi_nm^-1"]
AZIMUTHAL_UNITS["qip_nm^-1"] = RADIAL_UNITS["qip_nm^-1"]
AZIMUTHAL_UNITS["qoop_nm^-1"] = RADIAL_UNITS["qoop_nm^-1"]
AZIMUTHAL_UNITS["qip_A^-1"] = RADIAL_UNITS["qip_A^-1"]
AZIMUTHAL_UNITS["qoop_A^-1"] = RADIAL_UNITS["qoop_A^-1"]

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
    elif isinstance(obj, (Unit, UnitFiber)):
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
Q_IP = Q_IP_NM = RADIAL_UNITS["qip_nm^-1"]
Q_OOP = Q_OOP_NM = RADIAL_UNITS["qoop_nm^-1"]
Q_IP_A = RADIAL_UNITS["qip_A^-1"]
Q_OOP_A = RADIAL_UNITS["qoop_A^-1"]
Q_TOT = RADIAL_UNITS["qtot_nm^-1"]

def get_unit_fiber(name, incident_angle:float =0.0, tilt_angle:float =0.0, sample_orientation=1):
    """Retrieves a unit instance for Grazing-Incidence/Fiber Scattering with updated incident and tilt angles
    The unit angles are in radians

    :param float incident_angle: projection angle of the beam in the sample. Its rotation axis is the fiber axis or the normal vector of the thin film
    :param float tilt angle: roll angle. Its rotation axis is orthogonal to the beam, the horizontal axis of the lab frame
    :param int sample_orientation: 1-8, orientation of the fiber axis according to EXIF orientation values (see def rotate_sample_orientation)
    """
    if name in RADIAL_UNITS:
        unit = copy.deepcopy(RADIAL_UNITS.get(name, None))
    elif name in AZIMUTHAL_UNITS:
        unit = copy.deepcopy(AZIMUTHAL_UNITS.get(name, None))
    else:
        unit = None

    if isinstance(unit, UnitFiber):
        unit.set_incident_angle(incident_angle)
        unit.set_tilt_angle(tilt_angle)
        unit.set_sample_orientation(sample_orientation)
    return unit

def parse_fiber_unit(unit, incident_angle=None, tilt_angle=None, sample_orientation=None):
    if isinstance(unit, str):
        unit = get_unit_fiber(name=unit)
    elif isinstance(unit, UnitFiber):
        pass
    else:
        unit = to_unit(unit)

    if not isinstance(unit, UnitFiber):
        raise Exception(f"{unit} cannot be used as a FiberUnit")

    if incident_angle is not None:
        unit.set_incident_angle(incident_angle)

    if tilt_angle is not None:
        unit.set_tilt_angle(tilt_angle)

    if sample_orientation is not None:
        unit.set_sample_orientation(sample_orientation)

    return unit
