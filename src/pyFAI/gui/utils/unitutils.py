# coding: utf-8
# /*##########################################################################
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ############################################################################*/

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "10/10/2024"

import numpy
import collections.abc

from pyFAI import units


def tthToRad(
    twoTheta: numpy.ndarray,
    unit: units.Unit,
    wavelength: float = None,
    directDist: float = None,
):
    """
    Convert a two theta angle from original `unit` to radian.

    The `directDist` argument can be extracted from an azimuthal integrator the
    following way:

    .. code-block:: python

        directDist = ai.getFit2D()["directDist"]

    :param unit: instance of pyFAI.units.Unit
    :param wavelength: wavelength in m
    :param directDist: distance from sample to beam-center on the detector in _mm_
    :param ai: instance of pyFAI.integrator.azimuthal.AzimuthalIntegrator
    """
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.abc.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit == units.TTH_RAD:
        return twoTheta
    elif unit == units.TTH_DEG:
        return numpy.deg2rad(twoTheta)
    elif unit == units.Q_A:
        if wavelength is None:
            raise AttributeError("wavelength has to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.0e-10 * numpy.pi)) * 2.0
    elif unit == units.Q_NM:
        if wavelength is None:
            raise AttributeError("wavelength has to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.0e-9 * numpy.pi)) * 2.0
    elif unit == units.R_MM:
        if directDist is None:
            raise AttributeError("directDist has to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / directDist)
    elif unit == units.R_M:
        if directDist is None:
            raise AttributeError("directDist has to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / (directDist * 0.001))
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)


def from2ThRad(twoTheta, unit, wavelength=None, directDist=None, ai=None):
    """
    Convert a two theta angle to this `unit`.

    The `directDist` argument can be extracted from an azimuthal integrator the
    following way:

    .. code-block:: python

        directDist = ai.getFit2D()["directDist"]

    :param unit: instance of pyFAI.units.Unit
    :param wavelength: wavelength in m
    :param directDist: distance from sample to beam-center on the detector in _mm_
    :param ai: instance of pyFAI.integrator.azimuthal.AzimuthalIntegrator
    """
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.abc.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit.space == "2th":
        return twoTheta * unit.scale
    elif unit.space == "q":
        q_nm = (4.0e-9 * numpy.pi / wavelength) * numpy.sin(0.5 * twoTheta)
        return q_nm * unit.scale
    elif unit.space == "r":
        if directDist is not None:
            beamCentre_m = directDist * 1e-3  # convert in m
        else:
            beamCentre_m = ai.getFit2D()["directDist"] * 1e-3  # convert in m
        return beamCentre_m * numpy.tan(twoTheta) * unit.scale
    elif unit.space == "d":
        q_m = (4.0 * numpy.pi / wavelength) * numpy.sin(0.5 * twoTheta)
        return (2 * numpy.pi / q_m) * unit.scale
    elif unit.space == "d*2":
        rec_d2_nm = (2e-9 / wavelength * numpy.sin(0.5 * twoTheta)) ** 2
        return rec_d2_nm * unit.scale
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)
