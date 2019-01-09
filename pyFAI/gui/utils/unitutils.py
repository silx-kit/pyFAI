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
__date__ = "03/01/2019"


import numpy
import collections

from pyFAI import units


def tthToRad(twoTheta, unit, wavelength=None, directDist=None):
    """
    Convert a two theta angle from original `unit` to radian.

    `directDist = ai.getFit2D()["directDist"]`
    """
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit == units.TTH_RAD:
        return twoTheta
    elif unit == units.TTH_DEG:
        return numpy.deg2rad(twoTheta)
    elif unit == units.Q_A:
        if wavelength is None:
            raise AttributeError("wavelength have to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.e-10 * numpy.pi)) * 2.0
    elif unit == units.Q_NM:
        if wavelength is None:
            raise AttributeError("wavelength have to be specified")
        return numpy.arcsin((twoTheta * wavelength) / (4.e-9 * numpy.pi)) * 2.0
    elif unit == units.R_MM:
        if directDist is None:
            raise AttributeError("directDist have to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / directDist)
    elif unit == units.R_M:
        if directDist is None:
            raise AttributeError("directDist have to be specified")
        # GF: correct formula?
        return numpy.arctan(twoTheta / (directDist * 0.001))
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)


def from2ThRad(twoTheta, unit, wavelength=None, directDist=None, ai=None):
    if isinstance(twoTheta, numpy.ndarray):
        pass
    elif isinstance(twoTheta, collections.Iterable):
        twoTheta = numpy.array(twoTheta)

    if unit == units.TTH_DEG:
        return numpy.rad2deg(twoTheta)
    elif unit == units.TTH_RAD:
        return twoTheta
    elif unit == units.Q_A:
        return (4.e-10 * numpy.pi / wavelength) * numpy.sin(.5 * twoTheta)
    elif unit == units.Q_NM:
        return (4.e-9 * numpy.pi / wavelength) * numpy.sin(.5 * twoTheta)
    elif unit == units.R_MM:
        # GF: correct formula?
        if directDist is not None:
            beamCentre = directDist
        else:
            beamCentre = ai.getFit2D()["directDist"]  # in mm!!
        return beamCentre * numpy.tan(twoTheta)
    elif unit == units.R_M:
        # GF: correct formula?
        if directDist is not None:
            beamCentre = directDist
        else:
            beamCentre = ai.getFit2D()["directDist"]  # in mm!!
        return beamCentre * numpy.tan(twoTheta) * 0.001
    else:
        raise ValueError("Converting from 2th to unit %s is not supported", unit)
