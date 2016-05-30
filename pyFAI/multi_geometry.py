# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, with_statement, division

__doc__ = """Module for treating simultaneously multiple detector configuration
             within a single integration"""
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/04/2016"
__status__ = "stable"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger("pyFAI.multi_geometry")
from .azimuthalIntegrator import AzimuthalIntegrator
from . import units
from .utils import EPS32
import threading
import numpy
error = None


class MultiGeometry(object):
    """
    This is an Azimuthal integrator containing multiple geometries (when
    the detector is on a goniometer arm)

    """

    def __init__(self, ais, unit="2th_deg",
                 radial_range=(0, 180), azimuth_range=(-180, 180),
                 wavelength=None, empty=0.0, chi_disc=180):
        """
        Constructor of the multi-geometry integrator
        @param ais: list of azimuthal integrators
        @param radial_range: common range for integration
        @param azimuthal_range: common range for integration
        @param empty: value for empty pixels
        @param chi_disc: if 0, set the chi_discontinuity at
        """
        self._sem = threading.Semaphore()
        self.abolute_solid_angle = None
        self.ais = [ ai if isinstance(ai, AzimuthalIntegrator) else AzimuthalIntegrator.sload(ai) for ai in ais]
        self.wavelength = None
        if wavelength:
            self.set_wavelength(wavelength)
        self.radial_range = tuple(radial_range[:2])
        self.azimuth_range = tuple(azimuth_range[:2])
        self.unit = units.to_unit(unit)
        self.abolute_solid_angle = None
        self.empty = empty
        if chi_disc == 0:
            for ai in self.ais:
                ai.setChiDiscAtZero()
        elif chi_disc == 180:
            for ai in self.ais:
                ai.setChiDiscAtPi()
        else:
            logger.warning("Unable to set the Chi discontinuity at %s" % chi_disc)

    def __repr__(self, *args, **kwargs):
        return "MultiGeometry integrator with %s geometries on %s radial range (%s) and %s azimuthal range (deg)" % \
            (len(self.ais), self.radial_range, self.unit, self.azimuth_range)

    def integrate1d(self, lst_data, npt=1800,
                    correctSolidAngle=True, polarization_factor=None,
                    monitors=None, all=False):
        """Perform 1D azimuthal integration

        @param lst_data: list of numpy array
        @param npt: number of points int the integration
        @param correctSolidAngle: correct for solid angle (all processing are then done in absolute solid angle !)
        @param polarization_factor: Apply polarization correction ? is None: not applies. Else provide a value from -1 to +1
        @param monitors: normalization monitors value (list of floats)
        @param all: return a dict with all information in it.
        @return: 2th/I or a dict with everything depending on "all"
        """
        if monitors is None:
            monitors = [1.0] * len(self.ais)
        sum = numpy.zeros(npt, dtype=numpy.float64)
        count = numpy.zeros(npt, dtype=numpy.float64)
        for ai, data, monitor in zip(self.ais, lst_data, monitors):
            res = ai.integrate1d(data, npt=npt,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 all=True)
            count += res["count"]
            sac = (ai.pixel1 * ai.pixel2 / monitor / ai.dist ** 2) if correctSolidAngle else 1.0 / monitor
            sum += res["sum"] * sac

        I = sum / numpy.maximum(count, EPS32 - 1)
        I[count <= (EPS32 - 1)] = self.empty

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

    def integrate2d(self, lst_data, npt_rad=1800, npt_azim=3600,
                    correctSolidAngle=True, polarization_factor=None,
                    monitors=None, all=False):
        """Performs 2D azimuthal integration of multiples frames, one for each geometry

        @param lst_data: list of numpy array
        @param npt: number of points int the integration
        @param correctSolidAngle: correct for solid angle (all processing are then done in absolute solid angle !)
        @param polarization_factor: Apply polarization correction ? is None: not applies. Else provide a value from -1 to +1
        @param monitors: normalization monitors value (list of floats)
        @param all: return a dict with all information in it.
        @return: I/2th/chi or a dict with everything depending on "all"
        """
        if monitors is None:
            monitors = [1.0] * len(self.ais)
        sum = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        count = numpy.zeros((npt_azim, npt_rad), dtype=numpy.float64)
        for ai, data, monitor in zip(self.ais, lst_data, monitors):
            res = ai.integrate2d(data, npt_rad=npt_rad, npt_azim=npt_azim,
                                 correctSolidAngle=correctSolidAngle,
                                 polarization_factor=polarization_factor,
                                 radial_range=self.radial_range,
                                 azimuth_range=self.azimuth_range,
                                 method="splitpixel", unit=self.unit, safe=True,
                                 all=True)
            count += res["count"]
            sac = (ai.pixel1 * ai.pixel2 / monitor / ai.dist ** 2) if correctSolidAngle else 1.0 / monitor
            sum += res["sum"] * sac

        I = sum / numpy.maximum(count, EPS32 - 1)
        I[count <= (EPS32 - 1)] = self.empty

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

    def set_wavelength(self, value):
        """
        Changes the wavelength of a group of azimuthal integrators
        """
        self.wavelength = float(value)
        for ai in self.ais:
            ai.set_wavelength(self.wavelength)
