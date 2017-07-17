# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/02/2017"

from .DataModelAdaptor import DataModelAdaptor


class WavelengthToEnergyAdaptor(DataModelAdaptor):
    """Adapte a wavelength in angstrom to energy in KeV ."""

    angstrom = 1.00001501e-4
    """One angstrom in micro-meter."""

    hc = 1.2398
    """Product of h the Planck constant, and c the speed of light in vacuum."""

    def fromModel(self, value):
        """Returns energy in KeV from wavelength in angstrom"""
        if value is None:
            return None
        return self.hc / (value * 1000 * self.angstrom)

    def toModel(self, value):
        """Returns wavelength in angstrom from energy in KeV"""
        if value is None:
            return None
        return self.hc / (value * 1000 * self.angstrom)
