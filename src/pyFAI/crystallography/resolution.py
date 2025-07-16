#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Resolution functions

This module contains two classes to calculate the peak width as function of the scattering angle:
* Caglioti
* Langford
* Chernyshov

"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/07/2025"
__status__ = "production"


from collections.abc import Iterable
from ..containers import dataclass
from ..units import Unit, to_unit, TTH_DEG
import numpy


LN2 = numpy.log(2.0)


class _ResolutionFunction:
    """Abstract class, mises constructor and fwhm2"""
    def fwhm2(self, tth):
        """
        :param tth: 2theta value or array
        :return: array of the same shape containing the full-with at half maximum of the peak(s) in radians
        """
        raise NotImplementedError("`_ResolutionFunction`is an abstract class !")

    def sigma2(self, tth):
        "Assumes a normal distribution"
        return self.fwhm2(tth)/(8.0*LN2)

    def fwhm(self, tth):
        "Full Width at Half Maximum in radians"
        return numpy.sqrt(self.fwhm2(tth))

    def sigma(self, tth):
        "Assumes a normal distribution, standard deviation in radians"
        return numpy.sqrt(self.sigma2(tth))


@dataclass
class Constant(_ResolutionFunction):
    """Dummy constant resolution function (with units)"""
    C:float
    unit: Unit = TTH_DEG

    def __repr__(self):
        return f"Constant({self.C}, {self.unit})"

    def fwhm2(self, tth):
        """Calculate the full-with at half maximum of the peak(s) squared

        :param tth: 2theta value or array of them, in radians
        :return: array of the same shape as tth
        """
        C2 = (self.C/to_unit(self.unit).scale)**2
        if isinstance(tth, Iterable):
            return numpy.zeros_like(tth) + C2
        else:
            return C2


@dataclass
class Caglioti(_ResolutionFunction):
    """Caglioti, G., Paoletti, A. & Ricci, F. (1958). Nucl. Instrum. 3, 223-228."""
    U:float
    V:float
    W:float

    def __repr__(self):
        return f"Caglioti({self.U}, {self.V}, {self.W})"

    def fwhm2(self, tth):
        """Calculate the full-with at half maximum of the peak(s) squared
        :param tth: 2theta value or array in radians
        :return: array of the same shape as tth
        """
        t_th = numpy.tan(0.5 * tth)
        return self.U * t_th**2 + self.V * t_th + self.W


@dataclass
class Langford(_ResolutionFunction):
    """https://doi.org/10.1016/0146-3535(87)90018-9"""
    A:float
    B:float
    C:float
    D:float

    def __repr__(self):
        return f"Langford({self.A}, {self.B}, {self.C}, {self.D})"

    def fwhm2(self, tth):
        """Calculate the full-with at half maximum of the peak(s) squared
        :param tth: 2theta value or array in radians
        :return: array of the same shape as tth
        """
        stth = numpy.sin(tth)
        t_th2 = numpy.tan(0.5*tth)**2
        return self.A + self.B * stth * stth + self.C*t_th2 + self.D/t_th2


@dataclass
class Chernyshov(_ResolutionFunction):
    """https://doi.org/10.1107/S2053273321007506"""
    A:float
    B:float
    C:float

    def __repr__(self):
        return f"Chernyshov({self.A}, {self.B}, {self.C})"

    def fwhm2(self, tth):
        """Calculate the full-with at half maximum of the peak(s) squared
        :param tth: 2theta value or array in radians
        :return: array of the same shape as tth
        """
        ctth = numpy.cos(tth)
        return self.A * ctth**2 + self.B * ctth + self.C
