#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""Calibrant

A module containing classical calibrant and also tools to generate d-spacing.

This class is mostly empty and is left for compatibility purposes.
It should be DEPRECATED once modification related to crystallography are done
and tutorial updated.
"""

from __future__ import annotations

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "08/10/2025"
__status__ = "production"

from typing import List
from .crystallography.cell import Cell
from .crystallography.space_groups import ReflectionCondition
from .crystallography.calibrant import Calibrant
from .crystallography.calibrant_factory import CALIBRANT_FACTORY

__all__ = ["ALL_CALIBRANTS", "get_calibrant", "names",
            CALIBRANT_FACTORY, Calibrant, Cell, ReflectionCondition]

ALL_CALIBRANTS = CALIBRANT_FACTORY


def get_calibrant(calibrant_name: str, wavelength: float = None) -> Calibrant:
    """Returns a new instance of the calibrant by it's name.

    :param calibrant_name: Name of the calibrant
    :param wavelength: initialize the calibrant with the given wavelength (in m)
    """
    cal = CALIBRANT_FACTORY(calibrant_name)
    if wavelength:
        cal.wavelength = wavelength
    return cal


def names() -> List[str]:
    """Returns the list of registered calibrant names."""
    return CALIBRANT_FACTORY.keys()
