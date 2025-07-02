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

Interesting formula:
https://geoweb.princeton.edu/archival/duffy/xtalgeometry.pdf
"""

from __future__ import annotations

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/07/2025"
__status__ = "production"

import os
import logging
import numpy
import itertools
from typing import Optional, List
from math import sin, asin, cos, sqrt, pi, ceil
import threading
from .utils import get_calibration_dir
from .utils.decorators import deprecated
from . import units
from .crystallography.cell import Cell
from .crystallography.space_groups import ReflectionCondition as Reflection_condition
from .crystallography.calibrant import Calibrant
from .crystallography.calibrant_factory import CALIBRANT_FACTORY


ALL_CALIBRANTS = CALIBRANT_FACTORY


def get_calibrant(calibrant_name: str, wavelength: float = None) -> Calibrant:
    """Returns a new instance of the calibrant by it's name.

    :param calibrant_name: Name of the calibrant
    :param wavelength: initialize the calibrant with the given wavelength (in m)
    """
    cal = CALIBRANT_FACTORY(calibrant_name)
    if wavelength:
        cal.set_wavelength(wavelength)
    return cal


def names() -> List[str]:
    """Returns the list of registred calibrant names."""
    return CALIBRANT_FACTORY.keys()
