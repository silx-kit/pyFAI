# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2025 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""Module to read and sometimes write calibration files"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/07/2025"
__status__ = "development"
__docformat__ = "restructuredtext"


import os
from ..containers import dataclass


@dataclass
class Reflection:
    "Represent a familly of Miller plans"

    d_spacing: float = None
    intensity: float = None
    hkl: tuple = tuple()
    multiplicity: int = None


def read_dif(filename: str):
    """Read a dif-file as provided by the American Mineralogist database

        https://rruff.geo.arizona.edu/AMS/amcsd.php
        https://www.rruff.net/amcsd/

    :param filename: name of the file as string
    :return: list of reflections ordered by decreasing d-spacing
    """
    raw = []
    with open(filename) as fd:
        for line in fd:
            raw.append(line.strip())
    reflections = []
    started = False
    for line in raw:
        if line.startswith("2-THETA") and not started:
            started = True
            continue
        if started:
            if line.startswith("=" * 10):
                break
            words = line.split()
            if len(words) >= 7:
                reflections.append(
                    Reflection(
                        float(words[2]),
                        float(words[1]),
                        (int(words[3]), int(words[4]), int(words[5])),
                        int(words[6]),
                    )
                )
    reflections.sort(key=lambda r: r.d_spacing, reverse=True)
    return reflections
