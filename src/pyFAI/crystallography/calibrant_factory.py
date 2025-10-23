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

"""Calibrant factory

A module to build calibrants
"""

from __future__ import annotations

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/07/2025"
__status__ = "production"

import os
import logging
from ..utils import get_calibration_dir
from .calibrant import Calibrant
logger = logging.getLogger(__name__)


class BadCalibrantName(KeyError):
    pass


class CalibrantFactory:
    """Behaves like a dict but is actually a factory:

    Each time one retrieves an object it is a new genuine new calibrant (unmodified)
    """

    def __init__(self, basedir=None):
        """
        Constructor

        :param basedir: directory name where to search for the calibrants
        """
        if basedir is None:
            self.directory = get_calibration_dir()
        else:
            self.directory = basedir

        if not os.path.isdir(self.directory):
            logger.warning("No calibrant directory: %s", self.directory)
            self.all = {}
        else:
            if basedir is None:
                self.all = dict(
                    [
                        (os.path.splitext(i)[0], f"pyfai:{os.path.splitext(i)[0]}")
                        for i in os.listdir(self.directory)
                        if i.endswith(".D")
                    ]
                )
            else:
                self.all = dict(
                    [
                        (os.path.splitext(i)[0], os.path.join(self.directory, i))
                        for i in os.listdir(self.directory)
                        if i.endswith(".D")
                    ]
                )

    def __call__(self, calibrant_name:str) -> Calibrant:
        """Returns a new instance of a calibrant by it's name.

        :param calibrant_name: name of of the calibrant or filename
        :return: Freshly initialized calibrant (new instance each time)
        """
        if calibrant_name in self.all:
            return Calibrant(self.all[calibrant_name])
        raise BadCalibrantName(f"Calibrant '{calibrant_name}' is not registered !")

    def get(self, what: str, notfound=None):
        if what in self.all:
            return Calibrant(self.all[what])
        else:
            return notfound

    def __contains__(self, k: str):
        return k in self.all

    def __repr__(self):
        return "Calibrants available: %s" % (", ".join(list(self.all.keys())))

    def __len__(self):
        return len(self.all)

    def keys(self):
        return list(self.all.keys())

    def values(self):
        return [Calibrant(i) for i in self.all.values()]

    def items(self):
        return [(i, Calibrant(j)) for i, j in self.all.items()]

    has_key = __contains__


CALIBRANT_FACTORY = CalibrantFactory()
"""Default calibration factory provided by the library."""
