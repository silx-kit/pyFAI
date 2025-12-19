# !/usr/bin/env python
# -*- coding: utf-8 -*-
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
#

"""Multi-module detectors:

This module contains some helper function to define a detector from several modules
and later-on refine this module position from powder diffraction data
as demonstrated in https://doi.org/10.3390/cryst12020255
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/12/2025"
__status__ = "development"

from math import sin, cos
from dataclasses import dataclass
import numpy
from scipy import ndimage


# Those are the optimizable parameters ... 2 translations and one rotation.
@dataclass
class ModuleParam:
    d0: float = 0.0
    d1: float = 0.0
    rot: float = 0.0

    def set(self, iterable):
        self.d0, self.d1, self.rot = iterable[:3]

    def get(self):
        return (self.d0, self.d1, self.rot)


class SingleModule:
    """This represents one module out of a multi-module detector"""

    def __init__(self, detector, mask, index=None, fixed=False):
        self.parent_detector = detector
        self.parent_index = index
        if (index is not None) and index <= mask.max():
            self.mask = mask == index
        else:
            self.mask = mask
        self.fixed = False
        self.param = ModuleParam()
        self.center = None
        self.bounding_box = None
        self.calc_bounding_box()

    def __repr__(self):
        return f"Module centered at ({self.center[0]:.1f}, {self.center[1]:.1f})" + (
            "fixed" if self.fixed else ""
        )

    def calc_bounding_box(self):
        d0, d1 = numpy.where(self.mask)
        d0m = d0.min()
        d0M = d0.max()
        d1m = d1.min()
        d1M = d1.max()
        self.center = (0.5 * (d0M + d0m + 1), 0.5 * (d1M + d1m + 1))
        self.bounding_box = (slice(d0m, d0M + 1), slice(d1m, d1M + 1))
        return self.bounding_box

    def calc_displacement_map(self):
        p1, p2, _ = self.parent_detector.calc_cartesian_positions()
        p1 /= self.parent_detector.pixel1
        p2 /= self.parent_detector.pixel2
        mp1 = p1[self.mask]
        mp2 = p2[self.mask]
        mpc = numpy.vstack((mp1.ravel() - self.center[0], mp2.ravel() - self.center[1]))
        rot = self.param.rot
        c, s = cos(rot), sin(rot)
        rotm = numpy.array([[c, -s], [s, c]])
        numpy.dot(rotm, mpc, out=mpc)
        mpc += numpy.array(
            [[self.center[0] + self.param.d0], [self.center[1] + self.param.d1]]
        )
        mshape = mp1.shape
        p1[self.mask] = mpc[0].reshape(mshape)
        p2[self.mask] = mpc[1].reshape(mshape)
        return p1, p2


class MultiModule:
    """Split a detector in several modules"""

    def __init__(self):
        self.modules = {}  # this is contains all of modules
        self.lmask = None
        self.nlabels = 0
        self.detector = None

    def __repr__(self):
        return f"MultiModule with {self.nlabels} modules"

    def build_labels(self):
        self.lmask, self.nlabels = ndimage.label(numpy.logical_not(self.detector.mask))

    @classmethod
    def from_detector(cls, detector):
        """Alternative constructor

        :param detector: ensure the mask is definied"""
        self = cls()
        if detector.mask is None:
            raise RuntimeError("`detector` must provide an actual mask")
        self.detector = detector
        self.build_labels()
        for lbl in range(1, self.nlabels + 1):
            self.modules[lbl] = SingleModule(detector, self.lmask, index=lbl, fixed=False)
        return self

    @property
    def shape(self):
        return self.detector.shape

    def calc_displacement_map(self):
        p1, p2, _ = self.detector.calc_cartesian_positions()
        p1 /= self.detector.pixel1
        p2 /= self.detector.pixel2

        for lbl in range(1, self.nlabels + 1):
            m = self.modules[lbl]
            mp1, mp2 = m.calc_displacement_map()
            p1[m.mask] = mp1[m.mask]
            p2[m.mask] = mp2[m.mask]

        return p1, p2
