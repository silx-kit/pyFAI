# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2025-2026 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "13/01/2026"
__status__ = "development"

from math import sin, cos, pi
from dataclasses import dataclass
import numpy
from scipy import ndimage, optimize
from ..control_points import ControlPoints
from ..ext import _geometry
from ..io.ponifile import PoniFile
from ..third_party.classproperties import classproperty


module_d = numpy.dtype(
    [
        ("d0", numpy.float64),
        ("d1", numpy.float64),
        ("ring", numpy.int32),
        ("module", numpy.int32),
    ]
)


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

    @classproperty
    def nb_param(cls):
        return len(cls.__dataclass_fields__)


@dataclass
class PoniParam:
    dist: float = 0.0
    poni1: float = 0.0
    poni2: float = 0.0
    rot1: float = 0.0
    rot2: float = 0.0
    # rot3:float=0.0
    # wavelength:float=0.0

    @classproperty
    def nb_param(self):
        return len(self.__dataclass_fields__)


class SingleModule:
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
        return (
            f"Module centered at ({self.center[0, 0]:.1f}, {self.center[1, 0]:.1f})"
            + (", fixed." if self.fixed else ".")
        )

    def calc_bounding_box(self):
        d0, d1 = numpy.where(self.mask)
        d0m = d0.min()
        d0M = d0.max()
        d1m = d1.min()
        d1M = d1.max()
        self.center = numpy.atleast_2d([0.5 * (d0M + d0m + 1), 0.5 * (d1M + d1m + 1)]).T
        self.bounding_box = (slice(d0m, d0M + 1), slice(d1m, d1M + 1))
        return self.bounding_box

    def calc_displacement_map(self, d1=None, d2=None, param=None):
        if d1 is None and d2 is None:
            full_detector = True
            p1, p2, _ = self.parent_detector.calc_cartesian_positions()
            d1 = p1 / self.parent_detector.pixel1
            d2 = p2 / self.parent_detector.pixel2
            mp1 = d1[self.mask]
            mp2 = d2[self.mask]
        else:
            full_detector = False
            mp1 = d1
            mp2 = d2

        param = param or self.param

        mpc = numpy.vstack((mp1.ravel(), mp2.ravel()))
        if not self.fixed:
            self.center
            mpc -= self.center
            rot = param.rot
            c, s = cos(rot), sin(rot)
            rotm = numpy.array([[c, -s], [s, c]])
            mpc = (
                numpy.dot(rotm, mpc)
                + self.center
                + numpy.atleast_2d([param.d0, param.d1]).T
            )
        if full_detector:
            mshape = mp1.shape
            p1[self.mask] = mpc[0].reshape(mshape)
            p2[self.mask] = mpc[1].reshape(mshape)
        else:
            p1, p2 = mpc
        return p1, p2

    def calc_position(self, d1=None, d2=None, param=None):
        d1, d2 = self.calc_displacement_map(d1, d2, param)
        return d1 * self.parent_detector.pixel1, d2 * self.parent_detector.pixel2


class MultiModule:
    """Split a detector in several modules"""

    def __init__(self):
        self.modules = {}  # this is contains all of modules
        self.lmask = None
        self.detector = None
        self.nb_modules = 0

    def __repr__(self):
        return f"MultiModule with {self.nb_modules} modules:\n" + "\n".join(
            f"  {i:2d}: {j}" for i, j in self.modules.items()
        )

    def build_labels(self):
        self.lmask, self.nb_modules = ndimage.label(
            numpy.logical_not(self.detector.mask)
        )

    @classmethod
    def from_detector(cls, detector):
        """Alternative constructor

        :param detector: ensure the mask is definied"""
        self = cls()
        if detector.mask is None:
            raise RuntimeError("`detector` must provide an actual mask")
        self.detector = detector
        self.build_labels()
        for l in range(1, self.nb_modules + 1):  # noqa: E741
            self.modules[l] = SingleModule(detector, self.lmask, index=l, fixed=False)
        return self

    @property
    def shape(self):
        return self.detector.shape

    def calc_displacement_map(self):
        p1, p2, _ = self.detector.calc_cartesian_positions()
        p1 /= self.detector.pixel1
        p2 /= self.detector.pixel2

        for l in range(1, self.nb_modules + 1):  # noqa: E741
            m = self.modules[l]
            mp1, mp2 = m.calc_displacement_map()
            p1[m.mask] = mp1[m.mask]
            p2[m.mask] = mp2[m.mask]

        return p1, p2

    @property
    def free_modules(self):
        return sum(not m.fixed for m in self.modules.values())


class MultiModuleRefinement(MultiModule):
    def __init__(self):
        super().__init__()
        self.modulated_points = {}  # key: npt filename, value record array with coordinates, ring & module
        self.calibrants = {}  # contains the different calibrant objects for each control-point file
        self._q_theo = {}
        self.ponis = {}  # relative to control-point files #Unused ?

    def calc_cp_positions(self, param=None, key=None, center=True):
        """Calculate the physical position for control points of a given registered calibrant"""
        mcp = self.modulated_points[key]
        p1 = mcp.d0.copy()
        p2 = mcp.d1.copy()
        param_idx = 0
        center = 0.5 if center else 0
        for l in range(1, self.nb_modules + 1):  # noqa: E741
            m = self.modules[l]
            mask = mcp.module == l
            valid = mcp[mask]
            sub_param = (
                None
                if param is None or m.fixed
                else ModuleParam(*param[3 * param_idx : 3 * (param_idx + 1)])
            )
            param_idx += 0 if m.fixed else 1
            mp1, mp2 = m.calc_position(
                d1=valid.d0 + center, d2=valid.d1 + center, param=sub_param
            )
            p1[mask] = mp1
            p2[mask] = mp2
        return p1, p2

    def print_control_points_per_module(self, filename):
        if filename not in self.modulated_points:
            print(f"No control-point file named {filename}. Did you load it ?")
        else:
            print(filename, ":", self.calibrants.get(filename))
            modulated_cp = self.modulated_points[filename]
            for l in range(1, self.nb_modules + 1):  # noqa: E741
                print(l, (modulated_cp.module == l).sum())

    def load_control_points(self, filename, poni=None, verbose=False):
        """
        :param filename: file with control points
        :param poni: file with the (uncorrected) detector position
        :param verbose: set to True to print out the number of control points per module
        """
        cp = ControlPoints(filename)
        self.calibrants[filename] = cp.calibrant
        if poni:
            self.ponis[filename] = PoniFile(poni)
        # build modulated list of control points
        d0 = []
        d1 = []
        ring = []
        modules = []
        for i in cp.getList():
            d0.append(i[0])
            d1.append(i[1])
            ring.append(i[2])
            modules.append(0)
        modulated_cp = numpy.rec.fromarrays((d0, d1, ring, modules), dtype=module_d)
        linear = numpy.round(modulated_cp.d0).astype(numpy.int32) * self.shape[
            -1
        ] + numpy.round(modulated_cp.d1).astype(numpy.int32)
        modulated_cp.module = self.lmask.ravel()[linear]
        self.modulated_points[filename] = modulated_cp
        if verbose:
            self.print_control_points_per_module(filename)

    def init_q_theo(self, force=False):
        if force or not self._q_theo:
            self._q_theo = {
                key: 20.0
                * pi
                / numpy.array(calibrant.dspacing)[self.modulated_points[key].ring]
                for key, calibrant in self.calibrants.items()
            }

    def residu(self, param=None):
        """Calculate the delta_q value between the expected ring position and the actual one"""
        if not self._q_theo:
            self.init_q_theo()
        module_param = param[
            : ModuleParam.nb_param * sum(not m.fixed for m in self.modules.values())
        ]
        delta = []
        for idx, (key, calibrant) in enumerate(self.calibrants.items()):
            # print(key)
            tmp_e = self._q_theo[
                key
            ]  # This is the theoritical q_value for the given ring (in nm^-1)
            # print("exp", len(tmp_e), tmp_e)
            dp1, dp2 = self.calc_cp_positions(param=module_param, key=key)
            # print("dp", len(dp1), len(dp2))

            start_idx = (
                ModuleParam.nb_param * self.free_modules + idx * PoniParam.nb_param
            )
            end_idx = start_idx + PoniParam.nb_param
            poni_param = PoniParam(*param[start_idx:end_idx])
            # print(poni_param)
            tmp_c = _geometry.calc_q(
                poni_param.dist,
                poni_param.rot1,
                poni_param.rot2,
                0.0,
                dp1 - poni_param.poni1,
                dp2 - poni_param.poni2,
                calibrant.wavelength,
            )
            # print("residu", tmp_e, tmp_c)
            delta.append(tmp_c - tmp_e)
        return numpy.concatenate(delta)

    @property
    def nb_param(self):
        """Number of parameters for the refinement"""
        free = sum(not m.fixed for m in self.modules.values())
        return free * ModuleParam.nb_param + PoniParam.nb_param * len(self.calibrants)

    def init_param(self):
        """Generate the numpy array with all parameters"""
        param = numpy.zeros(self.nb_param)
        idx = 0
        for m in self.modules.values():
            if m.fixed:
                continue
            for i, n in enumerate(ModuleParam.__dataclass_fields__, start=idx):
                param[i] = m.param.__getattribute__(n)
            idx += ModuleParam.nb_param
        for p in self.ponis.values():
            for i, n in enumerate(PoniParam.__dataclass_fields__, start=idx):
                param[i] = p.__getattribute__(n)
            idx += PoniParam.nb_param
        return param

    def print_param(self, param):
        idx = 0
        for i, m in self.modules.items():
            if m.fixed:
                print(f"module #{i:2d}: Fixed")
            else:
                res = f"module #{i:2d}:"
                for i, n in enumerate(ModuleParam.__dataclass_fields__, start=idx):
                    res += f" {n:5s}= {param[i]},"
                idx += ModuleParam.nb_param
                print(res)
        for p in self.ponis:
            res = f"{p}:"
            for i, n in enumerate(PoniParam.__dataclass_fields__, start=idx):
                res += f" {n:5s}= {param[i]:6f},"
            print(res)
            idx += PoniParam.nb_param

    def cost(self, param):
        delta = self.residu(param)
        return numpy.dot(delta, delta)

    def refine(self, param, method="SLSQP"):
        method = "Nelder-Mead" if method.lower() == "simplex" else method
        return optimize.minimize(self.cost, param, method=method)
