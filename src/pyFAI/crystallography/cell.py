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

"""Crystallographic cell

A module with the Cell class defining a crystallographic cell

Interesting formula:
https://geoweb.princeton.edu/archival/duffy/xtalgeometry.pdf
"""

from __future__ import annotations

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2025"
__status__ = "production"

import os
import logging
import numpy
import itertools
from math import sin, cos, sqrt, pi, ceil
from ..io.calibrant_config import CalibrantConfig, Miller, Reflection
from ..utils.decorators import deprecated

logger = logging.getLogger(__name__)


class Cell:
    """
    This is a cell object, able to calculate the volume and d-spacing according to formula from:

    http://geoweb3.princeton.edu/research/MineralPhy/xtalgeometry.pdf
    """

    lattices = [
        "cubic",
        "tetragonal",
        "hexagonal",
        "rhombohedral",
        "orthorhombic",
        "monoclinic",
        "triclinic",
    ]
    types = {
        "P": "Primitive",
        "I": "Body centered",
        "F": "Face centered",
        "C": "Side centered",
        "R": "Rhombohedral",
    }

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        alpha: float = 90.0,
        beta: float = 90.0,
        gamma: float = 90.0,
        lattice: str = "triclinic",
        lattice_type: str = "P",
    ):
        """Constructor of the Cell class:

        Crystalographic units are Angstrom for distances and degrees for angles !

        :param a,b,c: unit cell length in Angstrom
        :param alpha, beta, gamma: unit cell angle in degrees
        :param lattice: "cubic", "tetragonal", "hexagonal", "rhombohedral", "orthorhombic", "monoclinic", "triclinic"
        :param lattice_type: P, I, F, C or R
        """
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lattice = lattice if lattice in self.lattices else "triclinic"

        self._volume = None
        self.S11 = None
        self.S12 = None
        self.S13 = None
        self.S22 = None
        self.S23 = None
        self.selection_rules = []
        "contains a list of functions returning True(allowed)/False(forbidden)/None(unknown), see space_groups.py"
        self._type = None
        self.type = lattice_type

    def __repr__(self, *args, **kwargs):
        return (
            f"{self.types[self.type]} {self.lattice} cell a={self.a:.4f} b={self.b:.4f} c={self.c:.4f}\N{Latin Capital Letter a with Ring Above} "
            f"\N{GREEK SMALL LETTER ALPHA}={self.alpha:.3f} \N{GREEK SMALL LETTER BETA}={self.beta:.3f} \N{GREEK SMALL LETTER GAMMA}={self.gamma:.3f}\N{DEGREE SIGN}"
        )

    @classmethod
    def cubic(cls, a, lattice_type="P"):
        """Factory for cubic lattices

        :param a: unit cell length
        """
        a = float(a)
        self = cls(a, a, a, 90, 90, 90, lattice="cubic", lattice_type=lattice_type)
        return self

    @classmethod
    def tetragonal(cls, a, c, lattice_type="P"):
        """Factory for tetragonal lattices

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(
            a, a, float(c), 90, 90, 90, lattice="tetragonal", lattice_type=lattice_type
        )
        return self

    @classmethod
    def orthorhombic(cls, a, b, c, lattice_type="P"):
        """Factory for orthorhombic lattices

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        """
        self = cls(
            float(a),
            float(b),
            float(c),
            90,
            90,
            90,
            lattice="orthorhombic",
            lattice_type=lattice_type,
        )
        return self

    @classmethod
    def hexagonal(cls, a, c, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param c: unit cell length
        """
        a = float(a)
        self = cls(
            a, a, float(c), 90, 90, 120, lattice="hexagonal", lattice_type=lattice_type
        )
        return self

    @classmethod
    def monoclinic(cls, a, b, c, beta, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param b: unit cell length
        :param c: unit cell length
        :param beta: unit cell angle
        """
        self = cls(
            float(a),
            float(b),
            float(c),
            90,
            float(beta),
            90,
            lattice_type=lattice_type,
            lattice="monoclinic",
        )
        return self

    @classmethod
    def rhombohedral(cls, a, alpha, lattice_type="P"):
        """Factory for hexagonal lattices

        :param a: unit cell length
        :param alpha: unit cell angle
        """
        a = float(a)
        alpha = float(alpha)
        self = cls(
            a,
            a,
            a,
            alpha,
            alpha,
            alpha,
            lattice="rhombohedral",
            lattice_type=lattice_type,
        )
        return self

    @classmethod
    def diamond(cls, a):
        """Factory for Diamond type FCC like Si and Ge

        :param a: unit cell length
        """
        self = cls.cubic(a, lattice_type="F")
        self.selection_rules.append(
            lambda h, k, l: not (  # noqa: E741
                (h % 2 == 0)
                and (k % 2 == 0)
                and (l % 2 == 0)
                and ((h + k + l) % 4 != 0)
            )
        )
        return self

    @property
    def volume(self):
        if self._volume is None:
            self._volume = self.a * self.b * self.c
            if self.lattice not in ["cubic", "tetragonal", "orthorhombic"]:
                deg2rad = pi / 180.0
                cosa = cos(self.alpha * deg2rad)
                cosb = cos(self.beta * deg2rad)
                cosg = cos(self.gamma * deg2rad)
                self._volume *= sqrt(
                    1.0 - cosa*cosa - cosb*cosb - cosg*cosg + 2 * cosa * cosb * cosg
                )
        return self._volume

    @property
    def type(self):
        return self._type
    @type.setter
    def type(self, lattice_type):
        self._type = lattice_type if lattice_type in self.types else "P"
        self.selection_rules = [lambda h, k, l: not (h == 0 and k == 0 and l == 0)]  # noqa: E741
        if self._type == "I":
            self.selection_rules.append(lambda h, k, l: (h + k + l) % 2 == 0)  # noqa: E741
        if self._type == "F":
            self.selection_rules.append(
                lambda h, k, l: (h % 2 + k % 2 + l % 2) in (0, 3)  # noqa: E741
            )
        if self._type == "R":
            self.selection_rules.append(lambda h, k, l: ((h - k + l) % 3 == 0))  # noqa: E741

    get_type = deprecated(type.fset, reason="property", replacement="type", since_version="2025.07")
    set_type = deprecated(type.fset, reason="property", replacement="type", since_version="2025.07")

    def d(self, hkl:tuple|Miller) -> float:
        """
        Calculate the actual d-spacing for a 3-tuple of integer representing a
        family of Miller plans

        :param hkl: 3-tuple of integers
        :return: the inter-planar distance in Angstrom
        """
        h, k, l = hkl  # noqa: E741
        deg2rad = pi / 180.0
        if self.lattice in ["cubic", "tetragonal", "orthorhombic"]:
            invd2 = (h / self.a) ** 2 + (k / self.b) ** 2 + (l / self.c) ** 2
        else:
            if self.S11 is None:
                alpha = self.alpha * deg2rad
                cosa = cos(alpha)
                sina = sin(alpha)
                beta = self.beta * deg2rad
                cosb = cos(beta)
                sinb = sin(beta)
                gamma = self.gamma * deg2rad
                cosg = cos(gamma)
                sing = sin(gamma)

                self.S11 = (self.b * self.c * sina) ** 2
                self.S22 = (self.a * self.c * sinb) ** 2
                self.S33 = (self.a * self.b * sing) ** 2
                self.S12 = self.a * self.b * self.c * self.c * (cosa * cosb - cosg)
                self.S23 = self.a * self.a * self.b * self.c * (cosb * cosg - cosa)
                self.S13 = self.a * self.b * self.b * self.c * (cosg * cosa - cosb)

            invd2 = (
                self.S11 * h * h
                + self.S22 * k * k
                + self.S33 * l * l
                + 2 * self.S12 * h * k
                + 2 * self.S23 * k * l
                + 2 * self.S13 * h * l
            )
            invd2 /= (self.volume) ** 2
        return sqrt(1 / invd2)

    def calculate_dspacing(self, dmin=1.0):
        """Calculate all d-spacing down to dmin

        Applies registered selection rules

        :param dmin: minimum value of spacing requested
        :return: dict d-spacing as string, list of tuple with Miller indices
                preceded with the numerical value
        """
        hmax = int(ceil(self.a / dmin))
        kmax = int(ceil(self.b / dmin))
        lmax = int(ceil(self.c / dmin))
        res = {}
        for hkl in itertools.product(
            range(-hmax, hmax + 1), range(-kmax, kmax + 1), range(-lmax, lmax + 1)
        ):
            # Apply selection rule
            valid = True
            for rule in self.selection_rules:
                valid = rule(*hkl)
                if not valid:
                    break
            if not valid:
                continue

            d = self.d(hkl)
            if d < dmin:
                continue

            # Truncate precision at 8 digits to mitigate numerical noise. At most, 8 digits are used for saving.
            d = numpy.round(d, 8)

            if d in res:
                res[d].append(Miller(*hkl))
            else:
                res[d] = [Miller(*hkl)]
        for lst in res.values():
            "sort each group of equivalent reflection in a systematic order"
            lst.sort(key=lambda x:x[-1::-1])
        return res

    def build_calibrant_config(self, dmin=1.0):
        """Build a CalibrantConfig from the cell"""
        config = CalibrantConfig(cell=str(self), space_group=self.type)
        reflections = self.calculate_dspacing(dmin)
        dspacing = list(reflections.keys())
        dspacing.sort(reverse=True)
        for d in dspacing:
            reflection = reflections[d]
            config.reflections.append(Reflection(d, hkl=reflection[-1], multiplicity=len(reflection)))
        return config


    def save(self, name, long_name=None, doi=None, dmin=1.0, dest_dir=None):
        """Save information about the cell in a d-spacing file, usable as Calibrant

        :param name: name of the calibrant
        :param doi: reference of the publication used to parametrize the cell
        :param dmin: minimal d-spacing
        :param dest_dir: name of the directory where to save the result
        """
        fname = name + ".D"
        if dest_dir:
            fname = os.path.join(dest_dir, fname)
        config = self.build_calibrant_config(dmin)
        if doi:
            config.reference = doi
        config.name = name
        config.description = long_name
        config.filename=fname
        config.save(fname)

    def to_calibrant(self, dmin=1.0):
        """Convert a Cell object to a Calibrant object

        :param dmin: minimum d-spacing to include in calibrant (in Angstrom)
        :return: Calibrant object
        """
        from .calibrant import Calibrant  # lazy loading to prevent cyclic imports
        config = self.build_calibrant_config(dmin)
        return  Calibrant(config=config)
