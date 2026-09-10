#!/usr/bin/env python
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""test suite for average library
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/09/2026"

import logging
import unittest
from typing import ClassVar

import numpy

from ..crystallography import Cell, ReflectionCondition, resolution
from .utilstest import UtilsTest

logger = logging.getLogger(__name__)


class TestCrystallography(unittest.TestCase):

    def test_constant(self):
        ref = [1] * 11
        c = resolution.Constant(180/numpy.pi)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.fwhm(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.sigma(1), float))

    def test_caglioti(self):
        ref = [0.04246609, 0.05619075, 0.07367654, 0.09299528, 0.11344279,
       0.1347813 , 0.15696523, 0.18004253, 0.20411724, 0.22933564,
       0.255883  ]
        c = resolution.Caglioti(1,1e-1,1e-2)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))

    def test_chernyshov(self):
        ref = [0.44740802, 0.4452937 , 0.43897227, 0.4285082 , 0.41400832,
       0.39562113, 0.37353566, 0.34798047, 0.31922269, 0.28756785,
       0.25336168]
        c = resolution.Chernyshov(1,1e-1,1e-2)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0,1,11)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))

    def test_langford(self):
        ref = [8.48615349, 4.23249535, 2.80994112, 2.09520315, 1.6636391 ,
       1.37371653, 1.16479474, 1.00657162, 0.8822346 , 0.7817234 ]
        c = resolution.Langford(1e-3, 1e-2, 1e-1, 1)
        self.assertTrue(isinstance(c.__repr__(), str))
        self.assertTrue(numpy.allclose(c.sigma(numpy.linspace(0.1,1,10)), ref))
        self.assertTrue(isinstance(c.fwhm(1), float))

    def test_bug_2755(self):
        "Missing default selection rule for C-type cells"
        phase1 = Cell.monoclinic(3, 4, 5, 115, lattice_type='C')
        res1 = len(phase1.calculate_dspacing(dmin=1))

        phase2 = Cell.monoclinic(3, 4, 5, 115, lattice_type='P')
        res0 = len(phase2.calculate_dspacing(dmin=1))
        phase2.selection_rules.append(ReflectionCondition.group5_C2)
        res2 = len(phase2.calculate_dspacing(dmin=1))
        self.assertEqual(res1, res2)
        self.assertGreater(res0, res2)


class TestLatticeCentring(unittest.TestCase):
    """Selection rules of the centred Bravais lattices, see issue #2794"""

    # nodes of the conventional cell for each centring
    NODES: ClassVar[dict] = {"P": [(0, 0, 0)],
             "A": [(0, 0, 0), (0, 0.5, 0.5)],
             "B": [(0, 0, 0), (0.5, 0, 0.5)],
             "C": [(0, 0, 0), (0.5, 0.5, 0)],
             "I": [(0, 0, 0), (0.5, 0.5, 0.5)],
             "F": [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]}

    def structure_factor(self, lattice_type, hkl):
        """Modulus of the structure factor of the lattice alone, the reference the
        selection rules have to agree with. Nota: the rhombohedral centring is left out,
        as two settings (obverse and reverse) are in use."""
        h, k, l = hkl  # noqa: E741
        amplitude = sum(numpy.exp(2j * numpy.pi * (h * x + k * y + l * z))
                        for x, y, z in self.NODES[lattice_type])
        return abs(amplitude)

    def test_selection_rules(self):
        """Every rule must allow exactly the reflections with a non-zero structure factor"""
        for lattice_type in self.NODES:
            rule = getattr(ReflectionCondition, f"type_{lattice_type}")
            for h in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):  # noqa: E741
                        if h == k == l == 0:
                            continue
                        expected = self.structure_factor(lattice_type, (h, k, l)) > 1e-10
                        self.assertEqual(rule(h, k, l), expected,
                                         f"lattice {lattice_type}, reflection {h}{k}{l}")

    def test_C_centred_orthorhombic(self):
        """Non-regression for #2794: 010 and 011 are extinct, 001 is not

        The C centring adds a node in the *ab* plane, so it does not halve the period along
        *c*: the (001) planes stay `c` apart and the 001 reflection is allowed, while the
        (010) planes get an extra plane half-way and 010 is extinct.
        """
        cell = Cell.orthorhombic(3.0, 4.0, 5.0, lattice_type="C")
        self.assertEqual(cell.type, "C", "the lattice type is taken into account")
        reflections = cell.calculate_dspacing(2.0)
        found = {(m.h, m.k, m.l) for millers in reflections.values() for m in millers}
        for hkl in ((0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)):
            self.assertNotIn(hkl, found, f"{hkl} is extinct in a C-centred lattice")
        for hkl in ((0, 0, 1), (1, 1, 0), (1, 1, 1), (0, 2, 0)):
            self.assertIn(hkl, found, f"{hkl} is allowed in a C-centred lattice")
        self.assertEqual(len(reflections), 5,
                         "5 reflections with d > 2 A: 001, 002, 110, 111 and 020")


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCrystallography))
    testsuite.addTest(loader(TestLatticeCentring))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
