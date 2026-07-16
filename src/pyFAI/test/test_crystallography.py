#!/usr/bin/env python
# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2026 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "16/07/2026"

import os
import unittest
import numpy
import logging
from .utilstest import UtilsTest
from ..crystallography import resolution, Cell, EquationOfState, ReflectionCondition
from ..crystallography.eos import (BirchMurnaghan, LatticeExpansion, Murnaghan, PVT,
                                   ThermalExpansion, Vinet, VolumeExpansion)
from ..io.calibrant_config import CalibrantConfig
from ..containers import Miller, Reflection

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


class _MurnaghanForTest(EquationOfState):
    """Minimalistic concrete model used to exercise the abstract class:
    V(P) = V0 * (1 + k0p*(P-P0)/k0)^(-1/k0p), analytically invertible."""

    name = "murnaghan-for-test"

    def __init__(self, k0, k0p, v0=None, t0=298.15, p0=0.0):
        super().__init__(v0=v0, t0=t0, p0=p0)
        self.k0 = float(k0)
        self.k0p = float(k0p)

    def volume_ratio(self, pressure=None, temperature=None):
        pressure, _temperature = self._reference_conditions(pressure, temperature)
        return (1.0 + self.k0p * (pressure - self.p0) / self.k0) ** (-1.0 / self.k0p)


class TestEquationOfState(unittest.TestCase):

    def test_abstract(self):
        self.assertRaises(TypeError, EquationOfState)

    def test_factory(self):
        self.assertIn("murnaghan-for-test", EquationOfState.names())
        eos = EquationOfState.factory("Murnaghan_For Test", k0=160.0, k0p=4.0)
        self.assertIsInstance(eos, _MurnaghanForTest)
        self.assertRaises(KeyError, EquationOfState.factory, "not-a-model")

    def test_reference_conditions(self):
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0)
        self.assertAlmostEqual(eos.volume_ratio(), 1.0, places=12)
        self.assertAlmostEqual(eos.linear_ratio(), 1.0, places=12)
        self.assertAlmostEqual(eos.pressure(ratio=1.0), eos.p0, places=8)

    def test_pressure_inversion(self):
        "The generic numerical inversion matches the analytical Murnaghan inverse"
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0)
        for p_ref in (0.5, 10.0, 100.0):
            ratio = eos.volume_ratio(pressure=p_ref)
            self.assertLess(ratio, 1.0)
            self.assertAlmostEqual(eos.pressure(ratio=ratio), p_ref, places=6)
        # dilatation: V > V0 corresponds to a negative pressure
        self.assertLess(eos.pressure(ratio=1.01), 0.0)

    def test_linear_ratio(self):
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0)
        self.assertAlmostEqual(eos.linear_ratio(pressure=10.0),
                               eos.volume_ratio(pressure=10.0) ** (1.0 / 3.0),
                               places=12)

    def test_volume(self):
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0, v0=100.0)
        self.assertAlmostEqual(eos.volume(), 100.0, places=10)
        self.assertAlmostEqual(eos.pressure(volume=eos.volume(pressure=5.0)), 5.0, places=6)
        bare = _MurnaghanForTest(k0=160.0, k0p=4.0)
        self.assertRaises(ValueError, bare.volume)
        self.assertRaises(ValueError, bare.pressure, 90.0)

    def test_temperature_isothermal(self):
        "An isothermal model cannot be used as a thermometer"
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0)
        self.assertRaises(ValueError, eos.temperature, None, None, 0.99)

    def test_serialization(self):
        eos = _MurnaghanForTest(k0=160.0, k0p=4.0, v0=100.0)
        dico = eos.as_dict()
        self.assertEqual(dico["model"], "murnaghan-for-test")
        clone = EquationOfState.from_dict(dico)
        self.assertEqual(eos, clone)
        self.assertIsInstance(repr(eos), str)


class TestVinet(unittest.TestCase):
    """Vinet EoS with parameters of gold: K0=167 GPa, K0'=6.0
    (Heinz & Jeanloz, J. Appl. Phys. 55 (1984), a0=4.0786A)"""

    def setUp(self):
        self.eos = Vinet(k0=167.0, k0p=6.0, v0=4.0786 ** 3)

    def test_factory(self):
        self.assertIn("Vinet", EquationOfState.names())
        clone = EquationOfState.factory("vinet", k0=167.0, k0p=6.0, v0=4.0786 ** 3)
        self.assertEqual(self.eos, clone)
        self.assertEqual(self.eos, EquationOfState.from_dict(self.eos.as_dict()))

    def test_reference_conditions(self):
        self.assertAlmostEqual(self.eos.volume_ratio(), 1.0, places=12)
        self.assertAlmostEqual(self.eos.pressure(ratio=1.0), 0.0, places=10)

    def test_linear_limit(self):
        "At low pressure the compression is linear: V/V0 = 1 - P/K0"
        p = 1e-3 * self.eos.k0
        self.assertAlmostEqual(self.eos.volume_ratio(pressure=p), 1.0 - 1e-3, places=5)

    def test_roundtrip(self):
        "P -> V/V0 -> P is the identity, including under extreme compression"
        for p_ref in (0.1, 5.0, 50.0, 300.0):
            ratio = self.eos.volume_ratio(pressure=p_ref)
            self.assertLess(ratio, 1.0)
            self.assertAlmostEqual(self.eos.pressure(ratio=ratio), p_ref, places=6)

    def test_monotonic(self):
        pressures = [0.0, 1.0, 10.0, 100.0, 1000.0]
        ratios = [self.eos.volume_ratio(pressure=p) for p in pressures]
        self.assertEqual(ratios, sorted(ratios, reverse=True))

    def test_tension(self):
        "Moderate tension expands the cell, beyond the spinodal there is no solution"
        self.assertGreater(self.eos.volume_ratio(pressure=-5.0), 1.0)
        self.assertAlmostEqual(self.eos.pressure(ratio=self.eos.volume_ratio(pressure=-5.0)),
                               -5.0, places=6)
        self.assertRaises(ValueError, self.eos.volume_ratio, -self.eos.k0)

    def test_gauge(self):
        "Pressure gauge mode: measured lattice parameter of gold -> pressure"
        a = 3.9500  # Angstrom
        p = self.eos.pressure(volume=a ** 3)
        self.assertGreater(p, 0.0)
        # consistency: back to the lattice parameter
        self.assertAlmostEqual(self.eos.linear_ratio(pressure=p) * 4.0786, a, places=6)


class TestBirchMurnaghan(unittest.TestCase):
    """Birch-Murnaghan 3rd order with parameters of gold (K0=167 GPa, K0'=5.5)"""

    def setUp(self):
        self.eos = BirchMurnaghan(k0=167.0, k0p=5.5, v0=4.0786 ** 3)

    def test_factory(self):
        self.assertIn("Birch-Murnaghan", EquationOfState.names())
        clone = EquationOfState.factory("birch_murnaghan", k0=167.0, k0p=5.5, v0=4.0786 ** 3)
        self.assertEqual(self.eos, clone)
        self.assertEqual(self.eos, EquationOfState.from_dict(self.eos.as_dict()))

    def test_reference_conditions(self):
        self.assertAlmostEqual(self.eos.volume_ratio(), 1.0, places=12)
        self.assertAlmostEqual(self.eos.pressure(ratio=1.0), 0.0, places=10)

    def test_second_order(self):
        "With K0'=4 the model reduces to 2nd order: P = 3/2 K0 (eta^7 - eta^5)"
        eos = BirchMurnaghan(k0=160.0)
        self.assertEqual(eos.k0p, 4.0)
        eta = 0.95 ** (-1.0 / 3.0)
        self.assertAlmostEqual(eos.pressure(ratio=0.95),
                               1.5 * 160.0 * (eta ** 7 - eta ** 5),
                               places=10)

    def test_linear_limit(self):
        "At low pressure the compression is linear: V/V0 = 1 - P/K0"
        p = 1e-3 * self.eos.k0
        self.assertAlmostEqual(self.eos.volume_ratio(pressure=p), 1.0 - 1e-3, places=5)

    def test_matches_vinet_at_low_pressure(self):
        "BM3 and Vinet agree to first order for moderate compressions"
        vinet = Vinet(k0=167.0, k0p=5.5)
        for p in (0.5, 2.0, 5.0):
            self.assertAlmostEqual(self.eos.volume_ratio(pressure=p),
                                   vinet.volume_ratio(pressure=p),
                                   places=4)

    def test_roundtrip(self):
        "P -> V/V0 -> P is the identity, including under extreme compression"
        for p_ref in (0.1, 5.0, 50.0, 300.0):
            ratio = self.eos.volume_ratio(pressure=p_ref)
            self.assertLess(ratio, 1.0)
            self.assertAlmostEqual(self.eos.pressure(ratio=ratio), p_ref, places=6)

    def test_monotonic(self):
        pressures = [0.0, 1.0, 10.0, 100.0, 1000.0]
        ratios = [self.eos.volume_ratio(pressure=p) for p in pressures]
        self.assertEqual(ratios, sorted(ratios, reverse=True))

    def test_tension(self):
        "Moderate tension expands the cell, beyond the spinodal there is no solution"
        ratio = self.eos.volume_ratio(pressure=-5.0)
        self.assertGreater(ratio, 1.0)
        self.assertAlmostEqual(self.eos.pressure(ratio=ratio), -5.0, places=6)
        self.assertRaises(ValueError, self.eos.volume_ratio, -self.eos.k0)

    def test_truncation_turnover(self):
        "With K0' < 4 the truncated model turns over under extreme compression"
        soft = BirchMurnaghan(k0=167.0, k0p=3.0)
        self.assertAlmostEqual(soft.pressure(ratio=soft.volume_ratio(pressure=50.0)),
                               50.0, places=6)
        self.assertRaises(ValueError, soft.volume_ratio, 3.0 * soft.k0)


class TestThermalExpansion(unittest.TestCase):
    """Fei-type thermal expansion with MgO-like parameters"""

    def setUp(self):
        self.eos = ThermalExpansion(alpha0=3.0e-5, alpha1=1.2e-8, alpha2=-0.5)

    def test_factory(self):
        self.assertIn("thermal-expansion", EquationOfState.names())
        clone = EquationOfState.factory("Thermal Expansion",
                                        alpha0=3.0e-5, alpha1=1.2e-8, alpha2=-0.5)
        self.assertEqual(self.eos, clone)
        self.assertEqual(self.eos, EquationOfState.from_dict(self.eos.as_dict()))

    def test_alpha(self):
        t = 400.0
        self.assertAlmostEqual(self.eos.alpha(t),
                               3.0e-5 + 1.2e-8 * t - 0.5 / t ** 2,
                               places=12)

    def test_constant_coefficient(self):
        "With a constant coefficient the ratio is exactly exp(alpha0 * (T-T0))"
        eos = ThermalExpansion(alpha0=3.5e-5)
        self.assertAlmostEqual(eos.volume_ratio(temperature=eos.t0 + 100.0),
                               numpy.exp(3.5e-3), places=12)
        self.assertAlmostEqual(eos.volume_ratio(), 1.0, places=12)

    def test_against_numerical_integration(self):
        "The analytical integral of alpha_V matches scipy.integrate.quad"
        from scipy.integrate import quad
        for t in (100.0, 500.0, 1500.0):
            expected = quad(self.eos.alpha, self.eos.t0, t)[0]
            self.assertAlmostEqual(numpy.log(self.eos.volume_ratio(temperature=t)),
                                   expected, places=10)

    def test_expansion_and_contraction(self):
        self.assertGreater(self.eos.volume_ratio(temperature=500.0), 1.0)
        self.assertLess(self.eos.volume_ratio(temperature=100.0), 1.0)

    def test_thermometer(self):
        """Temperature inversion: V/V0 -> T is the inverse of T -> V/V0.
        Only within the validity domain of the Fei model: alpha_V changes
        sign at sqrt(-alpha2/alpha0) ~ 129 K, below which V(T) is not monotonic."""
        for t_ref in (200.0, 350.0, 1500.0):
            ratio = self.eos.volume_ratio(temperature=t_ref)
            self.assertAlmostEqual(self.eos.temperature(ratio=ratio), t_ref, places=6)

    def test_thermometer_ambiguous(self):
        """Below ~129 K two temperatures share the same volume:
        the solution closest to the reference temperature is returned"""
        ratio = self.eos.volume_ratio(temperature=100.0)
        t = self.eos.temperature(ratio=ratio)
        self.assertAlmostEqual(self.eos.volume_ratio(temperature=t), ratio, places=10)
        self.assertGreater(t, 129.0)

    def test_thermometer_unreachable(self):
        "A volume smaller than the minimum of V(T) matches no temperature"
        self.assertRaises(ValueError, self.eos.temperature, None, None, 0.9)

    def test_isobaric(self):
        "The volume does not depend on pressure: no pressure can be inverted"
        self.assertRaises(ValueError, self.eos.pressure, None, None, 0.99)


class TestLatticeExpansion(unittest.TestCase):
    """Polynomial expansion of the lattice parameter, silicon-like"""

    def setUp(self):
        self.eos = LatticeExpansion(coefficients=[2.581e-6, 1.0e-9])

    def test_factory(self):
        self.assertIn("lattice-expansion", EquationOfState.names())
        clone = EquationOfState.factory("LatticeExpansion", coefficients=[2.581e-6, 1.0e-9])
        self.assertEqual(self.eos, clone)
        self.assertEqual(self.eos, EquationOfState.from_dict(self.eos.as_dict()))

    def test_polynomial(self):
        self.assertAlmostEqual(self.eos.linear_ratio(), 1.0, places=12)
        dt = 100.0
        self.assertAlmostEqual(self.eos.linear_ratio(temperature=self.eos.t0 + dt),
                               1.0 + 2.581e-6 * dt + 1.0e-9 * dt * dt,
                               places=12)

    def test_volume_is_cube(self):
        t = 500.0
        self.assertAlmostEqual(self.eos.volume_ratio(temperature=t),
                               self.eos.linear_ratio(temperature=t) ** 3,
                               places=12)

    def test_thermometer(self):
        for t_ref in (100.0, 350.0, 900.0):
            ratio = self.eos.volume_ratio(temperature=t_ref)
            self.assertAlmostEqual(self.eos.temperature(ratio=ratio), t_ref, places=6)

    def test_dspacing_scaling(self):
        "d-spacings of a cubic calibrant scale by the linear ratio"
        d0 = 3.13541554  # Si (111) at room temperature
        d = d0 * self.eos.linear_ratio(temperature=473.15)
        self.assertGreater(d, d0)
        self.assertLess((d - d0) / d0, 1e-3)


class TestMurnaghan(unittest.TestCase):
    """Murnaghan EoS, fully analytical in both directions"""

    def setUp(self):
        self.eos = Murnaghan(k0=167.0, k0p=5.5, v0=4.0786 ** 3)

    def test_factory(self):
        self.assertIn("Murnaghan", EquationOfState.names())
        clone = EquationOfState.factory("murnaghan", k0=167.0, k0p=5.5, v0=4.0786 ** 3)
        self.assertEqual(self.eos, clone)
        self.assertEqual(self.eos, EquationOfState.from_dict(self.eos.as_dict()))

    def test_reference_conditions(self):
        self.assertAlmostEqual(self.eos.volume_ratio(), 1.0, places=12)
        self.assertAlmostEqual(self.eos.pressure(ratio=1.0), 0.0, places=12)

    def test_roundtrip(self):
        "Both directions are analytical: the roundtrip is exact"
        for p_ref in (0.1, 5.0, 50.0):
            self.assertAlmostEqual(self.eos.pressure(ratio=self.eos.volume_ratio(pressure=p_ref)),
                                   p_ref, places=10)

    def test_generic_vs_analytic(self):
        "The analytical inversions match the generic numerical ones of the parent class"
        generic = _MurnaghanForTest(k0=167.0, k0p=5.5)
        for p in (1.0, 10.0, 100.0):
            self.assertAlmostEqual(self.eos.volume_ratio(pressure=p),
                                   generic.volume_ratio(pressure=p), places=12)
        self.assertAlmostEqual(self.eos.pressure(ratio=0.95),
                               generic.pressure(ratio=0.95), places=6)

    def test_matches_birch_murnaghan_at_low_pressure(self):
        bm3 = BirchMurnaghan(k0=167.0, k0p=5.5)
        for p in (0.5, 2.0):
            self.assertAlmostEqual(self.eos.volume_ratio(pressure=p),
                                   bm3.volume_ratio(pressure=p), places=4)

    def test_tension_limit(self):
        "The model is only defined above P0 - K0/K0'"
        limit = -self.eos.k0 / self.eos.k0p
        self.assertGreater(self.eos.volume_ratio(pressure=0.99 * limit), 1.0)
        self.assertRaises(ValueError, self.eos.volume_ratio, 1.01 * limit)


class TestPVT(unittest.TestCase):
    """Composite P-V-T model, gold-like parameters"""

    def setUp(self):
        self.isothermal = BirchMurnaghan(k0=167.0, k0p=5.5)
        self.thermal = ThermalExpansion(alpha0=4.2e-5)
        self.eos = PVT(self.isothermal, self.thermal, dk0dt=-0.02, v0=4.0786 ** 3)

    def test_reference_conditions(self):
        self.assertAlmostEqual(self.eos.volume_ratio(), 1.0, places=12)
        self.assertEqual(self.eos.t0, self.thermal.t0)
        self.assertEqual(self.eos.p0, self.isothermal.p0)

    def test_reduces_to_submodels(self):
        "At reference temperature only the compression acts, at reference pressure only the expansion"
        self.assertAlmostEqual(self.eos.volume_ratio(pressure=10.0),
                               self.isothermal.volume_ratio(pressure=10.0), places=12)
        self.assertAlmostEqual(self.eos.volume_ratio(temperature=500.0),
                               self.thermal.volume_ratio(temperature=500.0), places=12)

    def test_composition(self):
        "V(P,T) = V0(T)/V0 * compression with K0 corrected at T"
        p, t = 10.0, 500.0
        corrected = BirchMurnaghan(k0=167.0 - 0.02 * (t - self.eos.t0), k0p=5.5)
        self.assertAlmostEqual(self.eos.volume_ratio(pressure=p, temperature=t),
                               self.thermal.volume_ratio(temperature=t) * corrected.volume_ratio(pressure=p),
                               places=12)

    def test_gauge_at_temperature(self):
        "Pressure gauge mode at any temperature"
        for p_ref, t in ((5.0, 500.0), (25.0, 1200.0), (10.0, 150.0)):
            ratio = self.eos.volume_ratio(pressure=p_ref, temperature=t)
            self.assertAlmostEqual(self.eos.pressure(ratio=ratio, temperature=t), p_ref, places=8)

    def test_thermometer_at_pressure(self):
        "Thermometer mode at fixed pressure, generic numerical inversion"
        ratio = self.eos.volume_ratio(pressure=5.0, temperature=700.0)
        self.assertAlmostEqual(self.eos.temperature(ratio=ratio, pressure=5.0), 700.0, places=6)

    def test_serialization(self):
        "The nested representation is JSON-able and instantiable by the factory"
        import json
        dico = self.eos.as_dict()
        self.assertEqual(dico["model"], "PVT")
        self.assertEqual(dico, json.loads(json.dumps(dico)))
        clone = EquationOfState.from_dict(dico)
        self.assertEqual(self.eos, clone)
        self.assertAlmostEqual(clone.volume_ratio(pressure=10.0, temperature=500.0),
                               self.eos.volume_ratio(pressure=10.0, temperature=500.0),
                               places=12)

    def test_dk0pdt(self):
        "K0' is corrected linearly with temperature, like K0"
        eos = PVT(self.isothermal, self.thermal, dk0pdt=1.0e-3)
        t = 800.0
        corrected = BirchMurnaghan(k0=167.0, k0p=5.5 + 1.0e-3 * (t - eos.t0))
        self.assertAlmostEqual(eos.volume_ratio(pressure=20.0, temperature=t),
                               self.thermal.volume_ratio(temperature=t) * corrected.volume_ratio(pressure=20.0),
                               places=12)
        self.assertRaises(ValueError, PVT, self.thermal, self.thermal, dk0pdt=1.0e-3)

    def test_dk0dt_validation(self):
        "dk0dt requires an isothermal model exposing k0, and K0(T) must stay positive"
        self.assertRaises(ValueError, PVT, self.thermal, self.thermal, dk0dt=-0.02)
        self.assertRaises(ValueError, self.eos.volume_ratio, 10.0, 9000.0)


class TestVolumeExpansion(unittest.TestCase):
    """Polynomial expansion of the volume, the JCPDS parametrization"""

    def test_polynomial(self):
        eos = VolumeExpansion([4.26e-5, 1.0e-9])
        self.assertAlmostEqual(eos.volume_ratio(), 1.0, places=12)
        dt = 200.0
        self.assertAlmostEqual(eos.volume_ratio(temperature=eos.t0 + dt),
                               1.0 + 4.26e-5 * dt + 1.0e-9 * dt * dt,
                               places=12)

    def test_thermometer(self):
        eos = VolumeExpansion([4.26e-5])
        ratio = eos.volume_ratio(temperature=800.0)
        self.assertAlmostEqual(eos.temperature(ratio=ratio), 800.0, places=6)

    def test_serialization(self):
        self.assertIn("volume-expansion", EquationOfState.names())
        eos = VolumeExpansion([4.26e-5, 1.0e-9])
        self.assertEqual(eos, EquationOfState.from_dict(eos.as_dict()))


AU_JCPDS = """VERSION: 4
COMMENT: Gold
K0: 166.65
K0P: 5.4823
DK0DT: -0.021
SYMMETRY: CUBIC
A: 4.07860
ALPHAT: 4.26e-05
DIHKL: 2.35480, 100.0, 1, 1, 1
DIHKL: 2.03930, 52.0, 2, 0, 0
"""


class TestJCPDS(unittest.TestCase):
    """Serialization/deserialization of JCPDS (version 4) files with EoS"""

    def test_read(self):
        "Parse a Dioptas-like gold file"
        fname = os.path.join(UtilsTest.tempdir, "Au_ref.jcpds")
        with open(fname, "w") as fd:
            fd.write(AU_JCPDS)
        config = CalibrantConfig.from_JCPDS(fname)
        self.assertEqual(config.description, "Gold")
        self.assertIn("cubic", config.cell)
        self.assertEqual(len(config.reflections), 2)
        self.assertAlmostEqual(config.reflections[0].dspacing, 2.3548, places=6)
        self.assertEqual(config.reflections[0].hkl, (1, 1, 1))
        self.assertAlmostEqual(config.reflections[1].intensity, 52.0, places=6)
        eos = config.eos
        self.assertIsInstance(eos, PVT)
        self.assertAlmostEqual(eos.isothermal.k0, 166.65, places=6)
        self.assertAlmostEqual(eos.dk0dt, -0.021, places=6)
        self.assertAlmostEqual(eos.v0, 4.0786 ** 3, places=6)
        # sanity check in gauge mode: compression at 10 GPa, 500 K
        self.assertLess(eos.linear_ratio(pressure=10.0, temperature=298.15), 1.0)
        self.assertGreater(eos.volume_ratio(temperature=500.0), 1.0)

    def test_roundtrip(self):
        "CalibrantConfig -> JCPDS -> CalibrantConfig preserves reflections, cell and EoS"
        cell = Cell.cubic(4.0786, lattice_type="F")
        config = cell.build_calibrant_config(dmin=1.0)
        config.name = "Au"
        config.description = "Gold"
        config.eos = PVT(BirchMurnaghan(k0=166.65, k0p=5.4823),
                         VolumeExpansion([4.26e-5, 1.0e-9]),
                         dk0dt=-0.021,
                         v0=cell.volume)
        text = config.to_JCPDS()
        self.assertIn("SYMMETRY: CUBIC", text)
        self.assertIn("K0: 166.65", text)

        fname = os.path.join(UtilsTest.tempdir, "Au_out.jcpds")
        config.save_JCPDS(fname)
        clone = CalibrantConfig.from_JCPDS(fname)

        self.assertEqual(len(clone.reflections), len(config.reflections))
        for ref, out in zip(config.reflections, clone.reflections):
            self.assertAlmostEqual(ref.dspacing, out.dspacing, places=7)
            self.assertEqual(ref.hkl, out.hkl)
        self.assertIn("cubic", clone.cell)
        self.assertEqual(clone.eos, config.eos)
        self.assertAlmostEqual(clone.eos.linear_ratio(pressure=10.0, temperature=500.0),
                               config.eos.linear_ratio(pressure=10.0, temperature=500.0),
                               places=12)

    def test_thermal_only(self):
        "A calibrant with only a thermal model exports and reimports"
        cell = Cell.diamond(5.431179)
        config = cell.build_calibrant_config(dmin=1.5)
        config.description = "Silicon"
        config.eos = VolumeExpansion([7.8e-6])
        fname = os.path.join(UtilsTest.tempdir, "Si_out.jcpds")
        config.save_JCPDS(fname)
        clone = CalibrantConfig.from_JCPDS(fname)
        self.assertIsInstance(clone.eos, VolumeExpansion)
        self.assertAlmostEqual(clone.eos.coefficients[0], 7.8e-6, places=12)
        # the cell parameter is rounded to 5 decimal places in the cell description
        self.assertAlmostEqual(clone.eos.v0, cell.volume, places=3)

    def test_unsupported_model(self):
        "A Vinet EoS cannot be mapped onto the JCPDS parametrization"
        cell = Cell.cubic(4.0786, lattice_type="F")
        config = cell.build_calibrant_config(dmin=1.0)
        config.eos = Vinet(k0=167.0, k0p=6.0)
        self.assertRaises(ValueError, config.to_JCPDS)


class TestCalibrantHeaders(unittest.TestCase):
    """Parsing of the (heterogeneous, often hand-crafted) headers of .D files"""

    @classmethod
    def setUpClass(cls):
        from ..utils import get_calibration_dir
        cls.calibration_dir = get_calibration_dir()

    @classmethod
    def tearDownClass(cls):
        cls.calibration_dir = None

    def config(self, name):
        return CalibrantConfig.from_dspacing(os.path.join(self.calibration_dir, name + ".D"))

    def test_all_shipped_files_roundtrip(self):
        "Every calibrant file shipped with pyFAI parses and survives a write/parse cycle unchanged"
        counter = 0
        for fname in sorted(os.listdir(self.calibration_dir)):
            if not fname.endswith(".D"):
                continue
            counter += 1
            config = CalibrantConfig.from_dspacing(os.path.join(self.calibration_dir, fname))
            self.assertTrue(config.reflections, fname)

            tmp = os.path.join(UtilsTest.tempdir, fname)
            config.save(tmp)
            clone = CalibrantConfig.from_dspacing(tmp)
            for attribute in ("name", "description", "cell", "space_group", "reference"):
                self.assertEqual(getattr(config, attribute), getattr(clone, attribute),
                                 f"{attribute} of {fname}")
            self.assertEqual(len(config.reflections), len(clone.reflections), fname)
            for ref, out in zip(config.reflections, clone.reflections):
                self.assertEqual(ref.dspacing, out.dspacing, fname)
                self.assertEqual(ref.hkl, out.hkl, fname)
                self.assertEqual(ref.multiplicity, out.multiplicity, fname)
                self.assertEqual(ref.intensity, out.intensity, fname)
        self.assertGreater(counter, 30, "the shipped calibrant files were found")

    def test_to_cell_flavors(self):
        "The various hand-crafted cell descriptions are interpreted correctly"
        # unicode, centering word: `Face centered cubic cell a=5.411651Å ... α=90° ...`
        cell = self.config("CeO2").to_cell()
        self.assertEqual(cell.lattice, "cubic")
        self.assertEqual(cell.type, "F")
        self.assertAlmostEqual(cell.a, 5.411651, places=6)
        # ASCII names, no unit: `Cubic cell a=4.0495 ... alpha=90.000 ... (Fm3m)`
        cell = self.config("Al").to_cell()
        self.assertEqual(cell.lattice, "cubic")
        self.assertEqual(cell.type, "F", "centering recovered from the space group Fm3m")
        self.assertAlmostEqual(cell.a, 4.0495, places=4)
        # double lattice word: `Rhombohedral hexagonal cell ... (R-3c, 167)`
        cell = self.config("alpha_Al2O3_SRM676a_2015").to_cell()
        self.assertEqual(cell.lattice, "hexagonal")
        self.assertEqual(cell.type, "R")
        self.assertAlmostEqual(cell.c, 12.99231, places=5)
        self.assertAlmostEqual(cell.gamma, 120.0, places=6)
        # bare values: `14.2600  14.2600  14.2600   90.000   90.000   90.000 (Fm3)`
        cell = self.config("C60").to_cell()
        self.assertEqual(cell.lattice, "cubic")
        self.assertAlmostEqual(cell.a, 14.26, places=4)
        # bare values, hexagonal: `2.4560   2.4560   6.6960   90.000   90.000  120.000`
        cell = self.config("graphite").to_cell()
        self.assertEqual(cell.lattice, "hexagonal")
        self.assertAlmostEqual(cell.c, 6.696, places=4)
        # partial parameters: `Monoclinic cell a=29.59 b=6.15 c=3.98 beta=95.5 (P21/a)`
        cell = self.config("PBBA").to_cell()
        self.assertEqual(cell.lattice, "monoclinic")
        self.assertAlmostEqual(cell.beta, 95.5, places=4)
        self.assertAlmostEqual(cell.alpha, 90.0, places=6)
        # pseudo-crystal with infinite parameters and free text yield None
        self.assertIsNone(self.config("AgBh").to_cell())
        self.assertIsNone(self.config("CrOx").to_cell())
        self.assertIsNone(self.config("mock").to_cell())

    def test_dspacing_consistency(self):
        "The cell rebuilt from the header reproduces the tabulated d-spacings"
        for name in ("CeO2", "LaB6", "Si"):
            config = self.config(name)
            cell = config.to_cell()
            for reflection in config.reflections[:5]:
                if reflection.hkl:
                    self.assertAlmostEqual(cell.d(reflection.hkl) / reflection.dspacing, 1.0,
                                           places=3, msg=f"{name} {reflection.hkl}")

    def test_eos_roundtrip(self):
        "The EoS survives a write/parse cycle of the .D file"
        config = self.config("CeO2")
        cell = config.to_cell()
        config.eos = PVT(BirchMurnaghan(k0=220.0, k0p=4.4),
                         VolumeExpansion([3.5e-5]),
                         dk0dt=-0.02,
                         v0=cell.volume)
        tmp = os.path.join(UtilsTest.tempdir, "CeO2_eos.D")
        config.save(tmp)
        text = open(tmp).read()
        self.assertIn("# EoS: {", text)
        clone = CalibrantConfig.from_dspacing(tmp)
        self.assertEqual(clone.eos, config.eos)
        self.assertEqual(clone.cell, config.cell)
        self.assertEqual(len(clone.reflections), len(config.reflections))

    def test_intensity_without_multiplicity(self):
        "Reflections with an intensity but no multiplicity (e.g. imported from JCPDS) round-trip"
        config = CalibrantConfig(name="jcpds_like")
        config.reflections = [Reflection(dspacing=2.3548, intensity=100.0, hkl=Miller(1, 1, 1)),
                              Reflection(dspacing=2.0393, intensity=52.0, hkl=Miller(2, 0, 0))]
        tmp = os.path.join(UtilsTest.tempdir, "jcpds_like.D")
        config.save(tmp)
        clone = CalibrantConfig.from_dspacing(tmp)
        self.assertEqual(len(clone.reflections), 2)
        self.assertEqual(clone.reflections[0].intensity, 100.0)
        self.assertIsNone(clone.reflections[0].multiplicity)
        self.assertEqual(clone.reflections[1].hkl, (2, 0, 0))
        self.assertEqual(clone.reflections[1].intensity, 52.0)

    def test_eos_corrupted(self):
        "A hand-mangled EoS line does not prevent the calibrant from loading"
        config = self.config("CeO2")
        config.eos = BirchMurnaghan(k0=220.0, k0p=4.4)
        tmp = os.path.join(UtilsTest.tempdir, "CeO2_bad_eos.D")
        config.save(tmp)
        with open(tmp) as fd:
            text = fd.read()
        with open(tmp, "w") as fd:
            fd.write(text.replace('"model"', '"mangled'))
        with self.assertLogs("pyFAI.io.calibrant_config", level="WARNING"):
            clone = CalibrantConfig.from_dspacing(tmp)
        self.assertIsNone(clone.eos)
        self.assertEqual(len(clone.reflections), len(config.reflections))


def suite():
    testsuite = unittest.TestSuite()
    loader = unittest.defaultTestLoader.loadTestsFromTestCase
    testsuite.addTest(loader(TestCrystallography))
    testsuite.addTest(loader(TestEquationOfState))
    testsuite.addTest(loader(TestVinet))
    testsuite.addTest(loader(TestBirchMurnaghan))
    testsuite.addTest(loader(TestThermalExpansion))
    testsuite.addTest(loader(TestLatticeExpansion))
    testsuite.addTest(loader(TestMurnaghan))
    testsuite.addTest(loader(TestPVT))
    testsuite.addTest(loader(TestVolumeExpansion))
    testsuite.addTest(loader(TestJCPDS))
    testsuite.addTest(loader(TestCalibrantHeaders))
    return testsuite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    UtilsTest.clean_up()
