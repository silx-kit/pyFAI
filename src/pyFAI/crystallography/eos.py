#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2026-2026 European Synchrotron Radiation Facility, Grenoble, France
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

"""Equation of state

A module with the abstract EquationOfState class, parent of the different
models describing the evolution of the unit-cell volume with pressure and
temperature: V = f(P, T).

Those models allow the d-spacing of a calibrant to be recalculated at the
actual conditions of the experiment (cryostat, furnace, diamond anvil cell...)
and, conversely, a calibrant to be used as a pressure gauge.

Conventions used throughout this module (high-pressure crystallography usage):

* volumes in cubic Angstrom (or dimensionless, relative to V0)
* pressures in GPa, reference pressure defaults to 0 (ambient)
* temperatures in Kelvin, reference temperature defaults to 298.15 K

References:

* R.J. Angel, Equations of State, Reviews in Mineralogy and Geochemistry 41 (2000) 35-59.
  https://doi.org/10.2138/rmg.2000.41.2
* R.J. Angel, M. Alvaro, J. Gonzalez-Platas, EosFit7c, Z. Kristallogr. 229 (2014) 405-419.
  https://doi.org/10.1515/zkri-2013-1711
* Powder Diffraction: Theory and Practice, R.E. Dinnebier & S.J.L. Billinge (eds),
  RSC Publishing (2008), chapter 4.
"""

from __future__ import annotations

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "16/07/2026"
__status__ = "development"

import logging
from abc import ABC, abstractmethod
from math import exp
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# Reference conditions: ambient pressure (GPa) and room temperature (K)
P_REF = 0.0
T_REF = 298.15
# Upper bound (GPa) when bracketing an inverted pressure
P_MAX = 1e5
# Bounds (K) when bracketing an inverted temperature
T_MIN = 1.0
T_MAX = 1e4


def _normalize(name: str) -> str:
    """Normalize a model name for registry lookup: `Birch-Murnaghan` -> `birchmurnaghan`"""
    return "".join(c for c in name.lower() if c.isalnum())


class EquationOfState(ABC):
    """Abstract base class for equations of state V = f(P, T).

    Concrete models (Birch-Murnaghan, Vinet, Murnaghan, thermal expansion, ...)
    have to inherit from this class, define a unique ``name`` and implement
    :meth:`volume_ratio` which returns V/V0 at the given pressure (GPa) and
    temperature (K). Every other method has a generic implementation based on
    ``volume_ratio``, but may be overridden when an analytical form exists
    (e.g. the inverse P(V) of the Murnaghan model).

    Subclasses are automatically registered and can be instantiated by name:

    .. code-block:: python

        eos = EquationOfState.factory("Birch-Murnaghan", k0=160., k0p=4.)
        eos.volume_ratio(pressure=10)   # V/V0 at 10 GPa
        eos.linear_ratio(pressure=10)   # (V/V0)^(1/3), scales d-spacings of a cubic cell
        eos.pressure(ratio=0.95)        # inverse: P such as V/V0 = 0.95

    :param v0: unit-cell volume at the reference conditions, in A^3.
               Optional: only needed to work with absolute volumes.
    :param t0: reference temperature in K (298.15 K by default)
    :param p0: reference pressure in GPa (0 by default, i.e. ambient)
    """

    name = None
    "Unique, human readable identifier of the model, defined by each subclass"

    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            key = _normalize(cls.name)
            if key in EquationOfState._registry:
                logger.warning("Equation of state named `%s` already registered, overwriting it", cls.name)
            EquationOfState._registry[key] = cls

    def __init__(self,
                 v0: float | None = None,
                 t0: float = T_REF,
                 p0: float = P_REF):
        self.v0 = None if v0 is None else float(v0)
        self.t0 = float(t0)
        self.p0 = float(p0)

    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.as_dict().items() if k != "model")
        return f"{self.__class__.__name__}({params})"

    def __eq__(self, other):
        return isinstance(other, EquationOfState) and self.as_dict() == other.as_dict()

    def _reference_conditions(self, pressure=None, temperature=None):
        """Replace missing pressure/temperature by the reference conditions

        :return: 2-tuple (pressure in GPa, temperature in K)
        """
        return (self.p0 if pressure is None else float(pressure),
                self.t0 if temperature is None else float(temperature))

    @abstractmethod
    def volume_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative unit-cell volume V/V0 at the given conditions.

        Implementations must interpret a missing pressure or temperature as
        the reference conditions (p0, t0), where the ratio equals 1.

        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: V/V0, dimensionless
        """

    def volume(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the absolute unit-cell volume at the given conditions.

        Requires the reference volume ``v0`` to be defined.

        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: unit-cell volume in A^3
        """
        if self.v0 is None:
            raise ValueError("The reference volume `v0` is undefined: only relative volumes can be calculated, see `volume_ratio`")
        return self.v0 * self.volume_ratio(pressure, temperature)

    def linear_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative variation of any lattice length, (V/V0)^(1/3).

        This assumes an isotropic deformation of the unit-cell, exact for cubic
        lattices: cell lengths and d-spacings all scale by this factor.

        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: a/a0 = d/d0, dimensionless
        """
        return self.volume_ratio(pressure, temperature) ** (1.0 / 3.0)

    def _resolve_ratio(self, volume: float | None = None, ratio: float | None = None) -> float:
        """Turn either an absolute volume or a relative one into a V/V0 ratio

        :param volume: absolute unit-cell volume in A^3 (requires ``v0``)
        :param ratio: relative volume V/V0, alternative to ``volume``
        :return: V/V0 as float
        """
        if ratio is None:
            if volume is None:
                raise ValueError("Provide either `volume` or `ratio`")
            if self.v0 is None:
                raise ValueError("The reference volume `v0` is undefined: provide `ratio` instead of `volume`")
            ratio = volume / self.v0
        return float(ratio)

    def pressure(self,
                 volume: float | None = None,
                 temperature: float | None = None,
                 ratio: float | None = None) -> float:
        """Calculate the pressure from the unit-cell volume at a given temperature.

        This is the inverse of :meth:`volume_ratio` at fixed temperature,
        solved numerically. It is the *pressure gauge* mode: measure the cell
        volume of the calibrant, obtain the pressure. Subclasses may override
        it when an analytical inverse exists.

        :param volume: absolute unit-cell volume in A^3 (requires ``v0``)
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :param ratio: relative volume V/V0, alternative to ``volume``
        :return: pressure in GPa
        """
        ratio = self._resolve_ratio(volume, ratio)

        def fun(p):
            return self.volume_ratio(p, temperature) - ratio

        residual = fun(self.p0)
        if residual == 0.0:
            return self.p0
        # V decreases with P: compression lies above p0, dilatation below
        step = 1.0
        sign = 1.0 if residual > 0.0 else -1.0
        while step <= P_MAX:
            bound = self.p0 + sign * step
            if fun(bound) * residual < 0.0:
                lo, hi = sorted((self.p0, bound))
                return brentq(fun, lo, hi)
            step *= 2.0
        raise ValueError(f"No pressure in ]{-P_MAX}, {P_MAX}[ GPa matches V/V0={ratio} at this temperature")

    def temperature(self,
                    volume: float | None = None,
                    pressure: float | None = None,
                    ratio: float | None = None) -> float:
        """Calculate the temperature from the unit-cell volume at a given pressure.

        This is the inverse of :meth:`volume_ratio` at fixed pressure, solved
        numerically: the *thermometer* mode, measure the cell volume of the
        calibrant, obtain the temperature. Walks away from the reference
        temperature until the target volume is bracketed: when the model is
        not monotonic (outside of its validity domain), the solution closest
        to t0 is returned. Raises a ValueError when the volume is never
        reached, in particular for isothermal models where the volume does
        not depend on the temperature.

        :param volume: absolute unit-cell volume in A^3 (requires ``v0``)
        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param ratio: relative volume V/V0, alternative to ``volume``
        :return: temperature in K
        """
        ratio = self._resolve_ratio(volume, ratio)

        def fun(t):
            return self.volume_ratio(pressure, t) - ratio

        residual = fun(self.t0)
        if residual == 0.0:
            return self.t0
        step = 16.0
        message = f"No temperature in [{T_MIN}, {T_MAX}] K matches V/V0={ratio} at this pressure with {self!r}"
        # probe both sides of t0 for a bracket or at least a descent direction
        t_up = min(self.t0 + step, T_MAX)
        f_up = fun(t_up)
        if f_up * residual <= 0.0:
            return brentq(fun, self.t0, t_up)
        t_dn = max(self.t0 - step, T_MIN)
        f_dn = fun(t_dn)
        if f_dn * residual <= 0.0:
            return brentq(fun, t_dn, self.t0)
        if abs(f_up) < abs(residual):
            direction, t, f = 1.0, t_up, f_up
        elif abs(f_dn) < abs(residual):
            direction, t, f = -1.0, t_dn, f_dn
        else:
            raise ValueError(message)
        while True:
            t_next = t + direction * step
            if not T_MIN <= t_next <= T_MAX:
                raise ValueError(message)
            f_next = fun(t_next)
            if f_next * f <= 0.0:
                lo, hi = sorted((t, t_next))
                return brentq(fun, lo, hi)
            if abs(f_next) >= abs(f):
                # moving away from the target: extremum of a non-monotonic model passed
                raise ValueError(message)
            t, f = t_next, f_next

    def _invert_ratio(self, pressure: float, temperature: float | None = None, step: float = 0.05) -> float:
        """Numerically invert an *analytical* :meth:`pressure` into a V/V0 ratio.

        Helper for models which override :meth:`pressure` with a closed form
        (Vinet, Birch-Murnaghan, ...): do NOT call it from a class relying on
        the generic :meth:`pressure`, it would recurse infinitely.

        Walks x = (V/V0)^(1/3) away from 1 until the target pressure is
        bracketed, then refines with Brent. The walk stops when the pressure
        is no longer approached: outside of the monotonic domain of the model
        (tension spinodal, turnover of a truncated model under extreme
        compression) a ValueError is raised.

        :param pressure: target pressure in GPa
        :param temperature: temperature in K, forwarded to :meth:`pressure`
        :param step: increment of x used to bracket the solution
        :return: V/V0, dimensionless
        """

        def fun(x):
            return self.pressure(ratio=x ** 3, temperature=temperature) - pressure

        x, residual = 1.0, fun(1.0)
        if residual == 0.0:
            return 1.0
        direction = -1.0 if residual < 0.0 else 1.0  # compression: x < 1
        while True:
            x_next = x + direction * step
            if x_next <= 0.0:
                raise ValueError(f"No volume matches a pressure of {pressure} GPa with {self!r}")
            f_next = fun(x_next)
            if f_next * residual <= 0.0:
                lo, hi = sorted((x, x_next))
                return brentq(fun, lo, hi) ** 3
            if abs(f_next) >= abs(residual):
                raise ValueError(f"Pressure {pressure} GPa is beyond the validity range of {self!r}")
            x, residual = x_next, f_next

    def as_dict(self) -> dict:
        """Serialize the model to a dictionary of builtin types.

        The `model` key holds the registered name, the other items are the
        parameters expected by the constructor: ``from_dict(as_dict())`` is
        the identity.
        """
        dico = {"model": self.name}
        dico.update((k, v) for k, v in vars(self).items()
                    if not k.startswith("_") and v is not None)
        return dico

    @classmethod
    def from_dict(cls, dico: dict) -> EquationOfState:
        """Instantiate any registered model from its dictionary representation.

        :param dico: dictionary as produced by :meth:`as_dict`
        :return: instance of the concrete equation of state
        """
        dico = dict(dico)
        return cls.factory(dico.pop("model"), **dico)

    @classmethod
    def factory(cls, name: str, **parameters) -> EquationOfState:
        """Instantiate a registered model from its name (case/separator insensitive).

        :param name: registered name of the model, e.g. "Birch-Murnaghan"
        :param parameters: parameters passed to the constructor of the model
        :return: instance of the concrete equation of state
        """
        key = _normalize(name)
        if key not in cls._registry:
            raise KeyError(f"No equation of state named `{name}`, available models are: {cls.names()}")
        return cls._registry[key](**parameters)

    @classmethod
    def names(cls) -> list:
        """Return the names of all registered models"""
        return [klass.name for klass in cls._registry.values()]


class Vinet(EquationOfState):
    """Vinet (universal) equation of state, isothermal.

    .. math::

        P = P_0 + 3 K_0 \\frac{1-x}{x^2} \\exp\\left(\\frac{3}{2}(K_0'-1)(1-x)\\right)
        \\qquad x = (V/V_0)^{1/3}

    Derived from a universal interatomic potential, it behaves better than
    the Birch-Murnaghan model under very high compression (V/V0 < 0.6, metals
    in diamond anvil cells like Au, Pt, ...). The pressure is analytical in V,
    the volume is obtained by numerical inversion. Temperature is ignored.

    Reference: P. Vinet, J. Ferrante, J.H. Rose, J.R. Smith,
    Compressibility of solids, J. Geophys. Res. 92 (1987) 9319-9325.
    https://doi.org/10.1029/JB092iB09p09319

    :param k0: isothermal bulk modulus at the reference conditions, in GPa
    :param k0p: first pressure-derivative of the bulk modulus, dimensionless
    :param v0: unit-cell volume at the reference conditions, in A^3 (optional)
    :param t0: reference temperature in K (298.15 K by default)
    :param p0: reference pressure in GPa (0 by default, i.e. ambient)
    """

    name = "Vinet"

    def __init__(self,
                 k0: float,
                 k0p: float,
                 v0: float | None = None,
                 t0: float = T_REF,
                 p0: float = P_REF):
        super().__init__(v0=v0, t0=t0, p0=p0)
        self.k0 = float(k0)
        self.k0p = float(k0p)

    def pressure(self,
                 volume: float | None = None,
                 temperature: float | None = None,
                 ratio: float | None = None) -> float:
        """Analytical Vinet pressure from the unit-cell volume.

        :param volume: absolute unit-cell volume in A^3 (requires ``v0``)
        :param temperature: ignored, the model is isothermal
        :param ratio: relative volume V/V0, alternative to ``volume``
        :return: pressure in GPa
        """
        x = self._resolve_ratio(volume, ratio) ** (1.0 / 3.0)
        eta = 1.5 * (self.k0p - 1.0)
        return self.p0 + 3.0 * self.k0 * (1.0 - x) / (x * x) * exp(eta * (1.0 - x))

    def volume_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative unit-cell volume V/V0 at the given pressure.

        Numerical inversion of the analytical :meth:`pressure`.

        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param temperature: ignored, the model is isothermal
        :return: V/V0, dimensionless
        """
        pressure, temperature = self._reference_conditions(pressure, temperature)
        return self._invert_ratio(pressure, temperature)


class BirchMurnaghan(EquationOfState):
    """Birch-Murnaghan equation of state, third order, isothermal.

    .. math::

        P = P_0 + \\frac{3}{2} K_0 (\\eta^7 - \\eta^5)
        \\left[1 + \\frac{3}{4}(K_0'-4)(\\eta^2-1)\\right]
        \\qquad \\eta = (V_0/V)^{1/3}

    The de-facto standard of the high-pressure community, based on a series
    expansion of the Eulerian finite strain. With ``k0p = 4`` the third-order
    term vanishes and this is the second-order (two parameters) model.
    The pressure is analytical in V, the volume is obtained by numerical
    inversion. Temperature is ignored.

    Reference: F. Birch, Finite elastic strain of cubic crystals,
    Phys. Rev. 71 (1947) 809-824. https://doi.org/10.1103/PhysRev.71.809

    :param k0: isothermal bulk modulus at the reference conditions, in GPa
    :param k0p: first pressure-derivative of the bulk modulus, dimensionless
                (4 corresponds to the second-order truncation)
    :param v0: unit-cell volume at the reference conditions, in A^3 (optional)
    :param t0: reference temperature in K (298.15 K by default)
    :param p0: reference pressure in GPa (0 by default, i.e. ambient)
    """

    name = "Birch-Murnaghan"

    def __init__(self,
                 k0: float,
                 k0p: float = 4.0,
                 v0: float | None = None,
                 t0: float = T_REF,
                 p0: float = P_REF):
        super().__init__(v0=v0, t0=t0, p0=p0)
        self.k0 = float(k0)
        self.k0p = float(k0p)

    def pressure(self,
                 volume: float | None = None,
                 temperature: float | None = None,
                 ratio: float | None = None) -> float:
        """Analytical Birch-Murnaghan pressure from the unit-cell volume.

        :param volume: absolute unit-cell volume in A^3 (requires ``v0``)
        :param temperature: ignored, the model is isothermal
        :param ratio: relative volume V/V0, alternative to ``volume``
        :return: pressure in GPa
        """
        eta2 = self._resolve_ratio(volume, ratio) ** (-2.0 / 3.0)  # (V0/V)^(2/3)
        return self.p0 + 1.5 * self.k0 * (eta2 ** 3.5 - eta2 ** 2.5) \
            * (1.0 + 0.75 * (self.k0p - 4.0) * (eta2 - 1.0))

    def volume_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative unit-cell volume V/V0 at the given pressure.

        Numerical inversion of the analytical :meth:`pressure`.

        :param pressure: pressure in GPa (defaults to the reference pressure p0)
        :param temperature: ignored, the model is isothermal
        :return: V/V0, dimensionless
        """
        pressure, temperature = self._reference_conditions(pressure, temperature)
        return self._invert_ratio(pressure, temperature)


class ThermalExpansion(EquationOfState):
    """Isobaric thermal expansion with a polynomial expansion coefficient.

    The volumetric thermal expansion coefficient follows Fei's parametrization

    .. math::

        \\alpha_V(T) = \\alpha_0 + \\alpha_1 T + \\alpha_2 T^{-2}

    integrated exactly into

    .. math::

        V(T)/V_0 = \\exp\\left(\\int_{T_0}^{T} \\alpha_V(T')\\,dT'\\right)

    With ``alpha1 = alpha2 = 0`` this is the constant-coefficient model
    V/V0 = exp(alpha0 (T-T0)); with ``alpha2 = 0`` it is Berman's linear
    model. ``alpha2`` is usually negative (the expansion vanishes at low
    temperature) and the T^-2 term restricts the validity to temperatures
    well above 0 K. Pressure is ignored: the model only holds at the
    reference pressure.

    Reference: Y. Fei, Thermal expansion, in Mineral Physics and
    Crystallography: A Handbook of Physical Constants, AGU Reference Shelf 2
    (1995) 29-44. https://doi.org/10.1029/RF002p0029

    :param alpha0: constant term of the expansion coefficient, in K^-1
    :param alpha1: linear term, in K^-2
    :param alpha2: T^-2 term, in K
    :param v0: unit-cell volume at the reference conditions, in A^3 (optional)
    :param t0: reference temperature in K (298.15 K by default)
    :param p0: reference pressure in GPa (0 by default, i.e. ambient)
    """

    name = "thermal-expansion"

    def __init__(self,
                 alpha0: float,
                 alpha1: float = 0.0,
                 alpha2: float = 0.0,
                 v0: float | None = None,
                 t0: float = T_REF,
                 p0: float = P_REF):
        super().__init__(v0=v0, t0=t0, p0=p0)
        self.alpha0 = float(alpha0)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)

    def alpha(self, temperature: float | None = None) -> float:
        """Volumetric thermal expansion coefficient at the given temperature.

        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: alpha_V in K^-1
        """
        _pressure, temperature = self._reference_conditions(None, temperature)
        return self.alpha0 + self.alpha1 * temperature + self.alpha2 / (temperature * temperature)

    def _integral(self, temperature: float) -> float:
        """Antiderivative of alpha_V, evaluated at the given temperature"""
        return self.alpha0 * temperature \
            + 0.5 * self.alpha1 * temperature * temperature \
            - self.alpha2 / temperature

    def volume_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative unit-cell volume V/V0 at the given temperature.

        :param pressure: ignored, the model is isobaric
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: V/V0, dimensionless
        """
        pressure, temperature = self._reference_conditions(pressure, temperature)
        if temperature <= 0.0:
            raise ValueError(f"Invalid temperature: {temperature} K")
        return exp(self._integral(temperature) - self._integral(self.t0))


class LatticeExpansion(EquationOfState):
    """Isobaric thermal expansion of the lattice parameter as a polynomial.

    .. math::

        a(T)/a_0 = 1 + c_1 (T-T_0) + c_2 (T-T_0)^2 + ... \\qquad V/V_0 = (a/a_0)^3

    This is the form in which reference lattice parameters of cubic
    calibrants (Si, CeO2, LaB6, ...) are usually published, e.g. in chapter 4
    of Powder Diffraction: Theory and Practice: the coefficients can be used
    directly. It assumes an isotropic expansion, exact for cubic lattices.
    Pressure is ignored: the model only holds at the reference pressure.

    :param coefficients: polynomial coefficients [c1, c2, ...] of the
                         relative lattice parameter in (T-T0), in K^-1, K^-2, ...
    :param v0: unit-cell volume at the reference conditions, in A^3 (optional)
    :param t0: reference temperature in K (298.15 K by default)
    :param p0: reference pressure in GPa (0 by default, i.e. ambient)
    """

    name = "lattice-expansion"

    def __init__(self,
                 coefficients: list,
                 v0: float | None = None,
                 t0: float = T_REF,
                 p0: float = P_REF):
        super().__init__(v0=v0, t0=t0, p0=p0)
        self.coefficients = [float(c) for c in coefficients]

    def linear_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative lattice parameter a/a0 at the given temperature.

        Overridden with the exact polynomial (rather than the cubic root of
        the volume ratio).

        :param pressure: ignored, the model is isobaric
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: a/a0 = d/d0, dimensionless
        """
        pressure, temperature = self._reference_conditions(pressure, temperature)
        delta = temperature - self.t0
        ratio = 1.0
        power = 1.0
        for coef in self.coefficients:
            power *= delta
            ratio += coef * power
        return ratio

    def volume_ratio(self, pressure: float | None = None, temperature: float | None = None) -> float:
        """Calculate the relative unit-cell volume V/V0 = (a/a0)^3 at the given temperature.

        :param pressure: ignored, the model is isobaric
        :param temperature: temperature in K (defaults to the reference temperature t0)
        :return: V/V0, dimensionless
        """
        return self.linear_ratio(pressure, temperature) ** 3
