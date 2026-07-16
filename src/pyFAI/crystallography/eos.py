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
        if pressure == self.p0:
            return 1.0

        def fun(x):
            return self.pressure(ratio=x ** 3) - pressure

        if pressure > self.p0:  # compression: x in ]0, 1[
            return brentq(fun, 1e-3, 1.0) ** 3
        # dilatation (tension): x > 1, only defined down to the spinodal
        # where P(x) reaches its minimum and stops decreasing
        x_hi = 1.0
        residual = fun(x_hi)
        while True:
            x_next = x_hi + 0.05
            f_next = fun(x_next)
            if f_next <= 0.0:
                return brentq(fun, x_hi, x_next) ** 3
            if f_next >= residual:
                raise ValueError(f"Pressure {pressure} GPa is beyond the spinodal (tension limit) of {self!r}")
            x_hi, residual = x_next, f_next
