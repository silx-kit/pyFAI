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

"""Sensors description for detectors:

Defines Si_, CdTe_ & GaAs_MATERIAL
"""


__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "03/10/2025"
__status__ = "stable"

import os
import logging
import json
import copy
from math import exp
from collections import namedtuple
from ..containers import dataclass, fields
import numpy
from ..resources import resource_filename
from ..utils.stringutil import to_eng, from_eng

EnergyRange = namedtuple("EnergyRange", ["min", "max"])
logger = logging.getLogger(__name__)

class SensorMaterial:
    """This class represents a sensor material and
    is able to produce the linear absorption coefficient for it as function
    of the considered energy.
    """
    SCALES = {"cm":1,
              "mm":0.1,
              "m":100,
              "µm":1e-4,
              "um":1e-4}

    def __init__(self, name:str, density:float):
        """Constructor of the class
        :param name: name of the sensor material
        :param density: density of the material in g/cm³
        """
        self.name = name
        self.rho = density
        self._data = {} # range in keV, data as numpy array
        self.init()

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', {self.rho})"

    def __str__(self):
        return f"{self.name}-{self.__class__.__name__}"

    def init(self):
        """Read the sensor data and split them in chunks"""
        def end_block(block):
            descr = EnergyRange(float(block[0].split()[0]) * 1e3,
                                float(block[-1].split()[0]) * 1e3)
            data = numpy.vstack(tuple(numpy.fromstring(line, dtype=float, sep=" ") for line in block))
            data[:,0] *= 1e3 # MeV to keV for the energy
            self._data[descr] = data

        filename = resource_filename(os.path.join("sensors", self.name+".abs"))
        if not os.path.exists(filename):
            raise FileNotFoundError("No sensor material for `{self.name}`: {filename} does not exist !")
        block = []
        with open(filename, encoding="utf-8") as fd:
            for line in fd:
                if "#" in line:
                    continue
                if "K" in line or "L" in line or "M" in line:
                    end_block(block)
                    block = [" ".join(line.split()[-3:])]
                else:
                    block.append(line.strip())
            else:
                end_block(block)

    def _find_range(self, energy):
        """Helper method to find the right block"""
        for descr in self._data:
            if energy>=descr.min and energy<descr.max:
                return descr
        else:
            raise RuntimeError(f"Energy {energy} outside of tabulated range for {self}")

    def _scale(self, unit:str):
        """Helper function that return the scale multiplier"""
        if "^" in unit:
            unit = unit.split("^")[0]
        return self.SCALES[unit]

    def mu(self, energy:float, unit:str="cm^-1") -> float:
        """calculate the linear absorption coefficient

        :param energy: in keV
        :return: linear absorption coefficient in the given unit
        """
        data = self._data[self._find_range(energy)]
        return numpy.interp(energy, data[:,0], data[:,1]) * self.rho * self._scale(unit)

    def mu_en(self, energy:float=None, unit:str="cm^-1") -> float:
        """calculate the linear absorption coefficient based on the deposited energy

        :param energy: in keV
        :return: linear absorption coefficient of deposited energy in the given unit
        """
        data = self._data[self._find_range(energy)]
        return numpy.interp(energy, data[:,0], data[:,2]) * self.rho * self._scale(unit)

    def absorbance(self, energy:float, length: float, unit:str="m") -> float:
        """calculate the efficiency of a slab of sensor for absorbing the radiation

        :param energy: in keV
        :param length: thickness of the sensor
        :param unit: unit for the thickness of the detector
        :return: efficiency of the sensor between 0 (no absorbance) and 1 (all photons are absorbed)
        """
        return 1.0-exp(-self.mu(energy, unit)*length)


# For the record: some classical sensors materials
ALL_MATERIALS = {}
ALL_MATERIALS["Si"] = Si_MATERIAL = SensorMaterial("Si", density=2.329)
ALL_MATERIALS["Ge"] = Ge_MATERIAL = SensorMaterial("Ge", density=5.327)
ALL_MATERIALS["CdTe"] = CdTe_MATERIAL = SensorMaterial("CdTe", density=5.85)
ALL_MATERIALS["GaAs"] = GaAs_MATERIAL = SensorMaterial("GaAs", density=5.3176)
ALL_MATERIALS["Gd2O2S"] = Gd2O2S_MATERIAL = SensorMaterial("Gd2O2S", density=7.32)
ALL_MATERIALS["BaFBr0.85I0.15"] = BaFBr085I015_MATERIAL = SensorMaterial("BaFBr0.85I0.15", density=3.18) 
ALL_MATERIALS["Se"] = Se_MATERIAL = SensorMaterial("Se", density=4.26)


@dataclass
class SensorConfig:
    "class for configuration of a sensor"
    material: SensorMaterial|str
    thickness: float=None

    def __repr__(self):
        return json.dumps(self.as_dict(), indent=4)

    def __str__(self):
        name = self.material.name if isinstance(self.material, SensorMaterial) else self.material
        thick = to_eng(self.thickness, space="")+"m" if self.thickness else "\N{INFINITY}"
        return f"{name},{thick}"

    def as_dict(self):
        """Like asdict, but with some more features:
        """
        dico = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if isinstance(value, SensorMaterial):
                dico[key] = value.name
            elif value:
                dico[key] = value
        return dico

    @classmethod
    def from_dict(cls, dico:dict, inplace:bool=False):
        """Alternative constructor

        :param dico: dict with the config
        :param in-place: modify the dico in place ?
        :return: instance of the dataclass
        """
        if not inplace:
            dico = copy.copy(dico)

        to_init = {}
        for field in fields(cls):
            key = field.name
            if key in dico:
                value = dico.pop(key)
                if key=="material" and isinstance(value, str):
                    value = ALL_MATERIALS.get(value) or value
                to_init[key] = value
        return cls(**to_init)

    @classmethod
    def parse(cls, txt:str):
        """Alternate constructor from strings like `Si,320µm`"""
        if "," in txt:
            material,thick = txt.split(",")
        else:
            material = txt.strip()
            thick = "∞"
        dico = {"material": material.strip()}
        if thick != "∞":
            dico["thickness"] = from_eng(thick)
        return cls.from_dict(dico, inplace=True)
