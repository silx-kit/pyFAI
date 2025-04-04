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

"""Module with the configuration dataclass for diffraction mapping experiments"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/04/2025"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import time
import logging
logger = logging.getLogger(__name__)
from typing import NamedTuple
from .integration_config import mydataclass, WorkerConfig
CURRENT_VERSION = 1  # former version were unassigned


@mydataclass
class MotorRange:
    """ This object represents a motor range 

    :param start: Begining of the movement 
    :param stop: End of the movement, included
    :param step: Number of points (i.e. numberof steps + 1)
    :param name: Name of the motor
    """
    start: float = 0.0
    stop: float = 1.0
    points: int = 0
    name: str = ""

    @classmethod
    def _fromdict(cls, dico):
        "Mirror of _asdict: take the dict and populate the tuple to be returned"
        try:
            obj = cls(**dico)
        except TypeError as err:
            logger.warning("TypeError: %s", err)
            obj = cls(**{key: dico[key] for key in [i for i in cls._fields if i in dico]})
        return obj

    def as_dict(self):
        """Like asdict, but possibly with some more features
        """
        return dict(asdict(self))

    def __repr__(self):
        return f"{self.name}: MotorRange({self.start:.3f}, {self.start:.3f}, {self.points}"

    @property
    def step_size(self):
        if self.points < 1:
            return  
        return (self.stop-self.start)/self.points


@mydataclass
class DiffmapConfig:
    """Class with the configuration from the diffmap experiment."""
    version: int = CURRENT_VERSION
    experiment_title: str = ""
    slow_motor: MotorRange = None
    fast_motor: MotorRange = None
    offset: int = 0
    zigzag_scan: bool = False
    ai: WorkerConfig = None
    input_data: list = []
    output_file: str = None


    OPTIONAL: ClassVar[list] = []
    GUESSED: ClassVar[list] = []
    ENFORCED: ClassVar[list] = ["slow_motor", "fast_motor", "ai"]
    DEPRECATED: ClassVar[dict] = {
        "npt_fast": "fast_motor.points",
        "npt_slow": "slow_motor.points",
        "nbpt_fast": "fast_motor.points",
        "nbpt_slow": "slow_motor.points",
        "fast_motor_points": "fast_motor.points",
        "slow_motor_points": "slow_motor.points",
        "fast_motor_name"
        "slow_motor_name"
        "npt_rad": "ai.nbpt_rad",
        "npt_azim": "ai.nbpt_rad",
        }
    
    def __repr__(self):
        return json.dumps(self.as_dict(), indent=4)

    def as_dict(self):
        """Like asdict, but with some more features:
        * Handle dedicated nested dataclasses
        """
        dico = {}
        for key, value in asdict(self).items():
            if key in self.ENFORCED:
                if "as_dict" in dir(value):  # dataclass
                    dico[key] = value.as_dict()
                elif "as_str" in dir(value):
                    dico[key] = value.as_str()
                elif "_asdict" in dir(value):  # namedtuple
                    dico[key] = tuple(value)
                else:
                    dico[key] = value
            else:
                dico[key] = value
        return dico

    @classmethod
    def from_dict(cls, dico, inplace=False):
        """Alternative constructor,
            * Normalize the dico (i.e. upgrade to the most recent version)
            * accepts everything which is in OPTIONAL

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
                if key in cls.ENFORCED:
                    "Enforce a specific class type"
                    klass = field.type
                    if value is None:
                        to_init[key] = value
                    elif isinstance(value, klass):
                        to_init[key] = value
                    elif isinstance(value, dict):
                        to_init[key] = klass(**value)
                    elif isinstance(value, (list, tuple)):
                        to_init[key] = klass(*value)
                    else:
                        _logger.warning(f"Unable to construct class {klass} with input {value} for key {key} in WorkerConfig.from_dict()")
                        to_init[key] = value
                else:
                    to_init[key] = value
        self = cls(**to_init)

        for key in cls.GUESSED:
            if key in dico:
                dico.pop(key)
        for key in cls.OPTIONAL:
            if key in dico:
                value = dico.pop(key)
                self.__setattr__(key, value)

        if len(dico):
            _logger.warning("Those are the parameters which have not been converted !" + "\n".join(f"{key}: {val}" for key, val in dico.items()))
        return self

    def save(self, filename):
        """Dump the content of the dataclass as JSON file"""
        with open(filename, "w") as w:
            w.write(json.dumps(self.as_dict(), indent=2))