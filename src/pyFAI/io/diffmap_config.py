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
import copy
import json
import logging
logger = logging.getLogger(__name__)
from collections import namedtuple
from .tree import TreeItem
from .integration_config import dataclass, ClassVar, WorkerConfig, fields, asdict
from .nexus import is_hdf5

#constants:
CURRENT_VERSION = 1  # former version were unassigned


@dataclass
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
        return f"{self.name}: MotorRange({self.start:.3f}, {self.stop:.3f}, {self.points})"
    @property
    def step_size(self):
        if self.points < 1:
            return
        return (self.stop-self.start)/self.points


DataSetNT = namedtuple("DataSet", ("path", "h5", "nframes", "shape"), defaults=[None])

@dataclass
class DataSet:
    path: str
    h5: str = None
    nframes: int = None
    shape: tuple = None

    def __repr__(self):
        return f"Dataset('{self.path}', '{self.h5}', {self.nframes}, {self.shape})"

    def as_tuple(self):
        return DataSetNT(self.path, self.h5, self.nframes, self.shape)

    @classmethod
    def from_tuple(cls, tpl):
        return cls(*tpl)

    def is_hdf5(self):
        """Return True if the object is hdf5"""
        if self.h5 is None:
            self.h5 = is_hdf5(self.path)
        return bool(self.h5)

    def __len__(self):
        return self.nframes or 1


class ListDataSet(list):

    def commonroot(self):
        """
        :return: common directory
        """
        ll = [j.path.split(os.sep) for j in self]
        common = os.path.commonprefix(ll)
        if common:
            return os.sep.join(common + [""])

    def as_tree(self, sep=os.path.sep):
        """Convert the list into a tree

        :param sep: separator in the filenames
        :return: Root of the tree
        """
        prefix = self.commonroot()
        root = TreeItem()
        common = TreeItem(prefix, root)
        lprefix = len(prefix) if prefix else 0
        for dataset in self:
            base = dataset.path[lprefix:]
            elts = base.split(sep)
            element = common
            for item in elts:
                child = element.get(item)
                if not child:
                    child = TreeItem(item, element)
                element = child
        return root

    def empty(self):
        while self:
            self.pop()

    def serialize(self):
        "returns a list of serialized datasets"
        res = []
        for i in self:
            res.append(i.as_tuple())
        return res #[i.as_tuple() for i in self]

    @classmethod
    def from_serialized(cls, lst):
        "Alternative constructor with deserialization"
        self = cls()
        for ds in lst:
            if isinstance(ds, dict):
                self.append(DataSet(**ds))
            else:
                self.append(DataSet.from_tuple(ds))
        return self

@dataclass
class DiffmapConfig:
    """Class with the configuration from the diffmap experiment."""
    diffmap_config_version: int = CURRENT_VERSION
    experiment_title: str = ""
    slow_motor: MotorRange = None
    fast_motor: MotorRange = None
    offset: int = 0
    zigzag_scan: bool = False
    ai: WorkerConfig = None
    input_data: ListDataSet = None
    output_file: str = None


    OPTIONAL: ClassVar[list] = []
    GUESSED: ClassVar[list] = []
    ENFORCED: ClassVar[list] = ["slow_motor", "fast_motor", "ai", "input_data"]
    DEPRECATED: ClassVar[dict] = {
        "npt_fast": "fast_motor.points",
        "npt_slow": "slow_motor.points",
        "nbpt_fast": "fast_motor.points",
        "nbpt_slow": "slow_motor.points",
        "fast_motor_points": "fast_motor.points",
        "slow_motor_points": "slow_motor.points",
        "fast_motor_name": "fast_motor.name",
        "slow_motor_name": "slow_motor.name",
        "fast_motor_range": "fast_motor.start|stop",
        "slow_motor_range": "slow_motor.start|stop",
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
                methods = dir(value)
                if "as_dict" in methods:     # dataclass
                    dico[key] = value.as_dict()
                elif "as_str" in methods:
                    dico[key] = value.as_str()
                elif "_asdict" in methods:   # namedtuple
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
                    elif isinstance(value, (list, tuple)):
                        if "from_serialized" in dir(klass):
                            to_init[key] = klass.from_serialized(value)
                        else:
                            to_init[key] = klass(*value)
                    elif isinstance(value, klass):
                        to_init[key] = value
                    elif isinstance(value, dict):
                        to_init[key] = klass(**value)
                    else:
                        logger.warning(f"Unable to construct class {klass} with input {value} for key {key} in WorkerConfig.from_dict()")
                        to_init[key] = value
                else:
                    to_init[key] = value
        self = cls(**to_init)

        for key in cls.GUESSED:
            if key in dico:
                dico.pop(key)
        for key in cls.OPTIONAL+list(cls.DEPRECATED.keys()):
            if key in dico:
                value = dico.pop(key)
                self.__setattr__(key, value)

        if len(dico):
            logger.warning("Those are the parameters which have not been converted !" + "\n".join(f"{key}: {val}" for key, val in dico.items()))
        return self

    def save(self, filename):
        """Dump the content of the dataclass as JSON file"""
        with open(filename, "w") as w:
            w.write(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def from_file(cls, filename: str):
        """load the content of a JSON file and provide a dataclass instance"""
        with open(filename, "r") as f:
            dico = json.loads(f.read())
        return cls.from_dict(dico, inplace=True)

    # Compatibility layer !
    @property
    def npt_fast(self):
        return None if self.fast_motor is None else self.fast_motor.points
    @npt_fast.setter
    def npt_fast(self, value):
        if self.fast_motor is None:
            self.fast_motor = MotorRange()
        self.fast_motor.points = value

    @property
    def npt_slow(self):
        return None if self.slow_motor is None else self.slow_motor.points
    @npt_slow.setter
    def npt_slow(self, value):
        if self.slow_motor is None:
            self.slow_motor = MotorRange()
        self.slow_motor.points = value
    nbpt_fast = npt_fast
    nbpt_slow = npt_slow

    @property
    def fast_motor_name(self):
        return None if self.fast_motor is None else self.fast_motor.name
    @fast_motor_name.setter
    def fast_motor_name(self, value):
        if self.fast_motor is None:
            self.fast_motor = MotorRange()
        self.fast_motor.name = value

    @property
    def slow_motor_name(self):
        return None if self.slow_motor is None else self.slow_motor.name
    @slow_motor_name.setter
    def slow_motor_name(self, value):
        if self.slow_motor is None:
            self.slow_motor = MotorRange()
        self.slow_motor.name = value

    @property
    def fast_motor_range(self):
        return None if self.fast_motor is None else (self.fast_motor.start, self.fast_motor.stop)
    @fast_motor_range.setter
    def fast_motor_range(self, value):
        if self.fast_motor is None:
            self.fast_motor = MotorRange()
        self.fast_motor.start, self.fast_motor.stop = value

    @property
    def slow_motor_range(self):
        return None if self.slow_motor is None else (self.slow_motor.start, self.slow_motor.stop)
    @slow_motor_range.setter
    def slow_motor_range(self, value):
        if self.slow_motor is None:
            self.slow_motor = MotorRange()
        self.slow_motor.start, self.slow_motor.stop = value
