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
__date__ = "26/05/2025"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import posixpath
import copy
import json
import logging
logger = logging.getLogger(__name__)
from collections import namedtuple
import numpy
import h5py
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
    :param points: Number of points (i.e. numberof steps + 1)
    :param name: Name of the motor
    """
    start: float = None
    stop: float = None
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
        if self.points < 2:
            return
        return (self.stop-self.start)/self.steps

    @property
    def steps(self):
        return self.points-1

    @classmethod
    def _parse_old_config(cls, dico, prefix="slow"):
        self = cls()
        if dico.get("nbpt_" + prefix):
            self.points = int(dico["nbpt_" + prefix])
        elif dico.get("npt_" + prefix):
            self.points = int(dico["npt_" + prefix])
        elif dico.get(prefix+"_motor_points"):
            self.points = int(dico[prefix+"_motor_points"])
        if dico.get(prefix + "_motor_name"):
            self.name = str(dico[prefix + "_motor_name"])
        if dico.get(prefix + "_motor_range"):
            rng = dico[prefix + "_motor_range"]
            self.start = float(rng[0])
            self.stop = float(rng[-1])
        return self


def parse_bliss(filename, motors, transpose=False):
    """Parse a bliss master file (HDF5) for 2 motors which were scanned
    and calculate the frame index of each pixel, returned as a map.

    :param filename: name of the Bliss master file
    :param motors: 2-tuple of 1d-datasets names in the masterfile. Both have the same shape.
    :param transpose: set to True to have the fast dimention along y instead of x.
    :return: map_ptr, [MotorRange_y, MotorRange_x]
    """
    def build_dict(posi):
        "histogram like procedure with dict"
        dico = {}
        for i in posi:
            if i in dico:
                dico[i]+=1
            else:
                dico[i]=0
        return dico

    if len(motors) != 2:
        raise RuntimeError("Expected a list of 2 path pointing to 1d datasets, got {motors}!")

    #read all motor position datasets
    motor_raw = {}
    with h5py.File(filename) as h:
        for name in motors:
            motor_raw[name] = h[name][()]

    if (motor_raw[motors[0]].shape != motor_raw[motors[1]].shape):
        raise RuntimeError("Expected a list of 2 1d datasets, with the same shape!")

    ranges = {}
    for name in motor_raw:
        dico = build_dict(motor_raw[name])
        mr = MotorRange(name=posixpath.basename(name))
        mr.start = min(dico.keys())
        mr.stop = max(dico.keys())
        mr.points = len(dico.keys())
        ranges[name] = mr

    # The slow motor is often still, so its speed is null, much more often than the fast motor.
    ordered_names = []
    dm1 = numpy.diff(motor_raw[motors[0]])
    dm2 = numpy.diff(motor_raw[motors[1]])
    if (dm1==0).sum()>(dm2==0).sum():
        ordered_names = motors
    else:
        ordered_names = motors[-1::-1]
    ordered_ranges = [ranges[i] for i in ordered_names]

    slow = ordered_ranges[0]
    fast = ordered_ranges[1]
    slow_pos = motor_raw[ordered_names[0]]
    fast_pos = motor_raw[ordered_names[1]]

    #Build the map
    map_ptr = numpy.zeros((slow.points, fast.points), dtype=numpy.int32)
    slow_idx = numpy.round((slow_pos - slow.start) / slow.step_size).astype(int)
    fast_idx = numpy.round((fast_pos - fast.start) / fast.step_size).astype(int)
    map_ptr[slow_idx, fast_idx] = numpy.arange(slow.points*fast.points, dtype=int)

    if transpose:
        return map_ptr.T, ordered_ranges[-1::-1]
    else:
        return map_ptr, ordered_ranges


DataSetNT = namedtuple("DataSet", ("path", "h5", "nframes", "shape"), defaults=[None, None, None])

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
        :return: common directory structure
        """
        l = len(self)
        if l==0:
            return ""
        elif l==1:
            return os.path.normpath(self[0].path)
        else:
            common = os.path.commonpath([i.path for i in self])
            return common + os.sep

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
            if isinstance(ds, DataSet):
                self.append(ds)
            elif isinstance(ds, dict):
                self.append(DataSet(**ds))
            else:
                # list or tuple
                self.append(DataSet.from_tuple(ds))
        return self

    @property
    def nframes(self):
        return sum(len(i) for i in self)

    @property
    def shape(self):
        "Common shape. Emits a warning is unconsistant or None when empty"
        if len(self) == 0:
            return
        shape = self[0].shape  # could be None
        for j in self:
            if j.shape != shape:
                logger.warning(f"Shape of images in file {j.path} is {j.shape}, differs from first which is {shape}")

        return shape


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
        "npt_azim": "ai.nbpt_azim",
        }

    def __repr__(self):
        return json.dumps(self.as_dict(), indent=4)

    def as_dict(self):
        """Like asdict, but with some more features:
        * Handle dedicated nested dataclasses
        """
        dico = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if key in self.ENFORCED:
                methods = dir(value)
                if "as_dict" in methods:     # dataclass
                    dico[key] = value.as_dict()
                elif "as_str" in methods:
                    dico[key] = value.as_str()
                elif "_asdict" in methods:   # namedtuple
                    dico[key] = tuple(value)
                elif "serialize" in methods:
                    dico[key] = value.serialize()
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

        # Pre-normalize some keys ...
        if "diffmap_config_version" not in dico:
            old_config = True
            slow = MotorRange._parse_old_config(dico, "slow")
            fast = MotorRange._parse_old_config(dico, "fast")
        else:
            old_config = False

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
                        if "from_dict" in dir(klass):
                            to_init[key] = klass.from_dict(value)
                        else:
                            to_init[key] = klass(**value)
                    else:
                        logger.warning(f"Unable to construct class {klass} with input {value} for key {key} in WorkerConfig.from_dict()")
                        to_init[key] = value
                else:
                    to_init[key] = value
        self = cls(**to_init)
        if old_config:
            self.fast_motor = fast
            self.fast_motor = slow

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

    @property
    def slow_motor_points(self):
        return None if self.slow_motor is None else self.slow_motor.points
    @slow_motor_points.setter
    def slow_motor_points(self, value):
        if self.slow_motor is None:
            self.slow_motor = MotorRange()
        self.slow_motor.points = value

    @property
    def fast_motor_points(self):
        return None if self.fast_motor is None else self.fast_motor.points
    @fast_motor_points.setter
    def fast_motor_points(self, value):
        if self.fast_motor is None:
            self.fast_motor = MotorRange()
        self.fast_motor.points = value
