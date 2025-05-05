# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module function to manage poni files.
"""

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/04/2025"
__docformat__ = 'restructuredtext'

import collections
import time
import json
import pathlib
import logging
_logger = logging.getLogger(__name__)
import numpy
from .. import detectors
from ..utils import decorators

try:
    from ..gui.model.GeometryModel import GeometryModel
except ImportError:
    GeometryModel = None


class PoniFile(object):
    API_VERSION = 2.1  # valid version are 1, 2, 2.1

    def __init__(self, data=None, **kwargs):
        self._detector = None
        self._dist = None
        self._poni1 = None
        self._poni2 = None
        self._rot1 = None
        self._rot2 = None
        self._rot3 = None
        self._wavelength = None
        if data is None:
            if kwargs:
                data = kwargs
            else:
                return
        elif kwargs:
            raise ValueError("Passing both data and keyword arguments is not supported")
        if isinstance(data, dict):
            self.read_from_dict(data)
        elif isinstance(data, (str, pathlib.Path)):
            self.read_from_file(data)
        elif GeometryModel and isinstance(data, GeometryModel):
            self.read_from_geometryModel(data)
        else:
            self.read_from_duck(data)

    def __repr__(self):
        return json.dumps(self.as_dict(), indent=4)

    def make_headers(self, type_="list"):
        "Generate a header for files, as list or dict or str"
        if type_ == "dict":
            return self.as_dict()
        elif type_ == "str":
            return str(self)
        elif type_ == "list":
            return str(self).split("\n")
        else:
            _logger.error(f"Don't know how to handle type {type_} !")

    def read_from_file(self, filename):
        data = collections.OrderedDict()
        with open(filename) as opened_file:
            for line in opened_file:
                if line.startswith("#") or (":" not in line):
                    continue
                words = line.split(":", 1)

                key = words[0].strip().lower()
                try:
                    value = words[1].strip()
                except Exception as error:  # IGNORE:W0703:
                    _logger.error("Error %s with line: %s", error, line)
                data[key] = value
        self.read_from_dict(data)

    def read_from_dict(self, config):
        """Initialize this object using a dictionary.

        .. note:: The dictionary is versionned.
        Version:

            * 1: Historical version (i.e. unversioned)
            * 2: store detector and detector_config instead of pixelsize1, pixelsize2 and splinefile
            * 2.1: manage orientation of detector in detector_config

        """
        # Patch for worker version 4
        if "poni" in config and config.get("version", 0) > 3:
            config = config.get("poni", {})

        version = float(config.get("poni_version", 1))
        if "detector_config" in config:
            if "orientation" in config["detector_config"]:
                version = max(version, 2.1)
            else:
                version = max(version, 2)
        if version >= 2 and "detector_config" not in config:
            _logger.error("PoniFile claim to be version 2 but contains no `detector_config` !!!")

        if version == 2.1 and "orientation" not in config.get("detector_config", {}):
            _logger.error("PoniFile claim to be version 2.1 but contains no detector orientation !!!")
        self.API_VERSION = version

        if version == 1:
            # Handle former version of PONI-file
            if "detector" in config:
                self._detector = detectors.detector_factory(config["detector"])
            else:
                self._detector = detectors.Detector()

            if "pixelsize1" in config or "pixelsize2" in config:
                if isinstance(self._detector, detectors.NexusDetector):
                    # NexusDetector is already set
                    pass
                elif self._detector.force_pixel and ("pixelsize1" in config) and ("pixelsize2" in config):
                    pixel1 = float(config["pixelsize1"])
                    pixel2 = float(config["pixelsize2"])
                    self._detector = self._detector.__class__(pixel1=pixel1, pixel2=pixel2)
                else:
                    self._detector = detectors.Detector()
                    if "pixelsize1" in config:
                        self._detector.pixel1 = float(config["pixelsize1"])
                    if "pixelsize2" in config:
                        self._detector.pixel2 = float(config["pixelsize2"])

            if "splinefile" in config:
                if config["splinefile"].lower() != "none":
                    self._detector.set_splineFile(config["splinefile"])

        elif version in (2, 2.1):
                detector_name = config["detector"]
                detector_config = config.get("detector_config")
                self._detector = detectors.detector_factory(detector_name, detector_config)
        else:
            raise RuntimeError("PONI file verison %s too recent. Upgrade pyFAI.", version)

        if "distance" in config:
            self._dist = float(config["distance"]) if config["distance"] is not None else None
        elif "dist" in config:
            self._dist = float(config["dist"]) if config["dist"] is not None else None

        self._poni1 = float(config["poni1"]) if config.get("poni1") is not None else None
        self._poni2 = float(config["poni2"]) if config.get("poni2") is not None else None
        self._rot1 = float(config["rot1"]) if config.get("rot1") is not None else None
        self._rot2 = float(config["rot2"]) if config.get("rot2") is not None else None
        self._rot3 = float(config["rot3"]) if config.get("rot3") is not None else None
        self._wavelength = float(config["wavelength"]) if config.get("wavelength") is not None else None

    def read_from_duck(self, duck):
        """Initialize the object using an object providing the same API.

        The duck object must provide dist, poni1, poni2, rot1, rot2, rot3,
        wavelength, and detector.
        """
        if not (numpy.isreal(duck.dist) and \
                numpy.isreal(duck.poni1) and \
                numpy.isreal(duck.poni2) and \
                numpy.isreal(duck.rot1) and \
                numpy.isreal(duck.rot2) and \
                numpy.isreal(duck.rot3) and \
                numpy.isreal(duck.wavelength)):
            raise RuntimeError(f"Expected dist ({type(duck.dist)}), poni1 ({type(duck.poni1)}), poni2 ({type(duck.poni2)}), "
                               f"rot1 ({type(duck.rot1)}), rot2 ({type(duck.rot2)}), rot3 ({type(duck.rot3)}), wavelength ({type(duck.wavelength)})")
        self._dist = duck.dist
        self._poni1 = duck.poni1
        self._poni2 = duck.poni2
        self._rot1 = duck.rot1
        self._rot2 = duck.rot2
        self._rot3 = duck.rot3
        self._wavelength = duck.wavelength
        self._detector = duck.detector

    def read_from_geometryModel(self, model: GeometryModel, detector=None):
        """Initialize the object from a GeometryModel

        pyFAI.gui.model.GeometryModel.GeometryModel"""
        self._dist = model.distance().value()
        self._poni1 = model.poni1().value()
        self._poni2 = model.poni2().value()
        self._rot1 = model.rotation1().value()
        self._rot2 = model.rotation2().value()
        self._rot3 = model.rotation3().value()
        self._wavelength = model.wavelength().value()
        self._detector = detector

    def write(self, fd, comments=None):
        """Write this object to an open stream.

        :param fd: file descriptor (opened file)
        :param comments: extra comments as a string or a list of strings
        :return: None
        """
        detector = self.detector
        txt = ["# Nota: C-Order, 1 refers to the Y axis, 2 to the X axis",
              f"# Calibration done on {time.ctime()}",
              f"poni_version: {self.API_VERSION}",
              f"Detector: {detector.__class__.__name__ if detector else None}"]
        if self.API_VERSION == 1:
            if not detector.force_pixel:
                txt += [f"pixelsize1: {detector.pixel1}",
                        f"pixelsize2: {detector.pixel2}"]
            if detector.splineFile:
                txt.append(f"splinefile: {detector.splineFile}")
        elif self.API_VERSION >= 2 and detector is not None:
            detector_config = detector.get_config()
            if self.API_VERSION == 2:
                detector_config.pop("orientation")
            txt.append(f"Detector_config: {json.dumps(detector_config)}")

        txt += [f"Distance: {self._dist}",
                f"Poni1: {self._poni1}",
                f"Poni2: {self._poni2}",
                f"Rot1: {self._rot1}",
                f"Rot2: {self._rot2}",
                f"Rot3: {self._rot3}"
                ]
        if self._wavelength is not None:
            txt.append(f"Wavelength: {self._wavelength}")
        if comments:
            if isinstance(comments, str):
                txt.append(f"# {comments}")
            elif isinstance(comments, bytes):
                txt.append(f"# {comments.decode()}")
            else:  # assume it is a list/tuple/set:
                txt += [f"# {comment}" for comment in comments]
        txt.append("")
        fd.write("\n".join(txt))

    def as_dict(self):
        config = {"poni_version": self.API_VERSION,
                "dist": self._dist,
                "poni1": self._poni1,
                "poni2": self._poni2,
                "rot1": self._rot1,
                "rot2": self._rot2,
                "rot3": self._rot3,
                "detector": self._detector.__class__.__name__ if self._detector is not None else None
                }
        if self._detector is not None:
            config["detector_config"] = self._detector.get_config()
        if self._wavelength:
            config["wavelength"] = self._wavelength
        return config

    def as_integration_config(self):
        from .integration_config import WorkerConfig
        wc = WorkerConfig(application="poni",
                          poni=self,
                          nbpt_rad=500,
                          nbpt_azim=360,
                          unit="q_nm^-1",
                          method=("full", "csr", "cython"),
                          normalization_factor=1.0)
        return wc

    @property
    def detector(self):
        """:rtype: Union[None, object]"""
        return self._detector

    @property
    def dist(self):
        """:rtype: Union[None,float]"""
        return self._dist

    @property
    def poni1(self):
        """:rtype: Union[None,float]"""
        return self._poni1

    @property
    def poni2(self):
        """:rtype: Union[None,float]"""
        return self._poni2

    @property
    def rot1(self):
        """:rtype: Union[None,float]"""
        return self._rot1

    @property
    def rot2(self):
        """:rtype: Union[None,float]"""
        return self._rot2

    @property
    def rot3(self):
        """:rtype: Union[None,float]"""
        return self._rot3

    @property
    def wavelength(self):
        """:rtype: Union[None,float]"""
        return self._wavelength

    # Dict-like API, for (partial) compatibility. Avoid using it !
    @decorators.deprecated(reason="Ponifile should not be used as a dict", replacement=None, since_version="2025.02")
    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    @decorators.deprecated(reason="Ponifile should not be used as a dict", replacement=None, since_version="2025.02")
    def __getitem__(self, key):
        return self.__getattribute__(key)

    @decorators.deprecated(reason="Ponifile should not be used as a dict", replacement=None, since_version="2025.02")
    def __contains__(self, key):
        try:
            return self.__getattribute__(key) is not None
        except AttributeError:
            return False

    @decorators.deprecated(reason="Ponifile should not be used as a dict", replacement=None, since_version="2025.02")
    def get(self, key, default=None):
        try:
            return self.__getattribute__(key)
        except:
            return default

    def __eq__(self, other):
        """Checks the equality of two ponifile instances"""
        if not isinstance(other, self.__class__):
            return False
        if ((self._detector != other._detector) or
            (self._dist != other._dist) or
            (self._poni1 != other._poni1) or
            (self._poni2 != other._poni2) or
            (self._rot1 != other._rot1) or
            (self._rot2 != other._rot2) or
            (self._rot3 != other._rot3) or
            (self._wavelength != other._wavelength)):
            return False
        return True
