# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2019 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, division

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/05/2019"
__docformat__ = 'restructuredtext'


import collections
import time
import json
import logging

_logger = logging.getLogger(__name__)

from .. import detectors
from ..third_party import six


class PoniFile(object):

    def __init__(self, data=None):
        self._detector = None
        self._dist = None
        self._poni1 = None
        self._poni2 = None
        self._rot1 = None
        self._rot2 = None
        self._rot3 = None
        self._wavelength = None
        if data is None:
            pass
        elif isinstance(data, dict):
            self.read_from_dict(data)
        elif isinstance(data, six.string_types):
            self.read_from_file(data)
        else:
            self.read_from_duck(data)

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
        """
        version = int(config.get("poni_version", 1))

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

        elif version == 2:
                detector_name = config["detector"]
                detector_config = config["detector_config"]
                self._detector = detectors.detector_factory(detector_name, detector_config)
        else:
            raise RuntimeError("PONI file verison %s too recent. Upgrade pyFAI.", version)

        if "distance" in config:
            self._dist = float(config["distance"])
        if "poni1" in config:
            self._poni1 = float(config["poni1"])
        if "poni2" in config:
            self._poni2 = float(config["poni2"])
        if "rot1" in config:
            self._rot1 = float(config["rot1"])
        if "rot2" in config:
            self._rot2 = float(config["rot2"])
        if "rot3" in config:
            self._rot3 = float(config["rot3"])
        if "wavelength" in config:
            self._wavelength = float(config["wavelength"])

    def read_from_duck(self, duck):
        """Initialize the object using an object providing the same API.

        The duck object must provide dist, poni1, poni2, rot1, rot2, rot3,
        wavelength, and detector.
        """
        # TODO: It would be good to test attribute types
        self._dist = duck.dist
        self._poni1 = duck.poni1
        self._poni2 = duck.poni2
        self._rot1 = duck.rot1
        self._rot2 = duck.rot2
        self._rot3 = duck.rot3
        self._wavelength = duck.wavelength
        self._detector = duck.detector

    def write(self, fd):
        """Write this object to an open stream.
        """
        fd.write(("# Nota: C-Order, 1 refers to the Y axis,"
                 " 2 to the X axis \n"))
        fd.write("# Calibration done at %s\n" % time.ctime())
        fd.write("poni_version: 2\n")
        detector = self.detector
        fd.write("Detector: %s\n" % detector.__class__.__name__)
        fd.write("Detector_config: %s\n" % json.dumps(detector.get_config()))

        fd.write("Distance: %s\n" % self._dist)
        fd.write("Poni1: %s\n" % self._poni1)
        fd.write("Poni2: %s\n" % self._poni2)
        fd.write("Rot1: %s\n" % self._rot1)
        fd.write("Rot2: %s\n" % self._rot2)
        fd.write("Rot3: %s\n" % self._rot3)
        if self._wavelength is not None:
            fd.write("Wavelength: %s\n" % self._wavelength)

    def as_dict(self):
        config = collections.OrderedDict([("poni_version", 2)])
        config["detector"] = self.detector.__class__.__name__
        config["detector_config"] = self.detector.get_config()
        config["dist"] = self._dist
        config["poni1"] = self._poni1
        config["poni2"] = self._poni2
        config["rot1"] = self._rot1
        config["rot2"] = self._rot2
        config["rot3"] = self._rot3
        if self._wavelength:
            config["wavelength"] = self._wavelength
        return config

    @property
    def detector(self):
        """:rtype: Union[None,float]"""
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
