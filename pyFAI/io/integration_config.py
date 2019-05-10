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


import logging
import six

from . import ponifile
from .. import detectors
from .. import method_registry

_logger = logging.getLogger(__name__)


def _normalize_v1_darkflat_files(config, key):
    """Normalize dark and flat filename list from the version 1 to version 2.
    """
    if key not in config:
        return

    filenames = config[key]
    if filenames is None:
        return

    if isinstance(filenames, list):
        # Already a list, it's fine
        return

    if isinstance(filenames, six.string_types):
        if "," in filenames:
            # Create a list from a coma separated string list
            filenames = filenames.split(",")
            filenames = [f.strip() for f in filenames]
            config[key] = filenames


def _patch_v1_to_v2(config):
    """Rework the config dictionary from version 1 to version 2

    :param dict config: Dictionary reworked inplace.
    """

    value = config.pop("poni", None)
    detector = None
    if value:
        # Use the poni file while it is not overwrited by a key of the config
        # dictionary
        poni = ponifile.PoniFile(value)
        if "wavelength" not in config:
            config["wavelength"] = poni.wavelength
        if "dist" not in config:
            config["dist"] = poni.dist
        if "poni1" not in config:
            config["poni1"] = poni.poni1
        if "poni2" not in config:
            config["poni2"] = poni.poni2
        if "rot1" not in config:
            config["rot1"] = poni.rot1
        if "rot2" not in config:
            config["rot2"] = poni.rot2
        if "rot3" not in config:
            config["rot3"] = poni.rot3
        if "detector" not in config:
            detector = poni.detector

    # detector
    value = config.pop("detector", None)
    if value:
        # NOTE: pixel1/pixel2/splineFile was not parsed here
        detector_name = value.lower()
        detector = detectors.detector_factory(detector_name)

        if detector_name == "detector":
            value = config.pop("pixel1", None)
            if value:
                detector.set_pixel1(value)
            value = config.pop("pixel2", None)
            if value:
                detector.set_pixel2(value)
        else:
            # Drop it as it was not really used
            _ = config.pop("pixel1", None)
            _ = config.pop("pixel2", None)

        splineFile = config.pop("splineFile", None)
        if splineFile:
            detector.set_splineFile(splineFile)

    if detector is not None:
        # Feed the detector as version2
        config["detector"] = detector.__class__.__name__
        config["detector_config"] = detector.get_config()

    _normalize_v1_darkflat_files(config, "dark_current")
    _normalize_v1_darkflat_files(config, "flat_field")

    method = config.get("method", None)
    use_opencl = config.pop("do_OpenCL", False)

    if use_opencl is not None and method is not None:
        if use_opencl:
            _logger.warning("Both 'method' and 'do_OpenCL' are defined. 'do_OpenCL' is ignored.")

    if method is None:
        if use_opencl:
            method = "csr_ocl"
        else:
            method = "splitbbox"
        config["method"] = method

    config["version"] = 2
    config["application"] = "pyfai-integrate"


def _patch_v2_to_v3(config):
    """Rework the config dictionary from version 2 to version 3

    :param dict config: Dictionary reworked inplace.
    """
    old_method = config.pop("method")
    method = method_registry.Method.parsed(old_method)
    config["method"] = method.split, method.algo, method.impl
    config["opencl_device"] = method.target

    config["version"] = 3


def normalize(config, inplace=False):
    """Normalize the configuration file to the one supported internally\
    (the last one).

    :param dict config: The configuration dictionary to read
    :param bool inplace: In true, the dictionary is edited inplace
    :raise ValueError: If the configuration do not match.
    """
    if not inplace:
        config = config.copy()

    version = config.get("version", 1)
    if version == 1:
        # NOTE: Previous way to describe an integration process before pyFAI 0.17
        _patch_v1_to_v2(config)
    version = config["version"]
    if version == 2:
        _patch_v2_to_v3(config)

    application = config.get("application", None)
    if application != "pyfai-integrate":
        raise ValueError("Configuration application do not match. Found '%s'" % application)

    if version > 3:
        _logger.error("Configuration file %d too recent. This version of pyFAI maybe too old to read this configuration", version)

    return config


class ConfigurationReader(object):

    def __init__(self, config):
        self._config = config

    def pop_detector(self):
        """
        Returns the detector stored in the json configuration.

        :rtype: pyFAI.detectors.Detector
        """
        detector_class = self._config.pop("detector", None)
        if detector_class is not None:
            # NOTE: Default way to describe a detector since pyFAI 0.17
            detector_config = self._config.pop("detector_config", None)
            detector = detectors.detector_factory(detector_class, config=detector_config)
            return detector

        return None

    def pop_method(self, default=None):
        """Returns a Method from the method field from the json dictionary.

        :rtype: pyFAI.method_registry.Method
        """
        do_2d = self._config.pop("do_2D", 1)
        dim = 2 if do_2d else 1
        method = self._config.pop("method", default)
        target = self._config.pop("opencl_device", None)

        if isinstance(target, list):
            # Patch list to tuple
            target = tuple(target)

        if method is None:
            method = method_registry.Method(dim, "*", "*", "*", target=None)
        elif isinstance(method, six.string_types):
            method = method_registry.Method.parsed(method)
            method = method.fixed(dim=dim, target=target)
        elif isinstance(method, (list, tuple)):
            if len(method) != 3:
                raise TypeError("Method size %s unsupported." % len(method))
            split, algo, impl = method
            method = method_registry.Method(dim, split, algo, impl, target)
        else:
            raise TypeError("Method type %s unsupported." % type(method))
        return method
