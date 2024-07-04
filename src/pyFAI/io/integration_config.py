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

"""Module function to manage configuration files, all serialisable to JSON.
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/07/2024"
__docformat__ = 'restructuredtext'

import logging
from . import ponifile
from .. import detectors
from .. import load_integrators
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

    if isinstance(filenames, (str,)):
        if "," in filenames:
            # Create a list from a coma separated string list
            filenames = filenames.split(",")
            filenames = [f.strip() for f in filenames]
            config[key] = filenames


def _patch_v1_to_v2(config):
    """Rework the config dictionary from version 1 to version 2

    The main difference with version1:
    * management of the detector
     --> pixel1, pixel2, shape and spline are now part of the detector_config, no more top level
    * do_openCL is now deprecated, replaced with the method (which can still be a string)
    * replace comma-separated list of flat/dark with a python list of files.

    :param dict config: Dictionary reworked inplace.
    """
    detector = None
    value = config.pop("poni", None)
    if value is None and "poni_version" in config:
        # Anachronistic configuration, bug found in #2227
        value = config.copy()
        config.clear()  # Do not change the object: empty in place
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
    else:
        if "shape" in config and "pixel1" in config and "pixel2" in config:
            max_shape = config["shape"]
            pixel1 = config["pixel1"]
            pixel2 = config["pixel2"]
            spline = config.get("splineFile")
            detector = detectors.Detector(pixel1, pixel2, splineFile=spline, max_shape=max_shape)
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

    The main difference is in the management of the "method" which was a string
    and is now parsed into a 3- or 5-tuple containing the splitting scheme, the algorithm and the implementation.
    when 5-tuple, there is a reference to the opencl-target as well.
    The prefered version is to have method and opencl_device separated for ease of parsing.

    :param dict config: Dictionary reworked inplace.
    """
    old_method = config.pop("method")
    if isinstance(old_method, (list, tuple)):
        if len(old_method)==5:
            method = method_registry.Method(*old_method)
        else:
            if config.get("do_2D") and config.get("nbpt_azim", 0)>1:
                ndim = 2
                default = load_integrators.PREFERED_METHODS_2D[0]
            else:
                ndim = 1
                default = load_integrators.PREFERED_METHODS_1D[0]

            long_method = method_registry.IntegrationMethod.select_one_available(old_method,
                                                                                 dim=ndim, default=default)
            method = long_method.method
    else:
        method = method_registry.Method.parsed(old_method)
    config["method"] = method.split, method.algo, method.impl
    config["opencl_device"] = method.target

    config["version"] = 3


def normalize(config, inplace=False, do_raise=False):
    """Normalize the configuration file to the one supported internally\
    (the last one).

    :param dict config: The configuration dictionary to read
    :param bool inplace: In true, the dictionary is edited inplace
    :param bool do_raise: raise ValueError if set. Else use logger.error
    :raise ValueError: If the configuration do not match & do_raise is set
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
        txt = f"Configuration application do not match. Found '{application}'"
        if do_raise:
            raise ValueError(txt)
        else:
            _logger.error(txt)

    if version > 3:
        _logger.error("Configuration file %d too recent. This version of pyFAI maybe too old to read this configuration", version)

    return config


class ConfigurationReader(object):

    def __init__(self, config):
        ":param config: dictonary"
        self._config = config

    def pop_ponifile(self):
        """Returns the geometry subpart of the configuration"""
        dico = {"poni_version":2}
        mapping = { i:i for i in ('wavelength', 'poni1', 'poni2',
                                  'rot1', 'rot2', 'rot3', 'detector', 'detector_config')}
        mapping['dist'] = "distance"
        for key1, key2 in mapping.items():
            if key1 in self._config:
                value = self._config.pop(key1)
                if value is not None:
                    dico[key2] = value
        return ponifile.PoniFile(dico)

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
            lngm = load_integrators.PREFERED_METHODS_2D[0] if dim==2 else load_integrators.PREFERED_METHODS_1D[0]
            method = lngm.method
        elif isinstance(method, (str,)):
            method = method_registry.Method.parsed(method)
            method = method.fixed(dim=dim, target=target)
        elif isinstance(method, (list, tuple)):
            if len(method) == 3:
                split, algo, impl = method
                method = method_registry.Method(dim, split, algo, impl, target)
            elif 3 < len(method) <= 5:
                method = method_registry.Method(*method)
            else:
                raise TypeError(f"Method size {len(method)} is unsupported, method={method}.")
        else:
            raise TypeError(f"Method type {type(method)} unsupported, method={method}.")
        return method
