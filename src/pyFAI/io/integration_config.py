# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""
The WorkerConfig dataclass manages all options for the pyFAI.worker.

List of attributes as function of the version:

0. dist poni1 poni2 rot1 rot3 rot2 pixel1 pixel2 splineFile wavelength
   poni (path of the file) chi_discontinuity_at_0 do_mask do_dark
   do_azimuthal_range do_flat do_2D azimuth_range_min azimuth_range_max
   polarization_factor nbpt_rad do_solid_angle do_radial_range error_model
   delta_dummy nbpt_azim flat_field radial_range_min dark_current do_polarization
   mask_file detector unit radial_range_max val_dummy do_dummy method do_OpenCL

1. dark_current, flat_field can now be lists of files, used to be coma-separated strings

2. detector/detector_config are now defined
   As a consequence, all those keys are now invalid: pixel1 pixel2 splineFile
   and are integrated into those keys: detector detector_config

3. method becomes a 3-tuple like ("full", "csr", "cython")
   opencl_device contains the device id as 2-tuple of integer

4. `poni` is now a serialization of the poni, no more the path to the poni-file.
   The detector is integrated into it
   As a consequence, all those keys are now invalid:
   dist poni1 poni2 rot1 rot3 rot2 pixel1 pixel2 splineFile wavelength detector detector_config

5. Migrate to dataclass
   Support for `extra_options`
   rename some attributes
   * `integrator_name` -> `integrator_method`
   * `polarization_factor` -> `polarization_description`

In a similar way, PixelWiseWorkerConfig and DistortionWorkerConfig are dataclasses
to hold parameters for handling PixelWiseWorker and DistortionWorker, respectively.

All those data-classes are serializable to JSON.
"""

__author__ = "Jérôme Kieffer"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/02/2025"
__docformat__ = 'restructuredtext'

import sys
import os
import json
import logging
import copy
from typing import ClassVar, Union
import numpy
from .ponifile import PoniFile
from ..containers import PolarizationDescription, ErrorModel, dataclass, fields, asdict
from .. import detectors
from .. import method_registry
from ..integrator import load_engines as load_integrators
from ..utils import decorators
from ..units import Unit, to_unit
_logger = logging.getLogger(__name__)
CURRENT_VERSION = 5



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

    :param dict config: Dictionary reworked in-place.
    """
    detector = None
    value = config.pop("poni", None)
    if value is None and "poni_version" in config:
        # Anachronistic configuration, bug found in #2227
        value = config.copy()
        # warn user about unexpected keys that's gonna be destroyed:
        valid = ('wavelength', 'dist', 'poni1', 'poni2', 'rot1' , 'rot2' , 'rot3', 'detector', "shape", "pixel1", "pixel2", "splineFile")
        delta = set(config.keys()).difference(valid)
        if delta:
            _logger.warning("Integration_config v1 contains unexpected keys which will be discared: %s%s", os.linesep,
                            os.linesep.join([f"{key}: {config[key]}" for key in delta]))
        config.clear()  # Do not change the object: empty in place
    if value:
        # Use the poni file while it is not overwritten by a key of the config
        # dictionary
        poni = PoniFile(value)
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
    The preferred version is to have method and opencl_device separated for ease of parsing.

    :param dict config: Dictionary reworked in-place.
    """
    old_method = config.pop("method", "")
    if isinstance(old_method, (list, tuple)):
        if len(old_method) == 5:
            method = method_registry.Method(*old_method)
        else:
            if config.get("do_2D") and config.get("nbpt_azim", 0) > 1:
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


def _patch_v3_to_v4(config):
    """Rework the config dictionary from version 3 to version 4

    The geometric, detector and beam parameters (contained in the .poni file) now they are gathered in a dictionary in the "poni" key
    This will ease the methods that handle only the PONI parameters defined during the calibration step
    that now they can be retrieved just by getting the value of the key "poni" from the config. The rest of the parameters are
    characteristic of the integration protocol.

    :param dict config: Dictionary reworked in-place.
    """
    poni_dict = {}
    poni_parameters = ["dist",
                       "poni1",
                       "poni2",
                       "rot1",
                       "rot2",
                       "rot3",
                       "detector",
                       "detector_config",
                       "wavelength",
                       "poni_version",
                       ]
    for poni_param in poni_parameters:
        if poni_param in config:
            poni_dict[poni_param] = config.pop(poni_param)

    config["poni"] = poni_dict
    config["version"] = 4


def _patch_v4_to_v5(config):
    """Support for integrator_name/integrator_method and extra_options.

    :param dict config: Dictionary reworked in-place.
    """
    config["version"] = 5
    if "integrator_method" not in config:
        config["integrator_method"] = config.pop("integrator_name", None)
    if "extra_options" not in config:
        config["extra_options"] = None
    if "polarization_factor" in config:
        pf = config.pop("polarization_factor")
        if "polarization_offset" in config:
            config["polarization_description"] = PolarizationDescription(pf, config.pop("polarization_offset"))
        elif "__len__" in dir(pf):
            config["polarization_description"] = PolarizationDescription(*pf)
        elif pf is None:
            config["polarization_description"] = pf
        else:
            config["polarization_description"] = PolarizationDescription(float(pf), 0)

    # Invalidation of certain keys:
    for key1, key2 in [("do_mask", ["mask_image", "mask_file"]),
                       ("do_flat", ["flat_field", "flat_field_image"]),
                       ("do_dark", ["dark_current", "dark_current_image"]),
                       ('do_polarization', ["polarization_description"]),
                       ('do_dummy', ["val_dummy", "delta_dummy"]),
                       ("do_radial_range", ["radial_range_min", "radial_range_max"]),
                       ("do_radial_range", ["azimuth_range_min", "azimuth_range_max", "azimuthal_range_min", "azimuthal_range_max"])]:
        if key1 in config and not config[key1]:
            # there are keys to be removed
            for key in key2:
                if key in config:
                    config.pop(key)
    # Here we deal with keys which have been renamed:
    for key1, key2 in[("azimuth_range_min", "azimuthal_range_min"),
                      ("azimuth_range_max", "azimuthal_range_max"),
                      ('do_azimuthal_range', 'do_azimuth_range'),
                      ("flat_field", "flat_field_image"),
                      ("dark_current", "dark_current_image"),
                      ("mask_file", "mask_image"),
                      ("val_dummy", "dummy")]:
        if key2 in config and not key1 in config:
            config[key1] = config.pop(key2)


def normalize(config, inplace=False, do_raise=False, target_version=CURRENT_VERSION):
    """Normalize the configuration file to the one supported internally\
    (the last one).

    :param dict config: The configuration dictionary to read
    :param bool inplace: If True, the dictionary is edited in-place, else one works on a copy.
    :param bool do_raise: raise ValueError if set. Else use logger.error
    :param int target_version: stop updating when version has been reached.
    :raise ValueError: If the configuration do not match & do_raise is set
    """
    if not inplace:
        config = config.copy()
    version = config.get("version", 1)
    if version == 1:
        # NOTE: Previous way to describe an integration process before pyFAI 0.17
        _patch_v1_to_v2(config)
    if config["version"] == target_version: return config
    if config["version"] == 2:
        _patch_v2_to_v3(config)

    application = config.get("application", None)
    if application not in ("pyfai-integrate", "worker", "poni"):
        txt = f"Configuration application do not match. Found '{application}'"
        if do_raise:
            raise ValueError(txt)
        else:
            _logger.error(txt)
    if config["version"] == target_version: return config
    if config["version"] == 3:
        _patch_v3_to_v4(config)

    if config["version"] == target_version: return config
    if config["version"] == 4:
        _patch_v4_to_v5(config)

    if version > CURRENT_VERSION:
        _logger.error("Configuration file %d too recent. This version of pyFAI maybe too old to read this configuration (max=%d)",
                      version, CURRENT_VERSION)

    return config


class ConfigurationReader(object):
    "This class should be deprecated now ..."

    def __init__(self, config):
        ":param config: dictionary"
        decorators.deprecated_warning("Class", "ConfigurationReader", reason=None, replacement=None,
                       since_version="2025.01", only_once=True)
        self._config = config

    def pop_ponifile(self):
        """Returns the geometry subpart of the configuration"""
        if isinstance(self._config.get("poni", None), dict):
            return PoniFile(self._config["poni"])

        dico = {"poni_version":2}
        mapping = { i:i for i in ('wavelength', 'poni1', 'poni2',
                                  'rot1', 'rot2', 'rot3', 'detector', 'detector_config')}
        mapping['dist'] = "distance"
        for key1, key2 in mapping.items():
            if key1 in self._config:
                value = self._config.pop(key1)
                if value is not None:
                    dico[key2] = value
        return PoniFile(dico)

    def pop_detector(self):
        """
        Returns the detector stored in the json configuration.

        :rtype: pyFAI.detectors.Detector
        """
        if isinstance(self._config.get("poni", None), dict):
            poni_dict = self._config["poni"].copy()
            detector_class = poni_dict.pop("detector", None)
            if detector_class is not None:
                detector_config = poni_dict.pop("detector_config", None)
                detector = detectors.detector_factory(detector_class, config=detector_config)
                return detector
        else:
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
            lngm = load_integrators.PREFERED_METHODS_2D[0] if dim == 2 else load_integrators.PREFERED_METHODS_1D[0]
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


@dataclass
class WorkerConfig:
    """Class with the configuration from the worker."""
    application: str = "worker"
    version: int = CURRENT_VERSION
    poni: PoniFile = None
    nbpt_rad: int = None
    nbpt_azim: int = None
    unit: Unit = None
    chi_discontinuity_at_0: bool = False
    polarization_description: PolarizationDescription = None
    normalization_factor: float = 1.0
    val_dummy: float = None
    delta_dummy: float = None
    correct_solid_angle: bool = True
    dark_current: Union[str, list] = None
    flat_field: Union[str, list] = None
    mask_file: str = None
    error_model: ErrorModel = ErrorModel.NO
    method: object = None
    opencl_device: list = None
    azimuth_range: list = None
    radial_range: list = None
    integrator_class: str = "AzimuthalIntegrator"
    integrator_method: str = None
    extra_options: dict = None
    monitor_name: str = None
    shape: list = None
    OPTIONAL: ClassVar[list] = ["radial_range_min", "radial_range_max",
                                "azimuth_range_min", "azimuth_range_max",
                                "integrator_name", "do_poisson"]
    GUESSED: ClassVar[list] = ["do_2D", "do_mask", "do_dark", "do_flat", 'do_polarization',
                                'do_dummy', "do_radial_range", 'do_azimuthal_range', 'do_solid_angle']
    ENFORCED: ClassVar[list] = ["polarization_description", "poni", "error_model", "unit"]

    def __repr__(self):
        return json.dumps(self.as_dict(), indent=4)

    def as_dict(self):
        """Like asdict, but with some more features:
        * Handle ponifile & Unit dedicated classes
        * Handle namedtuple like Polarization
        * Handle Enums like ErrorModel
        """
        dico = {}
        for field in fields(self):
            key = field.name
            value = getattr(self, key)
            if key in self.ENFORCED:
                if "as_dict" in dir(value):  # ponifile
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
        normalize(dico, inplace=True)

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
                    elif isinstance(value, str) and ("parse" in klass.__dict__):  #  There is an issue with Enum!
                        to_init[key] = klass.parse(value)
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

    @classmethod
    def from_file(cls, filename: str):
        """load the content of a JSON file and provide a dataclass instance"""
        with open(filename, "r") as f:
            dico = json.loads(f.read())
        return cls.from_dict(dico, inplace=True)

    @property
    def do_2D(self):
        return False if self.nbpt_azim is None else self.nbpt_azim > 1

    @property
    def do_azimuthal_range(self):
        if self.azimuth_range:
            return bool(numpy.isfinite(self.azimuth_range[0]) and numpy.isfinite(self.azimuth_range[1]))
        else:
            return False

    @property
    def azimuth_range_min(self):
        if self.azimuth_range:
            return self.azimuth_range[0]
        else:
            return -numpy.inf

    @azimuth_range_min.setter
    def azimuth_range_min(self, value):
        if not self.azimuth_range:
            self.azimuth_range = [-numpy.inf, numpy.inf]
        self.azimuth_range[0] = value

    @property
    def azimuth_range_max(self):
        if self.azimuth_range:
            return self.azimuth_range[1]
        else:
            return numpy.inf

    @azimuth_range_max.setter
    def azimuth_range_max(self, value):
        if not self.azimuth_range:
            self.azimuth_range = [-numpy.inf, numpy.inf]
        self.azimuth_range[1] = value

    @property
    def do_radial_range(self):
        if self.radial_range:
            return bool(numpy.isfinite(self.radial_range[0]) and numpy.isfinite(self.radial_range[1]))
        else:
            return False

    @property
    def radial_range_min(self):
        if self.radial_range:
            return self.radial_range[0]
        else:
            return -numpy.inf

    @radial_range_min.setter
    def radial_range_min(self, value):
        if not self.radial_range:
            self.radial_range = [-numpy.inf, numpy.inf]
        self.radial_range[0] = -numpy.inf if value is None else value

    @property
    def radial_range_max(self):
        if self.radial_range:
            return self.radial_range[1]
        else:
            return numpy.inf

    @radial_range_max.setter
    def radial_range_max(self, value):
        if not self.radial_range:
            self.radial_range = [-numpy.inf, numpy.inf]
        self.radial_range[1] = numpy.inf if value is None else value

    @property
    def integrator_name(self):
        return self.integrator_method

    @integrator_name.setter
    def integrator_name(self, value):
        self.integrator_method = value

    @property
    def do_mask(self):
        return self.mask_file is not None

    @property
    def do_poisson(self):
        return int(self.error_model) == int(ErrorModel.POISSON)

    @do_poisson.setter
    def do_poisson(self, value):
        if value:
            self.error_model = ErrorModel.POISSON

    @property
    def dummy(self):
        return self.val_dummy

    @dummy.setter
    def dummy(self, value):
        if value:
            self.val_dummy = float(value) if value is not None else value

    @property
    def do_dummy(self):
        return self.val_dummy is not None

    @property
    def mask_image(self):
        return self.mask_file

    @mask_image.setter
    def mask_image(self, value):
        self.mask_file = None if not value else value

    @property
    def do_dark(self):
        return bool(self.dark_current)

    @property
    def do_flat(self):
        return bool(self.flat_field)

    @property
    def do_polarization(self):
        if self.polarization_description is None:
            return False
        else:
            return True
        if "__len__" in dir(self.polarization_factor):
            return bool(self.polarization_factor)
        else:
            return True

    @property
    def polarization_factor(self):
        if self.polarization_description is None:
            return None
        else:
            return self.polarization_description[0]

    @polarization_factor.setter
    def polarization_factor(self, value):
        if self.polarization_description is None:
            self.polarization_description = PolarizationDescription(value, 0)
        else:
            self.polarization_description = PolarizationDescription(value, self.polarization_description[1])

    @property
    def polarization_offset(self):
        if self.polarization_description is None:
            return None
        else:
            return self.polarization_description[1]

    @polarization_offset.setter
    def polarization_offset(self, value):
        if self.polarization_description is None:
            self.polarization_description = PolarizationDescription(0, value)
        else:
            self.polarization_description = PolarizationDescription(self.polarization_description[0], value)

    @property
    def dark_current_image(self):
        return self.dark_current

    @dark_current_image.setter
    def dark_current_image(self, value):
        self.dark_current = None if not value else value

    @property
    def flat_field_image(self):
        return self.flat_field

    @flat_field_image.setter
    def flat_field_image(self, value):
        self.flat_field = None if not value else value

    @property
    def do_solid_angle(self):
        return self.correct_solid_angle

    @do_solid_angle.setter
    def do_solid_angle(self, value):
        self.correct_solid_angle = None if value is None else bool(value)

    # Dict-like API, for (partial) compatibility
    @decorators.deprecated(reason="WorkerConfig now dataclass, no more a dict", replacement=None, since_version="2025.01")
    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    @decorators.deprecated(reason="WorkerConfig now dataclass, no more a dict", replacement=None, since_version="2025.01")
    def __getitem__(self, key):
        return self.__getattribute__(key)

    @decorators.deprecated(reason="WorkerConfig now dataclass, no more a dict", replacement=None, since_version="2025.01")
    def __contains__(self, key):
        try:
            return self.__getattribute__(key) is not None
        except AttributeError:
            return False

    @decorators.deprecated(reason="WorkerConfig now dataclass, no more a dict", replacement=None, since_version="2025.01")
    def get(self, key, default=None):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            return default

# @dataclass(slots=True)
# class PixelwiseWorkerConfig:
#     """Configuration for pyFAI.worker.PixelwiseWorker
#     """
#     dark_file: Union[str, os.PathLike]=None
#     flat_file: Union[str, os.PathLike]=None
#     solidangle_file: Union[str, os.PathLike]=None
#     polarization: float=None
#
#     dummy: float=None
#     delta_dummy: float=None
#     empty: float=None
#     dtype: str=None
