#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

"""This module contains the Worker class:

A tool able to perform azimuthal integration with:
additional saving capabilities like

- save as 2/3D structure in a HDF5 File
- read from HDF5 files

Aims at being integrated into a plugin like LImA or as model for the GUI

The configuration of this class is mainly done via a WorkerConfig object serialized as a JSON string.
For the valid keys, please refer to the doc of the dataclass `pyFAI.io.integration_config.WorkerConfig`
"""
from __future__ import annotations

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/10/2025"
__status__ = "development"

import threading
import os.path
import logging
import json
import numpy


from . import average
from . import method_registry
from .integrator.azimuthal import AzimuthalIntegrator
from .integrator.fiber import FiberIntegrator
from .containers import ErrorModel
from .method_registry import IntegrationMethod, Method
from .distortion import Distortion
from . import units
from .io import ponifile, image as io_image
from .io.integration_config import WorkerConfig, WorkerFiberConfig
from .engines.preproc import preproc as preproc_numpy
from .utils.mathutil import binning as rebin
logger = logging.getLogger(__name__)
try:
    import numexpr
except ImportError as err:
    logger.warning("Unable to import Cython version of preproc: %s", err)
    USE_NUMEXPR = False
else:
    USE_NUMEXPR = True

try:
    from .ext.preproc import preproc
except ImportError as err:
    logger.warning("Unable to import Cython version of preproc: %s", err)
    preproc = preproc_numpy
    USE_CYTHON = False
else:
    USE_CYTHON = True


def make_ai(config, consume_keys=False):
    """Create an Azimuthal integrator from the configuration.

    :param config: Key-value dictionary with all parameters
    :param bool consume_keys: If true the keys from the dictionary will be
        consumed when used.
    :return: A configured (but uninitialized) :class:`AzimuthalIntgrator`.
    """
    if not isinstance(config, WorkerConfig):
        config = WorkerConfig.from_dict(config, inplace=consume_keys)
    ai = AzimuthalIntegrator()
    _init_ai(ai, config)
    return ai


def _init_ai(ai, config, read_maps=True):
    """Initialize an :class:`AzimuthalIntegrator` from a configuration.

    :param AzimuthalIntegrator ai: An :class:`AzimuthalIntegrator`.
    :param config: WorkerConfig dataclass instance
    :param bool read_maps: If true mask, flat, dark will be read.
    :return: A configured (but uninitialized) :class:`AzimuthalIntgrator`.
    """
    ai._init_from_poni(ponifile.PoniFile(config.poni))

    if config.chi_discontinuity_at_0:
        ai.setChiDiscAtZero()
    else:
        ai.setChiDiscAtPi()

    if read_maps:
        filename = config.mask_file
        if filename:
            try:
                data = io_image.read_image_data(filename)
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s", filename, error)
            else:
                ai.detector.mask = data

        filename = config.dark_current
        if filename:
            ai.detector.set_darkcurrent(_reduce_images(_normalize_filenames(filename)))

        filename = config.flat_field
        if filename:
            ai.detector.set_flatfield(_reduce_images(_normalize_filenames(filename)))
    return ai


def _normalize_filenames(filenames):
    """Returns a list of strings from a string or a list of strings.

    :rtype: List[str]
    """
    if filenames is None or filenames == "":
        return []
    if isinstance(filenames, list):
        return filenames
    if isinstance(filenames, (str,)):
        # It's a single filename
        return [filenames]
    raise TypeError("Unsupported type %s for a list of filenames" % type(filenames))


def _reduce_images(filenames, method="mean"):
    """
    Reduce a set of filenames using a reduction method

    :param List[str] filenames: List of files used to compute the data
    :param str method: method used to compute the dark, "mean" or "median"
    """
    if isinstance(filenames, str):
        return io_image.read_image_data(filenames).astype(numpy.float32)
    if len(filenames) == 0:
        return None
    if len(filenames) == 1:
        return io_image.read_image_data(filenames[0]).astype(numpy.float32)
    else:
        return average.average_images(filenames, filter_=method, fformat=None, threshold=0)


class Worker(object):

    def __init__(self, azimuthalIntegrator=None,
                 shapeIn=None, shapeOut=(360, 500),
                 unit="r_mm", dummy=None, delta_dummy=None,
                 method=("bbox", "csr", "cython"),
                 integrator_name=None, extra_options=None):
        """
        :param AzimuthalIntegrator azimuthalIntegrator: An AzimuthalIntegrator instance
        :param tuple shapeIn: image size in input ->auto guessed from detector shape now
        :param tuple shapeOut: Integrated size: can be (1,2000) for 1D integration
        :param str unit: can be "2th_deg, r_mm or q_nm^-1 ...
        :param float dummy: the value making invalid pixels
        :param float delta_dummy: the precision for dummy values
        :param method: integration method: str like "csr" or tuple ("bbox", "csr", "cython") or IntegrationMethod instance.
        :param str integrator_name: Offers an alternative to "integrate1d" like "sigma_clip_ng"
        :param dict extra_options: extra kwargs for the integrator (like {"max_iter":3, "thres":0, "error_model": "azimuthal"} for sigma-clipping)
        """
        self._sem = threading.Semaphore()
        if azimuthalIntegrator is None:
            self.ai = AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntegrator
        self._normalization_factor = None  # Value of the monitor: divides the intensity by this value for normalization
        self.integrator_name = integrator_name
        self._processor = None
        self._nbpt_azim = None
        if isinstance(method, (str, list, tuple, Method, IntegrationMethod)):
            method = IntegrationMethod.parse(method)
        else:
            logger.error(f"Unable to parse method {method}")
        self.method = (method.split, method.algorithm, method.implementation)
        self.opencl_device = method.target
        self._method = None
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self._unit = units.to_unit(unit)
        self.polarization_factor = None
        self.dummy = dummy
        self.delta_dummy = delta_dummy
        self.correct_solid_angle = True
        self.dark_current_image = None
        self.flat_field_image = None
        self.mask_image = None
        self.subdir = ""
        self.extension = None
        self.needs_reset = True
        self.output = "numpy"  # exports as numpy array by default
        self._shape = shapeIn
        self.radial = None
        self.azimuthal = None
        self.propagate_uncertainties = None
        self.safe = True
        self.extra_options = {} if extra_options is None else extra_options.copy()
        self.radial_range = self.extra_options.pop("radial_range", None)
        self.azimuth_range = self.extra_options.pop("azimuth_range", None)
        self.error_model = ErrorModel.parse(self.extra_options.pop("error_model", None))

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Azimuthal Integrator:", self.ai.__repr__(),
                  f"Input image shape: {self.shape}",
                  f"Number of points in radial direction: {self.nbpt_rad}",
                  f"Number of points in azimuthal direction: {self.nbpt_azim}",
                  f"Unit in radial dimension: {self.unit}",
                  f"Correct for solid angle: {self.correct_solid_angle}",
                  f"Polarization factor: {self.polarization_factor}",
                  f"Dark current image: {self.dark_current_image}",
                  f"Flat field image: {self.flat_field_image}",
                  f"Mask image: {self.mask_image}",
                  f"Dummy: {self.dummy},\tDelta_Dummy: {self.delta_dummy}",
                  f"Directory: {self.subdir}, \tExtension: {self.extension}",
                  f"Radial range: {self.radial_range}",
                  f"Azimuth range: {self.azimuth_range}"]
        return os.linesep.join(lstout)

    def do_2D(self):
        if self.nbpt_azim is None:
            return False
        return self.nbpt_azim > 1

    def update_processor(self, integrator_name=None):
        dim = 2 if self.do_2D() else 1
        if integrator_name is None:
            integrator_name = self.integrator_name
        if integrator_name is None:
            integrator_name = f"integrate{dim}d"
        else:
            if "2d" in integrator_name and dim == 1:
                integrator_name = integrator_name.replace("2d", "1d")
            elif "1d" in integrator_name and dim == 2:
                integrator_name = integrator_name.replace("1d", "2d")
        self._processor = self.ai.__getattribute__(integrator_name)

        if isinstance(self.method, (list, tuple)):
            if isinstance(self.method, Method):
                methods = IntegrationMethod.select_method(dim=dim, split=self.method[1], algo=self.method[2], impl=self.method[3],
                                      target=self.opencl_device if isinstance(self.opencl_device, (tuple, list)) else self.method[4],
                                      target_type=self.opencl_device if isinstance(self.opencl_device, str) else self.method[4],
                                      degradable=True)
            else:
                methods = IntegrationMethod.select_method(dim=dim, split=self.method[0], algo=self.method[1], impl=self.method[2],
                                                      target=self.opencl_device if isinstance(self.opencl_device, (tuple, list)) else None,
                                                      target_type=self.opencl_device if isinstance(self.opencl_device, str) else None,
                                                      degradable=True)
            self._method = methods[0]
        elif isinstance(self.method, str) or self.method is None:
            self._method = IntegrationMethod.select_one_available(method=self.method, dim=dim)
        elif isinstance(self.method, IntegrationMethod):
            self._method = self.method
        else:
            logger.error(f"No method available for {dim}D integration on {self.method} with target {self.opencl_device}")
            self._method = IntegrationMethod.select_one_available(method=self.method, dim=dim)
        self.integrator_name = self._processor.__name__
        self.radial = None
        self.azimuthal = None

    @property
    def nbpt_azim(self):
        return self._nbpt_azim

    @nbpt_azim.setter
    def nbpt_azim(self, value):
        self._nbpt_azim = value
        self.update_processor()

    def reset(self):
        """
        this is just to force the integrator to initialize
        """
        if self.needs_reset:
            with self._sem:
                if self.needs_reset:
                    self.ai.reset()
                    self.needs_reset = False

    def reconfig(self, shape=None, sync=False):
        """
        This is just to force the integrator to initialize with a given input image shape

        :param shape: shape of the input image
        :param sync: return only when synchronized
        """
        if shape is not None:
            self._shape = shape
            if self.ai.detector.force_pixel:
                mask = self.ai.detector.mask
                logger.info(f"Before rebin, mask has {(mask!=0).sum()} pixels")
                res = self.ai.detector.guess_binning(shape)
                if res and mask is not None:
                    new_shape = numpy.array(self.ai.detector.shape)
                    if numpy.all(new_shape < numpy.array(mask.shape)):
                        self.ai.detector.mask = (rebin(mask, self.ai.detector.binning) != 0)
                        logger.info(f"reconfig: mask has been rebinned from {mask.shape} to {self.ai.detector.mask.shape}. Masking {self.ai.detector.mask.sum()} pixels")
            else:
                self.ai.detector.shape = shape
        self.ai.empty = self.dummy
        self.ai.reset()
        self.warmup(sync)

    def process(self, data, variance=None,
                dark=None,
                flat=None,
                normalization_factor=1.0,
                writer=None, metadata=None):
        """
        Process one frame

        :param data: numpy array containing the input image
        :param writer: An open writer in which 'write' will be called with the result of the integration
        """

        with self._sem:
            monitor = self._normalization_factor * normalization_factor if self._normalization_factor else normalization_factor
        kwarg = self.extra_options.copy()
        kwarg["unit"] = self.unit
        kwarg["dummy"] = self.dummy
        kwarg["delta_dummy"] = self.delta_dummy
        kwarg["method"] = self._method
        kwarg["polarization_factor"] = self.polarization_factor
        kwarg["data"] = data
        kwarg["correctSolidAngle"] = self.correct_solid_angle
        kwarg["safe"] = self.safe
        kwarg["variance"] = variance
        kwarg["dark"] = dark
        kwarg["flat"] = flat
        if self.error_model:
            kwarg["error_model"] = self.error_model

        if metadata is not None:
            kwarg["metadata"] = metadata

        if monitor is not None:
            kwarg["normalization_factor"] = monitor

        if self.do_2D():
            kwarg["npt_rad"] = self.nbpt_rad
            kwarg["npt_azim"] = self.nbpt_azim
        else:
            kwarg["npt"] = self.nbpt_rad

        if self.radial_range is not None:
            kwarg["radial_range"] = self.radial_range

        if self.azimuth_range is not None:
            kwarg["azimuth_range"] = self.azimuth_range

        try:
            integrated_result = self._processor(**kwarg)
        except Exception as err:
            logger.info("Backtrace", exc_info=True)
            err2 = [f"error in integration do_2d: {self.do_2D()}",
                    str(err.__class__.__name__),
                    str(err),
                    f"data.shape: {data.shape}",
                    f"data.size: {data.size}",
                    "ai:",
                    str(self.ai),
                    "method:",
                    str(kwarg.get("method"))
                    ]
            logger.error("\n".join(err2))
            raise err
        else:
            if self.radial is None:
                self.radial = integrated_result.radial
                if self.do_2D():
                    self.azimuthal = integrated_result.azimuthal
        if writer is not None:
            writer.write(integrated_result)
        if self.output == "raw":
            return integrated_result
        elif self.output == "numpy":
            if self.do_2D():
                if integrated_result.sigma is None:
                    return integrated_result.intensity
                else:
                    return integrated_result.intensity, integrated_result.sigma
            else:
                return numpy.vstack(integrated_result).T

    def setSubdir(self, path):
        """
        Set the relative or absolute path for processed data
        """
        self.subdir = path

    def setExtension(self, ext):
        """
        enforce the extension of the processed data file written
        """
        if ext:
            self.extension = ext
        else:
            self.extension = None

    def set_dark_current_file(self, imagefile):
        self.ai.detector.set_darkcurrent(_reduce_images(imagefile))
        self.dark_current_image = imagefile

    setDarkcurrentFile = set_dark_current_file

    def set_flat_field_file(self, imagefile):
        self.ai.detector.set_flatfield(_reduce_images(imagefile))
        self.flat_field_image = imagefile

    setFlatfieldFile = set_flat_field_file

    def set_mask_file(self, imagefile):
        self.ai.set_mask(mask=_reduce_images(imagefile))
        self.mask_image = imagefile

    setMaskFile = set_mask_file

    def set_config(self, config:dict | WorkerConfig, consume_keys:bool=False):
        """
        Configure the working from the dictionary|WorkerConfig.

        :param dict config: Key-value configuration or WorkerConfig dataclass instance
        :param bool consume_keys: If true the keys from the dictionary will be
            consumed when used.
        """
        if not isinstance(config, WorkerConfig):
            config = WorkerConfig.from_dict(config, inplace=consume_keys)
        _init_ai(self.ai, config, read_maps=False)

        # Do it here before reading the AI to be able to catch the io
        filename = config.mask_file
        if filename:
            try:
                data = io_image.read_image_data(filename)
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s", filename, error)
            else:
                self.ai.detector.mask = data
                self.mask_image = filename

        # Do it here while we have to store metadata
        filename = config.dark_current
        if filename:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_darkcurrent(data)
            self.dark_current_image = filenames

        # Do it here while we have to store metadata
        filename = config.flat_field
        if filename:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_flatfield(data)
            self.flat_field_image = filenames

        self._nbpt_azim = int(config.nbpt_azim) if config.nbpt_azim else 1
        self.method = config.method  # expand to Method ?
        self.opencl_device = config.opencl_device
        self.nbpt_rad = config.nbpt_rad
        self.unit = units.to_unit(config.unit or "2th_deg")
        self.error_model = ErrorModel.parse(config.error_model)
        self.polarization_factor = config.polarization_factor
        self.azimuth_range = config.azimuth_range
        self.radial_range = config.radial_range
        self.correct_solid_angle = True if config.correct_solid_angle is None else bool(config.correct_solid_angle)
        self.dummy = config.val_dummy
        self.delta_dummy = config.delta_dummy
        self._normalization_factor = config.normalization_factor
        self.extra_options = config.extra_options or {}
        self.update_processor(integrator_name=config.integrator_method)
        if config.monitor_name:
            logger.warning("Monitor name defined but unsupported by the worker.")

        logger.info(self.ai.__repr__())
        self.reset()
        # For now we do not calculate the LUT as the size of the input image is unknown

    def set_unit(self, value):
        self._unit = units.to_unit(value)

    def get_unit(self):
        return self._unit

    unit = property(get_unit, set_unit)

    def get_worker_config(self):
        """Returns the configuration as a WorkerConfig dataclass instance.

        :return: WorkerConfig dataclass instance
        """
        config = WorkerConfig(application="worker",
                              poni=dict(self.ai.get_config()),
                              unit=str(self._unit))
        for key in ["nbpt_azim", "nbpt_rad", "polarization_factor",  "extra_options",
                    "correct_solid_angle", "error_model", "method", "opencl_device",
                    "azimuth_range", "radial_range",
                    "dummy", "delta_dummy", "normalization_factor",
                    "dark_current_image", "flat_field_image",
                    "mask_image", "integrator_name", "shape"]:
            try:
                config.__setattr__(key, self.__getattribute__(key))
            except Exception as err:
                logger.error(f"exception {type(err)} at {key} ({err})")
        return config

    def get_config(self):
        """Returns the configuration as a JSON-serializable dictionary.
        :return: JSON-serializable dictionary
        """
        return self.get_worker_config().as_dict()

    def get_json_config(self):
        """return configuration as a JSON string"""
        return json.dumps(self.get_config(),
                          indent=2)

    def set_json_config(self, json_file):
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                config = json.load(f)
        else:
            config = json.loads(json_file)
        self.set_config(config)

    setJsonConfig = set_json_config

    def save_config(self, filename=None):
        """Save the configuration as a JSON file"""
        self.get_worker_config().save(filename or self.config_file)

    def sync_init(self):

        self.dummy_output = self.process(numpy.zeros(self.shape, dtype=numpy.float32))

    def _warmup(self):
        backup = self.output
        self.output = "raw"
        msk = "" if self.ai.detector.mask is None else f"mask {self.ai.detector.mask.shape} mask sum {self.ai.detector.mask.sum()}\n"
        logger.info(f"warm-up with shape {self.shape}, detector shape {self.ai.detector.shape}\n{msk}{self.ai}")
        integrated_result = self.process(numpy.zeros(self.shape, dtype=numpy.float32))
        self.radial = integrated_result.radial
        if self.do_2D():
            self.azimuthal = integrated_result.azimuthal
        else:
            self.azimuthal = None
        self.propagate_uncertainties =  (integrated_result.sigma is not None)
        self.output = backup

    def warmup(self, sync=False):
        """
        Process a dummy image to ensure everything is initialized

        :param sync: wait for processing to be finished

        """
        t = threading.Thread(target=self._warmup, name="_warmup")
        t.start()
        if sync:
            t.join()

    def get_normalization_factor(self):
        with self._sem:
            return self._normalization_factor

    def set_normalization_factor(self, value):
        with self._sem:
            self._normalization_factor = value

    normalization_factor = property(get_normalization_factor, set_normalization_factor)

    def set_method(self, method="csr"):
        "Set the integration method"
        dim = 2 if self.do_2D() else 1
        if method is None:
            method = method_registry.Method(dim, "*", "*", "*", target=None)
        elif isinstance(method, method_registry.Method):
            method = method.fixed(dim=dim)
        elif isinstance(method, (str,)):
            method = method_registry.Method.parsed(method)
            method = method.fixed(dim=dim)
        elif isinstance(method, (list, tuple)):
            if len(method) != 3:
                raise TypeError("Method size %s unsupported." % len(method))
            split, algo, impl = method
            method = method_registry.Method(dim, split, algo, impl, target=None)
        else:
            raise TypeError("Method type %s unsupported." % type(method))
        return method

    __call__ = process

    @staticmethod
    def validate_config(config, raise_exception=RuntimeError):
        """
        Validates a configuration for any inconsistencies

        :param config: dict containing the configuration
        :param raise_exception: Exception class to raise when configuration is not consistent
        :return: None or reason as a string when raise_exception is None, else raise the given exception
        """
        reason = None

        config = config.copy()
        if "poni" in config and config.get("version", 0) > 3:
            config.update(config.pop("poni"))
        if not config.get("dist"):
            reason = "Detector distance is undefined"
        elif config.get("poni1") is None:
            reason = "Distance `poni1` is undefined"
        elif config.get("poni2") is None:
            reason = "Distance `poni2` is undefined"
        elif config.get("rot1") is None:
            reason = "Rotation `rot1` is undefined"
        elif config.get("rot2") is None:
            reason = "Rotation `rot2` is undefined"
        elif config.get("rot3") is None:
            reason = "Rotation `rot3` is undefined"
        elif not config.get("nbpt_rad"):
            reason = "Number of radial bins is is undefined"
        elif config.get("do_2D") and not config.get("nbpt_rad"):
            reason = "Number of azimuthal bins is is undefined while 2D integration requested"
        elif config.get("wavelength") is None:
            unit = config.get("unit", "_").split("_")[0]
            if "q" in unit:
                reason = "Wavelength undefined but integration in q-space"
            elif "d" in unit:
                reason = "Wavelength undefined but integration in d*-space"
        if reason and isinstance(raise_exception, Exception):
            raise_exception(reason)
        return reason

    @property
    def shape(self):
        try:
            shape = self.ai.detector.shape
        except Exception:
            logger.warning("The detector does not define its shape !")
            return self._shape
        else:
            return shape


class WorkerFiber(Worker):
    def __init__(self, fiberIntegrator=None,
                 shapeIn=None,
                 npt_oop:int=1000,
                 npt_ip:int=1000,
                 unit_oop="qoop_nm^-1",
                 unit_ip="qip_nm^-1",
                 ip_range:tuple=None,
                 oop_range:tuple=None,
                 incident_angle:float=0.0,
                 tilt_angle:float=0.0,
                 sample_orientation:int=1,
                 dummy=None, delta_dummy=None,
                 method=("no", "csr", "cython"),
                 use_missing_wedge:bool=False,
                 integrator_name="integrate2d_grazing_incidence",
                 extra_options=None,
                 ):
        """
        :param FiberIntegrator FiberIntegrator: A FiberIntegrator instance
        :param tuple shapeIn: image size in input ->auto guessed from detector shape now
        :param tuple shapeOut: Integrated size: can be (1,2000) for 1D integration
        :param str unit_oop: fiber unit to be used along the out-of-plane direction
        :param str unit_ip: fiber unit to be used along the in-plane direction
        :param float dummy: the value making invalid pixels
        :param float delta_dummy: the precision for dummy values
        :param method: integration method: str like "csr" or tuple ("bbox", "csr", "cython") or IntegrationMethod instance.
        :param str integrator_name: Offers an alternative to "integrate1d" like "sigma_clip_ng"
        """
        self._sem = threading.Semaphore()
        if fiberIntegrator is None:
            self.fi = FiberIntegrator()
        else:
            self.fi = fiberIntegrator
        self.ai = self.fi
        self._normalization_factor = None  # Value of the monitor: divides the intensity by this value for normalization

        self.integrator_name = integrator_name
        self._processor = None
        if isinstance(method, (str, list, tuple, Method, IntegrationMethod)):
            method = IntegrationMethod.parse(method)
        else:
            logger.error(f"Unable to parse method {method}")
        self.method = (method.split, method.algorithm, method.implementation)
        self.opencl_device = method.target
        self._method = None
        self.use_missing_wedge = use_missing_wedge

        self.unit_ip = units.parse_fiber_unit(unit_ip,
                                              incident_angle=incident_angle,
                                              tilt_angle=tilt_angle,
                                              sample_orientation=sample_orientation,
                                              )
        config_unit = self.unit_ip.get_config_shared()
        self.unit_oop = units.parse_fiber_unit(unit_oop,
                                               **config_unit,)
        self.ip_range = ip_range
        self.oop_range = oop_range
        self._npt_oop = npt_oop
        self._npt_ip = npt_ip
        self.update_processor()

        self._incident_angle = config_unit["incident_angle"]
        self._tilt_angle = config_unit["tilt_angle"]
        self._sample_orientation = config_unit["sample_orientation"]

        self.polarization_factor = None
        self.dummy = dummy
        self.delta_dummy = delta_dummy
        self.correct_solid_angle = True
        self.dark_current_image = None
        self.flat_field_image = None
        self.mask_image = None
        self.subdir = ""
        self.extension = None
        self.needs_reset = True
        self.output = "numpy"  # exports as numpy array by default
        self._shape = shapeIn
        self.propagate_uncertainties = None
        self.safe = True
        self.extra_options = {} if extra_options is None else extra_options.copy()
        self.error_model = ErrorModel.parse(self.extra_options.pop("error_model", None))

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Fiber Integrator:", self.fi.__repr__(),
                  f"Input image shape: {self._shape}",
                  f"Number of points in out-of-plane direction: {self._npt_oop}",
                  f"Number of points in in-plane direction: {self._npt_ip}",
                  f"Unit in out-of-plane direction: {self.unit_oop}",
                  f"Unit in in-plane direction: {self.unit_ip}",
                  f"Out-of-plane range: {self.oop_range}",
                  f"In-plane range: {self.ip_range}",
                  f"Correct for solid angle: {self.correct_solid_angle}",
                  f"Polarization factor: {self.polarization_factor}",
                  f"Dark current image: {self.dark_current_image}",
                  f"Flat field image: {self.flat_field_image}",
                  f"Mask image: {self.mask_image}",
                  f"Dummy: {self.dummy},\tDelta_Dummy: {self.delta_dummy}",
                  f"Directory: {self.subdir}, \tExtension: {self.extension}",
        ]
        return os.linesep.join(lstout)

    def update_processor(self, integrator_name=None):
        integrator_name = integrator_name or self.integrator_name
        return super().update_processor(integrator_name=integrator_name)

    @property
    def npt_oop(self):
        return self._npt_oop

    @npt_oop.setter
    def npt_oop(self, value):
        self._npt_oop = value
        self.update_processor()

    @property
    def npt_ip(self):
        return self._npt_ip

    @npt_ip.setter
    def npt_ip(self, value):
        self._npt_ip = value
        self.update_processor()

    @property
    def incident_angle(self):
        return self._incident_angle

    @incident_angle.setter
    def incident_angle(self, incident_angle):
        self._incident_angle = incident_angle

    @property
    def tilt_angle(self):
        return self._tilt_angle

    @tilt_angle.setter
    def tilt_angle(self, tilt_angle):
        self._tilt_angle = tilt_angle

    @property
    def sample_orientation(self):
        return self._sample_orientation

    @sample_orientation.setter
    def sample_orientation(self, sample_orientation):
        self._sample_orientation = sample_orientation

    def do_2D(self):
        if self.npt_ip == 1 or self.npt_oop == 1:
            return False
        return True

    def set_config(self, config:dict | WorkerFiberConfig, consume_keys:bool=False):
        """
        Configure the working from the dictionary|WorkerFiberConfig.

        :param dict config: Key-value configuration or WorkerFiberConfig dataclass instance
        :param bool consume_keys: If true the keys from the dictionary will be
            consumed when used.
        """
        if not isinstance(config, WorkerFiberConfig):
            config = WorkerFiberConfig.from_dict(config, inplace=consume_keys)
        _init_ai(self.ai, config, read_maps=False)

        # Do it here before reading the AI to be able to catch the io
        filename = config.mask_file
        if filename:
            try:
                data = io_image.read_image_data(filename)
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s", filename, error)
            else:
                self.ai.detector.mask = data
                self.mask_image = filename

        # Do it here while we have to store metadata
        filename = config.dark_current
        if filename:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_darkcurrent(data)
            self.dark_current_image = filenames

        # Do it here while we have to store metadata
        filename = config.flat_field
        if filename:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_flatfield(data)
            self.flat_field_image = filenames

        self.npt_azim = int(config.nbpt_azim) if config.nbpt_azim else 1
        self.npt_rad = config.nbpt_rad
        self.unit_ip = units.parse_fiber_unit(**config.unit_ip)
        self.unit_oop = units.parse_fiber_unit(**config.unit_oop)
        config_unit = self.unit_ip.get_config_shared()
        self._incident_angle = config_unit["incident_angle"]
        self._tilt_angle = config_unit["tilt_angle"]
        self._sample_orientation = config_unit["sample_orientation"]
        self.fi.reset_integrator(
            incident_angle=self._incident_angle,
            tilt_angle=self._tilt_angle,
            sample_orientation=self._sample_orientation,
        )

        self.oop_range = config.oop_range
        self.ip_range = config.ip_range

        self.method = config.method  # expand to Method ?
        self.opencl_device = config.opencl_device
        self.error_model = ErrorModel.parse(config.error_model)
        self.polarization_factor = config.polarization_factor

        self.correct_solid_angle = True if config.correct_solid_angle is None else bool(config.correct_solid_angle)
        self.dummy = config.val_dummy
        self.delta_dummy = config.delta_dummy
        self._normalization_factor = config.normalization_factor
        self.extra_options = config.extra_options or {}
        self.update_processor(integrator_name=config.integrator_method)
        if config.monitor_name:
            logger.warning("Monitor name defined but unsupported by the worker.")

        self.reset()

    def get_worker_config(self):
        """Returns the configuration as a WorkerFiberConfig dataclass instance.

        :return: WorkerConfig dataclass instance
        """
        config = WorkerFiberConfig(application="worker",
                                   poni=dict(self.fi.get_config()),
                                   unit_ip=self.unit_ip,
                                   unit_oop=self.unit_oop,
        )

        for key in ["npt_oop", "npt_ip", "polarization_factor",  "extra_options",
                    "correct_solid_angle", "error_model", "method", "opencl_device",
                    "ip_range", "oop_range",
                    "dummy", "delta_dummy", "normalization_factor",
                    "dark_current_image", "flat_field_image",
                    "mask_image", "integrator_name", "shape"]:
            try:
                config.__setattr__(key, self.__getattribute__(key))
            except Exception as err:
                logger.error(f"exception {type(err)} at {key} ({err})")

        # FiberUnit serialization
        for key in ["unit_ip", "unit_oop"]:
            try:
                config.__setattr__(key,  self.__getattribute__(key).get_config())
            except Exception as err:
                logger.error(f"exception {type(err)} at {key} ({err})")
        return config

    def process(self, data,
                variance=None,
                dark=None,
                flat=None,
                normalization_factor=1.0,
                incident_angle=None,
                tilt_angle=None,
                sample_orientation=None,
                writer=None, metadata=None):
        """
        Process one frame

        :param data: numpy array containing the input image
        :param writer: An open writer in which 'write' will be called with the result of the integration
        """

        with self._sem:
            monitor = self._normalization_factor * normalization_factor if self._normalization_factor else normalization_factor
        kwarg = self.extra_options.copy()
        kwarg["dummy"] = self.dummy
        kwarg["delta_dummy"] = self.delta_dummy
        kwarg["method"] = self._method
        kwarg["polarization_factor"] = self.polarization_factor
        kwarg["data"] = data
        kwarg["correctSolidAngle"] = self.correct_solid_angle
        kwarg["safe"] = self.safe
        kwarg["variance"] = variance
        kwarg["dark"] = dark
        kwarg["flat"] = flat
        if self.error_model:
            kwarg["error_model"] = self.error_model

        if metadata is not None:
            kwarg["metadata"] = metadata

        if monitor is not None:
            kwarg["normalization_factor"] = monitor

        for key in ("npt_ip", "npt_oop",
                    "unit_ip", "unit_oop",
                    "ip_range", "oop_range",
                    "incident_angle", "tilt_angle", "sample_orientation",
                    "use_missing_wedge",
        ):
            kwarg[key] = self.__getattribute__(key)

        if incident_angle is not None:
            kwarg["incident_angle"] = incident_angle
        if tilt_angle is not None:
            kwarg["tilt_angle"] = tilt_angle
        if sample_orientation is not None:
            kwarg["sample_orientation"] = sample_orientation

        try:
            integrated_result = self._processor(**kwarg)
        except Exception as err:
            logger.info("Backtrace", exc_info=True)
            err2 = [f"error in integration do_2d: {self.do_2D()}",
                    str(err.__class__.__name__),
                    str(err),
                    f"data.shape: {data.shape}",
                    f"data.size: {data.size}",
                    "ai:",
                    str(self.ai),
                    "method:",
                    str(kwarg.get("method"))
                    ]
            logger.error("\n".join(err2))
            raise err

        if writer is not None:
            writer.write(integrated_result)
        if self.output == "raw":
            return integrated_result
        elif self.output == "numpy":
            if self.do_2D():
                if integrated_result.sigma is None:
                    return integrated_result.intensity
                else:
                    return integrated_result.intensity, integrated_result.sigma
            else:
                return numpy.vstack(integrated_result).T

class PixelwiseWorker(object):
    """
    Simple worker doing dark, flat, solid angle and polarization correction
    """

    def __init__(self, dark=None, flat=None, solidangle=None, polarization=None,
                 mask=None, dummy=None, delta_dummy=None, device=None,
                 empty=None, dtype="float32"):
        """Constructor of the worker

        :param dark: array
        :param flat: array
        :param solidangle: solid-angle array
        :param polarization: numpy array with 2D polarization corrections
        :param device: Used to influence OpenCL behavior: can be "cpu", "GPU", "Acc" or even an OpenCL context
        :param empty: value given for empty pixels by default
        :param dtype: unit (and precision) in which to perform calculation: float32 or float64
        """
        self.ctx = None
        if dark is not None:
            self.dark = numpy.ascontiguousarray(dark, dtype=numpy.float32)
        else:
            self.dark = None
        if flat is not None:
            self.flat = numpy.ascontiguousarray(flat, dtype=numpy.float32)
        else:
            self.flat = None
        if solidangle is not None:
            self.solidangle = numpy.ascontiguousarray(solidangle, dtype=numpy.float32)
        else:
            self.solidangle = None
        if polarization is not None:
            self.polarization = numpy.ascontiguousarray(polarization, dtype=numpy.float32)
        else:
            self.polarization = None

        if mask is None:
            self.mask = False
        elif mask.min() < 0 and mask.max() == 0:  # 0 is valid, <0 is invalid
            self.mask = (mask < 0).astype(numpy.int8)
        else:
            self.mask = mask.astype(numpy.int8)

        self.dummy = dummy
        self.delta_dummy = delta_dummy
        self.empty = float(empty) if empty else 0.0
        self.dtype = numpy.dtype(dtype).type

    def process(self, data, variance=None, normalization_factor=None,
                use_cython=USE_CYTHON):
        """
        Process the data and apply a normalization factor
        :param data: input data
        :param variance: the variance associated to the data
        :param normalization: normalization factor
        :return: processed data, optionally with the associated error if variance is provided
        """
        propagate_error = (variance is not None)
        if use_cython:
            method = preproc
        else:
            method = preproc_numpy
        temp_data = method(data,
                           variance=variance,
                           dark=self.dark,
                           flat=self.flat,
                           solidangle=self.solidangle,
                           polarization=self.polarization,
                           absorption=None,
                           mask=self.mask,
                           dummy=self.dummy,
                           delta_dummy=self.delta_dummy,
                           normalization_factor=normalization_factor,
                           empty=self.empty,
                           error_model=ErrorModel.NO,
                           dtype=self.dtype)
        if propagate_error:
            proc_data = temp_data[..., 0]
            proc_variance = temp_data[..., 1]
            proc_norm = temp_data[..., 2]
            proc_data /= proc_norm
            proc_error = numpy.sqrt(proc_variance) / proc_norm
            return proc_data, proc_error
        else:
            proc_data = temp_data
            return proc_data

    __call__ = process


class DistortionWorker(object):
    """
    Simple worker doing dark, flat, solid angle and polarization correction
    """

    def __init__(self, detector=None, dark=None, flat=None, solidangle=None, polarization=None,
                 mask=None, dummy=None, delta_dummy=None, method="LUT", device=None):
        """Constructor of the worker
        :param dark: array
        :param flat: array
        :param solidangle: solid-angle array
        :param polarization: numpy array with 2D polarization corrections
        :param dummy: value for bad pixels
        :param delta_dummy: precision for dummies
        :param method: LUT or CSR for the correction
        :param device: Used to influence OpenCL behavior: can be "cpu", "GPU", "Acc" or even an OpenCL context
        """

        self.ctx = None
        if dark is not None:
            self.dark = numpy.ascontiguousarray(dark, dtype=numpy.float32)
        else:
            self.dark = None
        if flat is not None:
            self.flat = numpy.ascontiguousarray(flat, dtype=numpy.float32)
        else:
            self.flat = None
        if solidangle is not None:
            self.solidangle = numpy.ascontiguousarray(solidangle, dtype=numpy.float32)
        else:
            self.solidangle = None
        if polarization is not None:
            self.polarization = numpy.ascontiguousarray(polarization, dtype=numpy.float32)
        else:
            self.polarization = None

        if mask is None:
            self.mask = False
            mask = numpy.zeros(detector.shape, dtype=bool)
        elif mask.min() < 0 and mask.max() == 0:  # 0 is valid, <0 is invalid
            mask = self.mask = (mask < 0)
        else:
            mask = self.mask = mask.astype(bool)

        self.dummy = dummy
        self.delta_dummy = delta_dummy

        if detector is not None:
            self.distortion = Distortion(detector, method=method, device=device,
                                     mask=mask, empty=self.dummy or 0)
            self.distortion.reset(prepare=True)  # enforce initialization
        else:
            self.distortion = None

    def process(self,
                data,
                variance=None,
                normalization_factor=1.0):
        """
        Process the data and apply a normalization factor
        :param data: input data
        :param variance: the variance associated to the data
        :param normalization: normalization factor
        :return: processed data as either an array (data) or two (data, error)
        """
        if self.distortion is not None:
            return self.distortion.correct_ng(data,
                                              variance=variance,
                                              dark=self.dark,
                                              flat=self.flat,
                                              solidangle=self.solidangle,
                                              polarization=self.polarization,
                                              dummy=self.dummy,
                                              delta_dummy=self.delta_dummy,
                                              normalization_factor=normalization_factor)
        else:
            proc_data = preproc(data,
                                variance=variance,
                                dark=self.dark,
                                flat=self.flat,
                                solidangle=self.solidangle,
                                polarization=self.polarization,
                                absorption=None,
                                mask=self.mask,
                                dummy=self.dummy,
                                delta_dummy=self.delta_dummy,
                                normalization_factor=normalization_factor,
                                empty=None)
            if variance is not None:
                pp_signal = proc_data[..., 0]
                pp_variance = proc_data[..., 1]
                pp_normalisation = proc_data[..., 2]
                if numexpr is None:
                    # Cheap, multi-threaded way:
                    res_signal = numexpr.evaluate("where(pp_normalisation==0.0, 0.0, pp_signal / pp_normalisation)")
                    res_error = numexpr.evaluate("where(pp_normalisation==0.0, 0.0, sqrt(pp_variance) / abs(pp_normalisation))")
                else:
                    # Take the numpy road:
                    res_signal = numpy.zeros_like(pp_signal)
                    res_error = numpy.zeros_like(pp_signal)
                    msk = numpy.where(pp_normalisation != 0)
                    res_signal[msk] = pp_signal[msk] / pp_normalisation[msk]
                    res_error[msk] = numpy.sqrt(pp_variance[msk]) / abs(pp_normalisation[msk])
                return res_signal, res_error
            else:
                return proc_data

    __call__ = process
