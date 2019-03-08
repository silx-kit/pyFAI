#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
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

The configuration of this class is mainly done via a dictionary transmitted as a JSON string:
Here are the valid keys:

- "dist"
- "poni1"
- "poni2"
- "rot1"
- "rot3"
- "rot2"
- "pixel1"
- "pixel2"
- "splineFile"
- "wavelength"
- "poni" #path of the file
- "chi_discontinuity_at_0"
- "do_mask"
- "do_dark"
- "do_azimuthal_range"
- "do_flat"
- "do_2D"
- "azimuth_range_min"
- "azimuth_range_max"
- "polarization_factor"
- "nbpt_rad"
- "do_solid_angle"
- "do_radial_range"
- "do_poisson"
- "delta_dummy"
- "nbpt_azim"
- "flat_field"
- "radial_range_min"
- "dark_current"
- "do_polarization"
- "mask_file"
- "detector"
- "unit"
- "radial_range_max"
- "val_dummy"
- "do_dummy"
- "method"
"""


from __future__ import with_statement, print_function, division

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "01/03/2019"
__status__ = "development"

import threading
import os.path
import logging
import json
import numpy

logger = logging.getLogger(__name__)

from .third_party import six
from . import average
from .azimuthalIntegrator import AzimuthalIntegrator
from .distortion import Distortion
from . import units
from .io import integration_config
import pyFAI.io.image
from .engines.preproc import preproc as preproc_numpy
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
    config = integration_config.normalize(config, inplace=consume_keys)
    ai = AzimuthalIntegrator()
    _init_ai(ai, config, consume_keys)
    return ai


def _init_ai(ai, config, consume_keys=False, read_maps=True):
    """Initialize an :class:`AzimuthalIntegrator` from a configuration.

    :param AzimuthalIntegrator ai: An :class:`AzimuthalIntegrator`.
    :param config: Key-value dictionary with all parameters
    :param bool consume_keys: If true the keys from the dictionary will be
        consumed when used.
    :param bool read_maps: If true mask, flat, dark will be read.
    :return: A configured (but uninitialized) :class:`AzimuthalIntgrator`.
    """
    if not consume_keys:
        config = dict(config)

    # Geometry
    for key in ("dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
        value = config.pop(key, None)
        if value is not None:
            ai.__setattr__(key, value)
    wavelength = config.pop("wavelength", None)
    if wavelength:
        if wavelength <= 0 or wavelength > 1e-6:
            logger.warning("Wavelength is in meter... unlikely value %s", wavelength)
        ai.wavelength = wavelength

    # Detector
    reader = integration_config.ConfigurationReader(config)
    detector = reader.pop_detector()
    if detector is not None:
        ai.detector = detector

    value = config.pop("chi_discontinuity_at_0", False)
    if value:
        ai.setChiDiscAtZero()
    else:
        ai.setChiDiscAtPi()

    if read_maps:
        filename = config.pop("mask_file", "")
        apply_process = config.pop("do_mask", True)
        if filename and apply_process:
            try:
                data = pyFAI.io.image.read_image_data(filename)
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s", filename, error)
            else:
                ai.mask = data

        filename = config.pop("dark_current", "")
        apply_process = config.pop("do_dark", True)
        if filename and apply_process:
            filenames = _normalize_filenames(filename)
            ai.set_darkfiles(filenames)

        filename = config.pop("flat_field", "")
        apply_process = config.pop("do_flat", True)
        if filename and apply_process:
            filenames = _normalize_filenames(filename)
            ai.set_flatfiles(filenames)

    return ai


def _normalize_filenames(filenames):
    """Returns a list of strings from a string or a list of strings.

    :rtype: List[str]
    """
    if filenames is None or filenames == "":
        return []
    if isinstance(filenames, list):
        return filenames
    if isinstance(filenames, six.string_types):
        # It's a single filename
        return [filenames]
    raise TypeError("Unsupported type %s for a list of filenames" % type(filenames))


def _reduce_images(filenames, method="mean"):
    """
    Reduce a set of filenames using a reduction method

    :param List[str] filenames: List of files used to compute the data
    :param str method: method used to compute the dark, "mean" or "median"
    """
    if len(filenames) == 0:
        return None
    if len(filenames) == 1:
        return pyFAI.io.image.read_image_data(filenames[0]).astype(numpy.float32)
    else:
        return average.average_images(filenames, filter_=method, fformat=None, threshold=0)


class Worker(object):
    def __init__(self, azimuthalIntegrator=None,
                 shapeIn=(2048, 2048), shapeOut=(360, 500),
                 unit="r_mm", dummy=None, delta_dummy=None,
                 azimuthalIntgrator=None):
        """
        :param AzimuthalIntegrator azimuthalIntegrator: An AzimuthalIntegrator instance
        :param AzimuthalIntegrator azimuthalIntgrator: An AzimuthalIntegrator instance (deprecated)
        :param shapeIn: image size in input
        :param shapeOut: Integrated size: can be (1,2000) for 1D integration
        :param unit: can be "2th_deg, r_mm or q_nm^-1 ...
        """
        # TODO remove it in few month (added on 2016-08-04)
        if azimuthalIntgrator is not None:
            logger.warning("'Worker(azimuthalIntgrator=...)' parameter is deprecated cause it contains a typo. Please use 'azimuthalIntegrator='")
            azimuthalIntegrator = azimuthalIntgrator

        self._sem = threading.Semaphore()
        if azimuthalIntegrator is None:
            self.ai = AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntegrator
#        self.config = {}
#        self.config_file = "azimInt.json"
#        self.nbpt_azim = 0
#        if type(config) == dict:
#            self.config = config
#        elif type(config) in types.StringTypes:
#            if os.path.isfile(config):
#                with open(config, "r") as f:
#                    self.config = json.load(f)
#                self.config_file(config)
#            else:
#                self.config = json.loads(config)
#        if self.config:
#            self.configure()
        self._normalization_factor = None  # Value of the monitor: divides the intensity by this value for normalization
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
        self.do_poisson = None
        self.needs_reset = True
        self.output = "numpy"  # exports as numpy array by default
        self.shape = shapeIn
        self.method = "csr"
        self.radial = None
        self.azimuthal = None
        self.radial_range = None
        self.azimuth_range = None
        self.safe = True

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Azimuthal Integrator:", self.ai.__repr__(),
                  "Input image shape: %s" % list(self.shape),
                  "Number of points in radial direction: %s" % self.nbpt_rad,
                  "Number of points in azimuthal direction: %s" % self.nbpt_azim,
                  "Unit in radial dimension: %s" % self.unit,
                  "Correct for solid angle: %s" % self.correct_solid_angle,
                  "Polarization factor: %s" % self.polarization_factor,
                  "Dark current image: %s" % self.dark_current_image,
                  "Flat field image: %s" % self.flat_field_image,
                  "Mask image: %s" % self.mask_image,
                  "Dummy: %s,\tDelta_Dummy: %s" % (self.dummy, self.delta_dummy),
                  "Directory: %s, \tExtension: %s" % (self.subdir, self.extension),
                  "Radial range: %s" % self.radial_range,
                  "Azimuth range: %s" % self.azimuth_range]
        return os.linesep.join(lstout)

    def do_2D(self):
        return self.nbpt_azim > 1

    def reset(self):
        """
        this is just to force the integrator to initialize
        """
        if self.needs_reset:
            with self._sem:
                if self.needs_reset:
                    self.ai.reset()
                    self.needs_reset = False

    def reconfig(self, shape=(2048, 2048), sync=False):
        """
        This is just to force the integrator to initialize with a given input image shape

        :param shape: shape of the input image
        :param sync: return only when synchronized
        """
        self.shape = shape
        self.ai.reset()
        self.warmup(sync)

    def process(self, data, variance=None, normalization_factor=1.0, writer=None, metadata=None):
        """
        Process a frame
        #TODO:
        dark, flat, sa are missing

        :param data: numpy array containing the input image
        :param writer: An open writer in which 'write' will be called with the result of the integration
        """

        with self._sem:
            monitor = self._normalization_factor * normalization_factor if self._normalization_factor else normalization_factor
        kwarg = {"unit": self.unit,
                 "dummy": self.dummy,
                 "delta_dummy": self.delta_dummy,
                 "method": self.method,
                 "polarization_factor": self.polarization_factor,
                 # "filename": None,
                 "safe": self.safe,
                 "data": data,
                 "correctSolidAngle": self.correct_solid_angle,
                 "safe": self.safe,
                 "variance": variance
                 }

        if metadata is not None:
            kwarg["metadata"] = metadata

        if monitor is not None:
            kwarg["normalization_factor"] = monitor

        if self.do_2D():
            kwarg["npt_rad"] = self.nbpt_rad
            kwarg["npt_azim"] = self.nbpt_azim
            # if "filename" in kwarg:
            #    if self.extension:
            #        kwarg["filename"] += self.extension
            #    else:
            #        kwarg["filename"] += ".azim"
        else:
            kwarg["npt"] = self.nbpt_rad
            # if "filename" in kwarg:
            #    if self.extension:
            #        kwarg["filename"] += self.extension
            #    else:
            #        kwarg["filename"] += ".xy"
        kwarg["error_model"] = self.error_model

        if self.radial_range is not None:
            kwarg["radial_range"] = self.radial_range

        if self.azimuth_range is not None:
            kwarg["azimuth_range"] = self.azimuth_range

        error = None
        try:
            if self.do_2D():
                integrated_result = self.ai.integrate2d(**kwarg)
                self.radial = integrated_result.radial
                self.azimuthal = integrated_result.azimuthal
                result = integrated_result.intensity
                if variance is not None:
                    error = integrated_result.sigma
            else:
                integrated_result = self.ai.integrate1d(**kwarg)
                self.radial = integrated_result.radial
                self.azimuthal = None
                result = numpy.vstack(integrated_result).T

        except Exception as err:
            logger.debug("Backtrace", exc_info=True)
            err2 = ["error in integration do_2d: %s" % self.do_2D(),
                    str(err.__class__.__name__),
                    str(err),
                    "data.shape: %s" % (data.shape,),
                    "data.size: %s" % data.size,
                    "ai:",
                    str(self.ai),
                    "method:",
                    kwarg.get("method")
                    # str(self.ai._csr_integrator),
                    # "csr size: %s" % self.ai._lut_integrator.size
                    ]
            logger.error("\n".join(err2))
            raise err

        if writer is not None:
            writer.write(integrated_result)

        if self.output == "raw":
            return integrated_result
        elif self.output == "numpy":
            if (variance is not None) and (error is not None):
                return result, error
            else:
                return result

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

    def setDarkcurrentFile(self, imagefile):
        self.ai.set_darkfiles(imagefile)
        self.dark_current_image = imagefile

    def setFlatfieldFile(self, imagefile):
        self.ai.set_flatfiles(imagefile)
        self.flat_field_image = imagefile

    def setJsonConfig(self, json_file):
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                config = json.load(f)
        else:
            config = json.loads(json_file)
        self.set_config(config)

    def set_config(self, config, consume_keys=False):
        """
        Configure the working from the dictionary.

        :param dict config: Key-value configuration
        :param bool consume_keys: If true the keys from the dictionary will be
            consumed when used.
        """
        if not consume_keys:
            # Avoid to edit the input argument
            config = dict(config)

        integration_config.normalize(config, inplace=True)
        _init_ai(self.ai, config, consume_keys=True, read_maps=False)

        # Do it here before reading the AI to be able to catch the io
        filename = config.pop("mask_file", "")
        apply_process = config.pop("do_mask", True)
        if filename and apply_process:
            try:
                data = pyFAI.io.image.read_image_data(filename)
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s", filename, error)
            else:
                self.ai.mask = data

        # Do it here while we have to store metadata
        filename = config.pop("dark_current", "")
        apply_process = config.pop("do_dark", True)
        if filename and apply_process:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_darkcurrent(data)
            self.dark_current_image = "%s(%s)" % (method, ",".join(filenames))

        # Do it here while we have to store metadata
        filename = config.pop("flat_field", "")
        apply_process = config.pop("do_flat", True)
        if filename and apply_process:
            filenames = _normalize_filenames(filename)
            method = "mean"
            data = _reduce_images(filenames, method=method)
            self.ai.detector.set_flatfield(data)
            self.flat_field_image = "%s(%s)" % (method, ",".join(filenames))

        # Uses it anyway in case do_2D is customed after the configuration
        value = config.pop("nbpt_azim", None)
        if value:
            self.nbpt_azim = int(value)
        else:
            self.nbpt_azim = 1

        reader = integration_config.ConfigurationReader(config)
        self.method = reader.pop_method("csr")

        if self.method.dim == 1:
            self.nbpt_azim = 1

        value = config.pop("nbpt_rad", None)
        if value:
            self.nbpt_rad = int(value)

        value = config.pop("unit", units.TTH_DEG)
        self.unit = units.to_unit(value)

        value = config.pop("do_poisson", False)
        self.do_poisson = bool(value)

        value = config.pop("polarization_factor", None)
        apply_value = config.pop("do_polarization", True)
        if value and apply_value:
            self.polarization_factor = value
        else:
            self.polarization_factor = None

        value1 = config.pop("azimuth_range_min", None)
        value2 = config.pop("azimuth_range_max", None)
        apply_values = config.pop("do_azimuthal_range", True)
        if apply_values and value1 is not None and value2 is not None:
            self.azimuth_range = float(value1), float(value2)

        value1 = config.pop("radial_range_min", None)
        value2 = config.pop("radial_range_max", None)
        apply_values = config.pop("do_radial_range", True)
        if apply_values and value1 is not None and value2 is not None:
            self.radial_range = float(value1), float(value2)

        value = config.pop("do_solid_angle", True)
        self.correct_solid_angle = bool(value)

        self.dummy = config.pop("delta_dummy", None)
        self.delta_dummy = config.pop("val_dummy", None)
        apply_values = config.pop("do_dummy", True)
        if not apply_values:
            self.dummy, self.delta_dummy = None, None

        self._normalization_factor = config.pop("normalization_factor", None)

        if "monitor_name" in config:
            logger.warning("Monitor name defined but unsupported by the worker.")

        logger.info(self.ai.__repr__())
        self.reset()
        # For now we do not calculate the LUT as the size of the input image is unknown

    def set_unit(self, value):
        self._unit = units.to_unit(value)

    def get_unit(self):
        return self._unit

    unit = property(get_unit, set_unit)

    def set_error_model(self, value):
        if value == "poisson":
            self.do_poisson = True
        elif value is None or value == "":
            self.do_poisson = False
        else:
            raise RuntimeError("Unsupported error model '%s'" % value)

    def get_error_model(self):
        if self.do_poisson:
            return "poisson"
        return None

    error_model = property(get_error_model, set_error_model)

    def get_config(self):
        """Returns the configuration as a dictionary.

        FIXME: The returned dictionary is not exhaustive.
        """
        config = {"unit": str(self.unit)}
        for key in ["dist", "poni1", "poni2", "rot1", "rot3", "rot2", "pixel1", "pixel2", "splineFile", "wavelength"]:
            try:
                config[key] = self.ai.__getattribute__(key)
            except:
                pass
        for key in ["nbpt_azim", "nbpt_rad", "polarization_factor", "dummy", "delta_dummy",
                    "correct_solid_angle", "dark_current_image", "flat_field_image",
                    "mask_image", "do_poisson", "shape", "method"]:
            try:
                config[key] = self.__getattribute__(key)
            except:
                pass

        return config

    def get_json_config(self):
        """return configuration as a JSON string"""
        pass  # TODO

    def save_config(self, filename=None):
        if not filename:
            filename = self.config_file

    def warmup(self, sync=False):
        """
        Process a dummy image to ensure everything is initialized

        :param sync: wait for processing to be finished

        """
        t = threading.Thread(target=self.process,
                             name="init2d",
                             args=(numpy.zeros(self.shape, dtype=numpy.float32),))
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

    __call__ = process


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
        :param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context
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
        :return: processed data, optionally with the assiciated error if variance is provided
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
                           poissonian=0,
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
                 mask=None, dummy=None, delta_dummy=None, device=None):
        """Constructor of the worker
        :param dark: array
        :param flat: array
        :param solidangle: solid-angle array
        :param polarization: numpy array with 2D polarization corrections
        :param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context

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
            self.mask = (mask < 0)
        else:
            self.mask = mask.astype(bool)

        self.dummy = dummy
        self.delta_dummy = delta_dummy
        if device is not None:
            logger.warning("GPU is not yet implemented")

        if detector is None:
            self.distortion = None
        else:
            self.distortion = Distortion(detector, method="LUT", device=device,
                                         mask=self.mask, empty=self.dummy or 0)

    def process(self, data, variance=None,
                normalization_factor=1.0):
        """
        Process the data and apply a normalization factor
        :param data: input data
        :param variance: the variance associated to the data
        :param normalization: normalization factor
        :return: processed data
        """
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

        if self.distortion is not None:
            return self.distortion.correct(proc_data, self.dummy, self.delta_dummy)
        else:
            return data

    __call__ = process
