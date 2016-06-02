#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

from __future__ import with_statement, print_function

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "02/05/2016"
__status__ = "development"
__doc__ = """This module contains the Worker class:

A tool able to perform azimuthal integration with:
additional saving capabilities like
- save as 2/3D structure in a HDF5 File
- read from HDF5 files

Aims at being integrated into a plugin like LImA or as model for the GUI

The configuration of this class is mainly done via a dictionary transmitted as a JSON string:
Here are the valid keys:
    "dist",
    "poni1",
    "poni2",
    "rot1"
    "rot3"
    "rot2"
    "pixel1"
    "pixel2"

    "splineFile"
    "wavelength"

    "poni" #path of the file

    "chi_discontinuity_at_0"
    "do_mask"
    "do_dark"
    "do_azimuthal_range"
    "do_flat"
    "do_2D"
    "azimuth_range_min"
    "azimuth_range_max"

    "polarization_factor"
    "nbpt_rad"
    "do_solid_angle"
    "do_radial_range"
    "do_poisson"
    "delta_dummy"
    "nbpt_azim"
    "flat_field"
    "radial_range_min"
    "dark_current"
    "do_polarization"
    "mask_file"
    "detector"
    "unit"
    "radial_range_max"
    "val_dummy"
    "do_dummy"
    "method"
}

"""

import threading
import os.path
import logging
logger = logging.getLogger("pyFAI.worker")
import numpy
import fabio
from .detectors import detector_factory
from .azimuthalIntegrator import AzimuthalIntegrator
from .distortion import Distortion
from . import units
import json
# from .io import h5py, HDF5Writer


def make_ai(config):
    """Create an Azimuthal integrator from the configuration
    Static method !

    @param config: dict with all parameters
    @return: configured (but uninitialized) AzimuthalIntgrator
    """
    poni = config.get("poni")
    if poni and os.path.isfile(poni):
        ai = AzimuthalIntegrator.sload(poni)
    detector = config.get("detector", None)
    if detector:
        ai.detector = detector_factory(detector)

    wavelength = config.get("wavelength", 0)
    if wavelength:
        if wavelength <= 0 or wavelength > 1e-6:
            logger.warning("Wavelength is in meter ... unlikely value %s" % wavelength)
        ai.wavelength = wavelength

    splinefile = config.get("splineFile")
    if splinefile and os.path.isfile(splinefile):
        ai.detector.splineFile = splinefile

    for key in ("pixel1", "pixel2", "dist", "poni1", "poni2", "rot1", "rot2", "rot3"):
        value = config.get(key)
        if value is not None:
            ai.__setattr__(key, value)
    if config.get("chi_discontinuity_at_0"):
        ai.setChiDiscAtZero()

    mask_file = config.get("mask_file")
    if mask_file and config.get("do_mask"):
        if os.path.exists(mask_file):
            try:
                mask = fabio.open(mask_file).data
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s" % (mask_file, error))
            else:
                ai.mask = mask

    dark_files = [i.strip() for i in config.get("dark_current", "").split(",")
                  if os.path.isfile(i.strip())]
    if dark_files and config.get("do_dark"):
        ai.set_darkfiles(dark_files)

    flat_files = [i.strip() for i in config.get("flat_field", "").split(",")
                  if os.path.isfile(i.strip())]
    if flat_files and config.get("do_flat"):
        ai.set_flatfiles(flat_files)

    return ai


class Worker(object):
    def __init__(self, azimuthalIntgrator=None,
                 shapeIn=(2048, 2048), shapeOut=(360, 500),
                 unit="r_mm", dummy=None, delta_dummy=None):
        """
        @param azimuthalIntgrator: pyFAI.AzimuthalIntegrator instance
        @param shapeIn: image size in input
        @param shapeOut: Integrated size: can be (1,2000) for 1D integration
        @param unit: can be "2th_deg, r_mm or q_nm^-1 ...
        """
        self._sem = threading.Semaphore()
        if azimuthalIntgrator is None:
            self.ai = AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntgrator
#        self.config = {}
#        self.config_file = "azimInt.json"
#        self.nbpt_azim = 0
#        if type(config) == dict:
#            self.config = config
#        elif type(config) in types.StringTypes:
#            if os.path.isfile(config):
#                self.config = json.load(open(config, "r"))
#                self.config_file(config)
#            else:
#                self.config = json.loads(config)
#        if self.config:
#            self.configure()
        self._normalization_factor = None  # Value of the monitor: divides the intensity by this value for normalization
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self._unit = units.to_unit(unit)
        self.polarization = None
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
        self.method = "lut"
        self.radial = None
        self.azimuthal = None
        self.radial_range = None
        self.azimuth_range = None
        self.is_safe = True

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Azimuthal Integrator:", self.ai.__repr__(),
                  "Input image shape: %s" % list(self.shape),
                  "Number of points in radial direction: %s" % self.nbpt_rad,
                  "Number of points in azimuthal direction: %s" % self.nbpt_azim,
                  "Unit in radial dimension: %s" % self.unit.REPR,
                  "Correct for solid angle: %s" % self.correct_solid_angle,
                  "Polarization factor: %s" % self.polarization,
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
        # print self.__repr__()

    def reconfig(self, shape=(2048, 2048), sync=False):
        """
        This is just to force the integrator to initialize with a given input image shape

        @param shape: shape of the input image
        @param sync: return only when synchronized
        """
        self.shape = shape
        self.ai.reset()
        self.warmup(sync)

    def process(self, data):
        """
        Process a frame
        #TODO:
        dark, flat, sa are missing

        @param: data: numpy array containing the input image
        """

        with self._sem:
            monitor = self._normalization_factor
        kwarg = {"unit": self.unit,
                 "dummy": self.dummy,
                 "delta_dummy": self.delta_dummy,
                 "method": self.method,
                 "polarization_factor":self.polarization,
                 # "filename": None,
                 "safe": self.is_safe,
                 "data": data,
                 "correctSolidAngle": self.correct_solid_angle,
                 }

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
        if self.do_poisson:
            kwarg["error_model"] = "poisson"
        else:
            kwarg["error_model"] = None

        if self.radial_range is not None:
            kwarg["radial_range"] = self.radial_range

        if self.azimuth_range is not None:
            kwarg["azimuth_range"] = self.azimuth_range

        try:
            if self.do_2D():
                rData, self.radial, self.azimuthal = self.ai.integrate2d(**kwarg)
            else:
                rData = self.ai.integrate1d(**kwarg)
                self.radial = rData[0]
                self.azimuthal = None
                rData = numpy.vstack(rData).T

        except Exception as err:
            err2 = ["error in integration",
                    str(err),
                    "data.shape: %s" % (data.shape,),
                    "data.size: %s" % data.size,
                    "ai:",
                    str(self.ai),
                    "csr:",
                    # str(self.ai._csr_integrator),
                    # "csr size: %s" % self.ai._lut_integrator.size
                    ]
            logger.error(err2)
            raise err

        if self.output == "numpy":
            return rData

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

    def setJsonConfig(self, jsonconfig):
        print("start config ...")
        if os.path.isfile(jsonconfig):
            config = json.load(open(jsonconfig, "r"))
        else:
            config = json.loads(jsonconfig)
        if "poni" in config:
            poni = config["poni"]
            if poni and os.path.isfile(poni):
                self.ai = AzimuthalIntegrator.sload(poni)

        detector = config.get("detector", "detector")
        self.ai.detector = detector_factory(detector)

        if "wavelength" in config:
            wavelength = config["wavelength"]
            try:
                fwavelength = float(wavelength)
            except ValueError:
                logger.error("Unable to convert wavelength to float: %s" % wavelength)
            else:
                if fwavelength <= 0 or fwavelength > 1e-6:
                    logger.warning("Wavelength is in meter ... unlikely value %s" % fwavelength)
                self.ai.wavelength = fwavelength

        splineFile = config.get("splineFile")
        if splineFile and os.path.isfile(splineFile):
            self.ai.detector.splineFile = splineFile
        self.ai.pixel1 = float(config.get("pixel1", 1))
        self.ai.pixel2 = float(config.get("pixel2", 1))
        self.ai.dist = config.get("dist", 1)
        self.ai.poni1 = config.get("poni1", 0)
        self.ai.poni2 = config.get("poni2", 0)
        self.ai.rot1 = config.get("rot1", 0)
        self.ai.rot2 = config.get("rot2", 0)
        self.ai.rot3 = config.get("rot3", 0)

        if config.get("chi_discontinuity_at_0"):
            self.ai.setChiDiscAtZero()
        else:
            self.ai.setChiDiscAtPi()

        mask_file = config.get("mask_file")
        do_mask = config.get("do_mask")
        if mask_file and os.path.exists(mask_file) and do_mask:
            try:
                mask = fabio.open(mask_file).data
            except Exception as error:
                logger.error("Unable to load mask file %s, error %s" % (mask_file, error))
            else:
                self.ai.mask = mask
                self.mask_image = os.path.abspath(mask_file)

        self.ai.set_darkfiles([i.strip() for i in config.get("dark_current", "").split(",")
                               if os.path.isfile(i.strip())])
        self.ai.set_flatfiles([i.strip() for i in config.get("flat_field", "").split(",")
                               if os.path.isfile(i.strip())])
        self.dark_current_image = self.ai.darkfiles
        self.flat_field_image = self.ai.flatfiles
        if config.get("do_2D"):
            self.nbpt_azim = int(config.get("nbpt_azim"))
        else:
            self.nbpt_azim = 1
        if config.get("nbpt_rad"):
            self.nbpt_rad = int(config["nbpt_rad"])
        self.unit = units.to_unit(config.get("unit", units.TTH_DEG))
        self.do_poisson = config.get("do_poisson")
        if config.get("do_polarization"):
            self.polarization = config.get("polarization")
        else:
            self.polarization = None
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
        """return configuration as a dictionary"""
        config = {"unit": str(self.unit)}
        for key in ["dist", "poni1", "poni2", "rot1", "rot3", "rot2", "pixel1", "pixel2", "splineFile", "wavelength"]:
            try:
                config[key] = self.ai.__getattribute__(key)
            except:
                pass
        for key in ["nbpt_azim", "nbpt_rad", "polarization", "dummy", "delta_dummy",
                    "correct_solid_angle", "dark_current_image", "flat_field_image",
                    "mask_image", "do_poisson", "shape", "method"]:
            try:
                config[key] = self.__getattribute__(key)
            except:
                pass

        return config
#
#    "poni" #path of the file
#
#    "chi_discontinuity_at_0"
#    "do_mask"
#    "do_dark"
#    "do_azimuthal_range"
#    "do_flat"
#    "do_2D"
#    "azimuth_range_min"
#    "azimuth_range_max"
#
#    "polarization_factor"
#    "nbpt_rad"
#    "do_solid_angle"
#    "do_radial_range"
#    "do_poisson"
#    "delta_dummy"
#    "nbpt_azim"
#    "flat_field"
#    "radial_range_min"
#    "dark_current"
#    "do_polarization"
#    "mask_file"
#    "detector"
#    "unit"
#    "radial_range_max"
#    "val_dummy"
#    "do_dummy"
#    "method"
# }

    def get_json_config(self):
        """return configuration as a JSON string"""
        pass  # TODO

    def save_config(self, filename=None):
        if not filename:
            filename = self.config_file

    def warmup(self, sync=False):
        """
        Process a dummy image to ensure everything is initialized

        @param sync: wait for processing to be finished

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


class PixelwiseWorker(object):
    """
    Simple worker doing dark, flat, solid angle and polarization correction
    """
    def __init__(self, dark=None, flat=None, solidangle=None, polarization=None,
                 mask=None, dummy=None, delta_dummy=None, device=None):
        """
        @param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context
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

    def process(self, data, normalization=None):
        """
        Process the data and apply a normalization factor
        @param data: input data
        @param normalization: normalization factor
        @return processed data
        """
        _shape = data.shape
        #        ^^^^   this is why data is mandatory !
        if self.dummy is not None:
            if self.delta_dummy is None:
                self.mask = numpy.logical_or((data == self.dummy), self.mask)
            else:
                self.mask = numpy.logical_or(abs(data - self.dummy) <= self.delta_dummy,
                                             self.mask)
            do_mask = True
        else:
            do_mask = (self.mask is not False)
        # Explicitly make an copy !
        data = numpy.array(data, dtype=numpy.float32)
        if self.dark is not None:
            data -= self.dark
        if self.flat is not None:
            data /= self.flat
        if self.solidangle is not None:
            data /= self.solidangle
        if self.polarization is not None:
            data /= self.polarization
        if normalization is not None:
            data /= normalization
        if do_mask:
            data[self.mask] = self.dummy or 0
        return data


class DistortionWorker(object):
    """
    Simple worker doing dark, flat, solid angle and polarization correction
    """
    def __init__(self, detector=None, dark=None, flat=None, solidangle=None, polarization=None,
                 mask=None, dummy=None, delta_dummy=None, device=None):
        """
        @param device: Used to influance OpenCL behavour: can be "cpu", "GPU", "Acc" or even an OpenCL context
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

        if detector is None:
            self.distortion = None
        else:
            self.distortion = Distortion(detector, method="CSR", device=device)

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

    def process(self, data, normalization=None):
        """
        Process the data and apply a normalization factor
        @param data: input data
        @param normalization: normalization factor
        @return processed data
        """
        _shape = data.shape
        #        ^^^^   this is why data is mandatory !
        if self.dummy is not None:
            if self.delta_dummy is None:
                self.mask = numpy.logical_or((data == self.dummy), self.mask)
            else:
                self.mask = numpy.logical_or(abs(data - self.dummy) <= self.delta_dummy,
                                        self.mask)
            do_mask = True
        else:
            do_mask = (self.mask is not False)
        # Explicitly make an copy !
        data = numpy.array(data, dtype=numpy.float32)
        if self.dark is not None:
            data -= self.dark
        if self.flat is not None:
            data /= self.flat
        if self.solidangle is not None:
            data /= self.solidangle
        if self.polarization is not None:
            data /= self.polarization

        if do_mask:
            data[self.mask] = self.dummy or 0
        if self.distortion is not None:
            return self.distortion.correct(data)
        else:
            return data



