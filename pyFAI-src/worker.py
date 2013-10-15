#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
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
"""

This module contains the Worker class:

A tool able to perform azimuthal integration with:
additional saving capabilities like
- save as 2/3D structure in a HDF5 File
- read from HDF5 files

Aims at being integrated into a plugin like LImA or as model for the GUI 

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/10/2013"
__status__ = "development"

import threading, os
import logging
logger = logging.getLogger("pyFAI.worker")
import numpy
try:
    import h5py
except:
    h5py = None
from .azimuthalIntegrator import AzimuthalIntegrator
from . import units

class Worker(object):
    def __init__(self, azimuthalIntgrator=None, shapeIn=(2048, 2048), shapeOut=(360, 500), unit="r_mm"):
        """
        @param azimuthalIntgrator: pyFAI.AzimuthalIntegrator instance
        @param shapeIn: image size in input
        @param shapeOut: Integrated size: can be (1,2000) for 1D integration
        @param unit: can be "2th_deg, r_mm or q_nm^-1 ...
        """
        self._sem = threading.Semaphore()
        if azimuthalIntgrator is None:
            self.ai = pyFAI.AzimuthalIntegrator()
        else:
            self.ai = azimuthalIntgrator
        self.nbpt_azim, self.nbpt_rad = shapeOut
        self._unit = units.to_unit(unit)
        self.polarization = None
        self.dummy = None
        self.delta_dummy = None
        self.correct_solid_angle = True
        self.dark_current_image = None
        self.flat_field_image = None
        self.mask_image = None
        self.subdir = ""
        self.extension = None
        self.do_poisson = None
        self.needs_reset = True
        self.output = "numpy" #exports as numpy array by default
        self.shape = shapeIn
        self.method = "lut"

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
                "Directory: %s, \tExtension: %s" % (self.subdir, self.extension)]
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

        if self.do_2D():
            t = threading.Thread(target=self.ai.integrate2d,
                                 name="init2d",
                                 args=(numpy.zeros(self.shape, dtype=numpy.float32),
                                        self.nbpt_rad, self.nbpt_azim),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 )
        else:
            t = threading.Thread(target=self.ai.integrate1d,
                                 name="init1d",
                                 args=(numpy.zeros(self.shape, dtype=numpy.float32),
                                        self.nbpt_rad),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 )
        t.start()
        if sync:
            t.join()

    def process(self, data) :
        """
        Process a frame
        """
        kwarg = {"unit": self.unit,
                 "dummy": self.dummy,
                 "delta_dummy": self.delta_dummy,
                 "method": self.method,
                 "polarization_factor":self.polarization,
                 # "filename": None,
                 "safe": True,
                 "data": data,
                 }


        if self.do_2D():
            kwarg["nbPt_rad"] = self.nbpt_rad
            kwarg["nbPt_azim"] = self.nbpt_azim
            if "filename" in kwarg:
                if self.extension:
                    kwarg["filename"] += self.extension
                else:
                    kwarg["filename"] += ".azim"
        else:
            kwarg["nbPt"] = self.nbpt_rad
            if "filename" in kwarg:
                if self.extension:
                    kwarg["filename"] += self.extension
                else:
                    kwarg["filename"] += ".xy"
        if self.do_poisson:
            kwarg["error_model"] = "poisson"
        else:
            kwarg["error_model"] = None

        try:
            if self.do_2D():
                rData = self.ai.integrate2d(**kwarg)[0]
            else:
                rData = self.ai.integrate1d(**kwarg)[1]
        except:
            print(data.shape, data.size)
            print(self.ai)
            print(self.ai._lut_integrator)
            print(self.ai._lut_integrator.size)
            raise
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
                self.ai = pyFAI.load(poni)

        detector = config.get("detector", "detector")
        self.ai.detector = pyFAI.detectors.detector_factory(detector)

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
            self.nbpt_azim = int(config.get("azim_pt"))
        else:
            self.nbpt_azim = 1
        if config.get("rad_pt"):
            self.nbpt_rad = int(config.get("rad_pt"))
        self.unit = pyFAI.units.to_unit(config.get("unit", pyFAI.units.TTH_DEG))
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
