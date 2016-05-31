#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
#
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
__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "31/05/2016"
__status__ = "DEPRECATED -> see worker"
__docformat__ = 'restructuredtext'

import os
import logging
import types
import json
import numpy
import threading
import fabio
logger = logging.getLogger("pyFAI.processor")
from . import azimuthalIntegrator
from . import units
from .utils import float_, int_
from .detectors import detector_factory
AzimuthalIntegrator = azimuthalIntegrator.AzimuthalIntegrator


class Processor(object):
    """
    This class can be set-up from a configuration file to define an azimuthal integrator
    with all pre-processing and all post processing configured
    """
    def __init__(self, config=None):
        self.ai = AzimuthalIntegrator()
        self.config = {}
        self.config_file = "azimInt.json"
        self.nbpt_azim = 0
        if type(config) == dict:
            self.config = config
        elif type(config) in types.StringTypes:
            if os.path.isfile(config):
                self.config = json.load(open(config, "r"))
                self.config_file(config)
            else:
                self.config = json.loads(config)
        if self.config:
            self.configure()

    def __repr__(self):
        """
        pretty print of myself
        """
        lstout = ["Azimuthal Integrator:", self.ai.__repr__(),
                "Input image shape: %s" % list(self.shapeIn),
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
        logger.debug("did a reset")
        self.ai.reset()

    def configure(self, config=None):

        if config is None:
            config = self.config
        if not config:
            config = {}

        if "poni" in config:
            poni = config["poni"]
            if poni and os.path.isfile(poni):
                self.ai = AzimuthalIntegrator.loads(poni)
        else:
            self.ai = AzimuthalIntegrator()
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
            self.nbpt_rad = int(config.get("nbpt_rad"))
        self.unit = units.to_unit(config.get("unit", units.TTH_DEG))
        self.do_poisson = config.get("do_poisson")
        if config.get("do_polarization"):
            self.polarization = config.get("polarization")
        else:
            self.polarization = None
        logger.info(self.ai.__repr__())
        self.reset()
        # For now we do not calculate the LUT as the size of the input image is unknown

    def get_config(self):
        """
        retrieves the configuration
        """
        to_save = { "poni": str(self.poni.text()).strip(),
                    "detector": str(self.detector.currentText()).lower(),
                    "wavelength":float_(self.wavelength.text()),
                    "splineFile":str(self.splineFile.text()).strip(),
                    "pixel1": float_(self.pixel1.text()),
                    "pixel2":float_(self.pixel2.text()),
                    "dist":float_(self.dist.text()),
                    "poni1":float_(self.poni1.text()),
                    "poni2":float_(self.poni2.text()),
                    "rot1":float_(self.rot1.text()),
                    "rot2":float_(self.rot2.text()),
                    "rot3":float_(self.rot3.text()),
                    "do_dummy": bool(self.do_dummy.isChecked()),
                    "do_mask":  bool(self.do_mask.isChecked()),
                    "do_dark": bool(self.do_dark.isChecked()),
                    "do_flat": bool(self.do_flat.isChecked()),
                    "do_polarization":bool(self.do_polarization.isChecked()),
                    "val_dummy":float_(self.val_dummy.text()),
                    "delta_dummy":float_(self.delta_dummy.text()),
                    "mask_file":str(self.mask_file.text()).strip(),
                    "dark_current":str(self.dark_current.text()).strip(),
                    "flat_field":str(self.flat_field.text()).strip(),
                    "polarization_factor":float_(self.polarization_factor.value()),
                    "nbpt_rad":int_(self.rad_pt.text()),
                    "do_2D":bool(self.do_2D.isChecked()),
                    "nbpt_azim":int_(self.azim_pt.text()),
                    "chi_discontinuity_at_0": bool(self.chi_discontinuity_at_0.isChecked()),
                    "do_solid_angle": bool(self.do_solid_angle.isChecked()),
                    "do_radial_range": bool(self.do_radial_range.isChecked()),
                    "do_azimuthal_range": bool(self.do_azimuthal_range.isChecked()),
                    "do_poisson": bool(self.do_poisson.isChecked()),
                    "radial_range_min":float_(self.radial_range_min.text()),
                    "radial_range_max":float_(self.radial_range_max.text()),
                    "azimuth_range_min":float_(self.azimuth_range_min.text()),
                    "azimuth_range_max":float_(self.azimuth_range_max.text()),
                   }
        if self.q_nm.isChecked():
            to_save["unit"] = "q_nm^-1"
        elif self.q_A.isChecked():
            to_save["unit"] = "q_A^-1"
        elif self.tth_deg.isChecked():
            to_save["unit"] = "2th_deg"
        elif self.tth_rad.isChecked():
            to_save["unit"] = "2th_rad"
        elif self.r_mm.isChecked():
            to_save["unit"] = "r_mm"
        try:
            with open(filename, "w") as myFile:
                json.dump(to_save, myFile, indent=4)
        except IOError as error:
            logger.error("Error while saving config: %s" % error)
        else:
            logger.debug("Saved")

    def save_config(self, filename=None):
        if not filename:
            filename = self.config_file


    def warmup(self):
        """
        Reset and Process a dummy image to ensure eveything is initialized
        """
        self.ai.reset()

        if self.do_2D():
            threading.Thread(target=self.ai.integrate2d,
                                 name="init2d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad, self.nbpt_azim),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 ).start()
        else:
            threading.Thread(target=self.ai.integrate1d,
                                 name="init1d",
                                 args=(numpy.zeros(self.shapeIn, dtype=numpy.float32),
                                        self.nbpt_rad),
                                 kwargs=dict(method="lut", unit=self.unit)
                                 ).start()
