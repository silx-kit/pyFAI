#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://forge.epn-campus.eu/projects/azimuthal
#
#    File: "$Id$"
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
__date__ = "04/09/2013"
__status__ = "developement"
__docformat__ = 'restructuredtext'

import os
import logging
import types
import fabio
logger = logging.getLogger("pyFAI.processor")
from . import azimuthalIntegratory
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

    def save_config(self):
        pass

    def warmup(self):
        """
        Reset and Process a dummy image to ensure eveything is initialized
        """
        self.shapeIn
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
