# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2024 Australian Synchrotron
#                  2024-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Emily Massahud
#                            Jerome Kieffer
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

"""Module containing SingleGeometry class."""

from __future__ import annotations

__authors__ = ["Emily Massahud", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/03/2024"
__status__ = "development"

import logging
import os

import numpy
from silx.image import marchingsquares

from .azimuthalIntegrator import AzimuthalIntegrator
from .control_points import ControlPoints
from .detectors import detector_factory
from .ext.mathutil import build_qmask
from .geometry import Geometry
from .geometryRefinement import GeometryRefinement
from .massif import Massif
from .ring_extraction import RingExtraction
from .utils import StringTypes

logger = logging.getLogger(__name__)

class SingleGeometry(object):
    """This class represents a single geometry of a detector position on a
    goniometer arm
    """

    def __init__(self, label, image=None, metadata=None, pos_function=None,
                 control_points=None, calibrant=None, detector=None, geometry=None):
        """Constructor of the SingleGeometry class, used for calibrating a
        multi-geometry setup with a moving detector.

        :param label: name of the geometry, a string or anything unmutable
        :param image: image with Debye-Scherrer rings as 2d numpy array
        :param metadata: anything which contains the goniometer position
        :param pos_function: a function which takes the metadata as input
                                 and returns the goniometer arm position
        :param control_points: a pyFAI.control_points.ControlPoints instance
            (optional parameter)
        :param calibrant: a pyFAI.calibrant.Calibrant instance.
                        Contains the wavelength to be used (optional parameter)
        :param detector: a pyFAI.detectors.Detector instance or something like
                        that Contains the mask to be used (optional parameter)
        :param geometry: an azimuthal integrator or a ponifile
                        (or a dict with the geometry) (optional parameter)
        """
        dict_geo = {}
        self.label = label
        self.image = image
        self.metadata = metadata  # may be anything
        self.calibrant = calibrant
        if control_points is None or isinstance(control_points, ControlPoints):
            self.control_points = control_points
        else:
            # Probaly a NPT file
            self.control_points = ControlPoints(control_points, calibrant=calibrant)

        if detector is not None:
            self.detector = detector_factory(detector)
        else:
            self.detector = None
        if isinstance(geometry, Geometry):
            dict_geo = geometry.getPyFAI()
        elif isinstance(geometry, StringTypes) and os.path.exists(geometry):
            dict_geo = Geometry.sload(geometry).getPyFAI()
        elif isinstance(geometry, dict):
            dict_geo = geometry

        if self.detector is not None:
            dict_geo["detector"] = self.detector
        if self.control_points is not None:
            dict_geo["data"] = self.control_points.getList()
        if self.calibrant is not None:
            dict_geo["calibrant"] = self.calibrant
            if self.calibrant.wavelength:
                dict_geo["wavelength"] = self.calibrant.wavelength
        if "max_shape" in dict_geo:
            # not used in constructor
            dict_geo.pop("max_shape")
        self.geometry_refinement = GeometryRefinement(**dict_geo)
        if self.detector is None:
            self.detector = self.geometry_refinement.detector
        self.pos_function = pos_function
        self.massif = None

    def get_position(self):
        """This method  is in charge of calculating the motor position from metadata/label/..."""
        return self.pos_function(self.metadata)

    def extract_cp(self, max_rings: int | None = None, pts_per_deg: float = 1.0, Imin:float = 0):
        """Performs an automatic keypoint extraction and updates the geometry refinement part

        :param max_ring: extract at most N rings from the image
        :param pts_per_deg: number of control points per azimuthal degree (increase for better
            precision)
        :param Imin: minimum of intensity above the background to keep the point
        """

        ring_extractor = RingExtraction(
            self.image, self.detector, self.calibrant, self.geometry_refinement, self.massif
        )
        return ring_extractor.extract_control_points(max_rings, pts_per_deg, Imin)

    def get_ai(self):
        """Create a new azimuthal integrator to be used.

        :return: Azimuthal Integrator instance
        """
        config = self.geometry_refinement.get_config()
        ai = AzimuthalIntegrator()
        ai.set_config(config)
        return ai

    def get_wavelength(self):
        assert self.calibrant.wavelength == self.geometry_refinement.wavelength
        return self.geometry_refinement.wavelength

    def set_wavelength(self, value):
        self.calibrant.setWavelength_change2th(value)
        self.geometry_refinement.set_wavelength(value)

    wavelength = property(get_wavelength, set_wavelength)
