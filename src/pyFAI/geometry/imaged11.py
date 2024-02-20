# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2019-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""This modules contains helper function to convert to/from ImageD11 geometry
"""

__author__ = "Jérôme Kieffer, Carsten DETLEFS"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/02/2024"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
logger = logging.getLogger(__name__)
from math import cos, sin
from .fit2d import convert_to_Fit2d
from ..units import to_unit, LENGTH_UNITS
from ..detectors import Detector
from ..io.ponifile import PoniFile
from collections import namedtuple

_ImageD11Geometry = namedtuple("_ImageD11Geometry",
                               "distance o11 o12 o21 o22 tilt_x tilt_y tilt_z wavelength y_center y_size z_center z_size spline shape",
                               defaults=[None]*15)

class ImageD11Geometry(_ImageD11Geometry):
    """ This object represents the geometry as configured in Fit2D

    :param directDist: Distance from sample to the detector along the incident beam in mm. The detector may be extrapolated when tilted.
    :param centerX: Position of the beam-center on the detector in pixels, along the fastest axis of the image.
    :param centerY: Position of the beam-center on the detector in pixels, along the slowest axis of the image.
    :param tilt: Angle of tilt of the detector in degrees
    :param tiltPlanRotation: Direction of the tilt (unefined when tilt is 0)
    :param detector: Detector definition as is pyFAI.
    :param wavelength: Wavelength of the beam in Angstrom
    """

    def __new__(cls, *args, **kwargs):
        self = super().__new__(cls, *args, **kwargs)
        self._distance_unit = None
        self._wavelength_unit = None
        return self

    @classmethod
    def _fromdict(cls, dico, distance_unit=None, wavelength_unit=None):
        "Mirror of _asdict: take the dict and populate the tuple to be returned"
        try:
            obj = cls(**dico)
            obj._wavelength_unit = to_unit(wavelength_unit, LENGTH_UNITS)
            obj._distance_unit = to_unit(distance_unit, LENGTH_UNITS)
        except TypeError:# as err:
            # logger.warning("TypeError: %s", err)
            obj = cls(**{key: dico[key] for key in [i for i in cls._fields if i in dico]})
            if "wavelength_unit" in dico:
                obj._wavelength_unit = to_unit(dico["wavelength_unit"], LENGTH_UNITS)
            if "distance_unit" in dico:
                obj._distance_unit = to_unit(dico["distance_unit"], LENGTH_UNITS)
        return obj

    def _asdict(self):
        """work arround for bug in ImageD11"""
        dico = super()._asdict()
        if self.spline is None:
            dico.pop("spline")
        return dico

    @property
    def wavelength_unit(self):
        return self._wavelength_unit
    @wavelength_unit.setter
    def wavelength_unit(self, value):
        if self._wavelength_unit is None:
            self._wavelength_unit = value
        else:
            raise TypeError(f"{type(self)} object does not support item assignment")

    @property
    def distance_unit(self):
        """ This is the unit of the distance and the pixel size"""
        return self._distance_unit
    @distance_unit.setter
    def distance_unit(self, value):
        """ This is the unit of the distance and the pixel size"""
        if self._distance_unit is None:
            self._distance_unit = value
        else:
            raise TypeError(f"{type(self)} object does not support item assignment")

def convert_to_ImageD11(poni, distance_unit="µm", wavelength_unit="nm"):
    """Convert a Geometry|PONI object to the geometry of ImageD11
    Please see the doc in  doc/source/geometry_conversion.rst or
    http://www.silx.org/doc/pyFAI/latest/geometry_conversion.html#geometry-definition-of-imaged11

    :param poni: azimuthal integrator, geometry or poni
    :param distance_unit: unit used for distance and pixel size in ImageD11
    :param wavelength_unit: unit used for wavelength
    :return: same geometry as a Fit2dGeometry named-tuple
    """
    poni = PoniFile(poni)
    detector = poni.detector
    distance_unit = to_unit(distance_unit, LENGTH_UNITS)
    wavelength_unit = to_unit(wavelength_unit, LENGTH_UNITS)
    f2d = convert_to_Fit2d(poni)
    # TODO: manage orientation here
    id11 = {"o11": 1,
            "o12": 0,
            "o21": 0,
            "o22": -1}
    id11["distance"] = (f2d.directDist or 0) * 1e-3 * distance_unit.scale
    id11["y_center"] = (f2d.centerX or 0)  # in pixel
    id11["z_center"] = (f2d.centerY or 0)  # in pixel
    id11["tilt_x"] = poni.rot3
    id11["tilt_y"] = poni.rot2
    id11["tilt_z"] = -poni.rot1
    if poni.wavelength:
        id11["wavelength"] = poni.wavelength * wavelength_unit.scale
    id11["y_size"] = detector.pixel2 * distance_unit.scale
    id11["z_size"] = detector.pixel1 * distance_unit.scale
    id11["shape"] = detector.shape or detector.max_shape
    id11["spline"] = detector.splineFile

    return ImageD11Geometry._fromdict(id11, distance_unit=distance_unit, wavelength_unit=wavelength_unit)

def convert_from_ImageD11(id11):
    """Set the geometry from the parameter set which contains distance,
    o11, o12, o21, o22, tilt_x, tilt_y tilt_z, wavelength, y_center, y_size,
    z_center and z_size.
    Please refer to the documentation in doc/source/geometry_conversion.rst
    http://www.silx.org/doc/pyFAI/latest/geometry_conversion.html#geometry-definition-of-imaged11
    for the orientation and units of those values.

    :param id11: ImageD11Geometry instance or dict with the values to set.
    :return: PoniFile like object
    """
    if isinstance(id11, dict):
        id11 = ImageD11Geometry._fromdict(id11)

    o11 = id11.o11
    o12 = id11.o12
    o21 = id11.o21
    o22 = id11.o22

    # TODO: double check !!!
    if  (o11, o12, o21, o22) == (1, 0, 0, -1):
        orientation = 3
    elif  (o11, o12, o21, o22) == (1, 0, 0, +1):
        orientation = 1
    elif (o11, o12, o21, o22) == (-1, 0, 0, -1):
        orientation = 4
    elif (o11, o12, o21, o22) == (-1, 0, 0, +1):
        orientation = 2
    else:
        raise RuntimeError("rotated orientations are not supported")

    if id11.wavelength_unit:
        wl_scale = id11.wavelength_unit.scale
    else:
        wl_scale = 1e9 # nm by default (compatibility with implementation from Carsten in 2019)
    if id11.distance_unit:
        len_scale = id11.distance_unit.scale
    else:
        len_scale = 1e6 # µm by default (compatibility with implementation from Carsten in 2019)

    poni = PoniFile()
    poni._rot3 = id11.tilt_x or 0
    poni._rot2 = id11.tilt_y or 0
    poni._rot1 = -id11.tilt_z or 0
    distance = (id11.distance or 0) / len_scale
    poni._dist = distance * cos(poni.rot2) * cos(poni.rot1)
    pixel_v = (id11.z_size or 0) / len_scale
    pixel_h = (id11.y_size or 0) / len_scale
    poni._poni1 = -distance * sin(poni.rot2) + pixel_v * (id11.z_center or 0.0)
    poni._poni2 = +distance * cos(poni.rot2) * sin(poni.rot1) + pixel_h * (id11.y_center or 0)
    shape = id11.shape
    spline = id11.spline
    poni._detector = Detector(pixel1=pixel_v, pixel2=pixel_h, splineFile=spline, max_shape=shape, orientation=orientation)
    wl = id11.wavelength
    if wl:
        poni._wavelength = wl / wl_scale

    return poni
