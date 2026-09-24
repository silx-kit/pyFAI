# !/usr/bin/env python
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
__date__ = "28/08/2026"
__status__ = "production"
__docformat__ = 'restructuredtext'


import logging
from collections import namedtuple
from math import cos, sin

from ..detectors import Detector
from ..io.ponifile import PoniFile
from ..units import LENGTH_UNITS, to_unit
from .fit2d import convert_to_Fit2d
from .utils import FLIPPED_AXES

logger = logging.getLogger(__name__)

ORIENTATION_TO_FLIP_MATRIX = {1: (-1, +1),
                              2: (-1, -1),
                              3: (+1, -1),
                              4: (+1, +1)}
"""ImageD11 flip matrix (o11, o22) for each pyFAI detector orientation.

Orientation 3, the native one in pyFAI, already needs ``o22 = -1`` because the
horizontal transverse axis points in opposite directions in the two
conventions: towards the center of the storage ring for pyFAI, away from it for
ImageD11. The mirrored axes of the other orientations then compose with it.
"""



_ImageD11Geometry = namedtuple("_ImageD11Geometry",
                               "distance o11 o12 o21 o22 tilt_x tilt_y tilt_z wavelength y_center y_size z_center z_size spline shape",
                               defaults=[None]*15)

class ImageD11Geometry(_ImageD11Geometry):
    """ This object represents the geometry as configured in Fit2D

    :param directDist: Distance from sample to the detector along the incident beam in mm. The detector may be extrapolated when tilted.
    :param centerX: Position of the beam-center on the detector in pixels, along the fastest axis of the image.
    :param centerY: Position of the beam-center on the detector in pixels, along the slowest axis of the image.
    :param tilt: Angle of tilt of the detector in degrees
    :param tiltPlanRotation: Direction of the tilt (undefined when tilt is 0)
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
        """workaround for bug in ImageD11"""
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
    orientation = detector.orientation
    if orientation not in ORIENTATION_TO_FLIP_MATRIX:
        raise ValueError(f"Invalid orientation {orientation}, expected 1 to 4")
    o11, o22 = ORIENTATION_TO_FLIP_MATRIX[orientation]
    # pyFAI mirrors the pixel *index* while ImageD11 mirrors the *coordinate*
    # about the beam center, so the mirrored axes need their center re-expressed.
    flip_slow, flip_fast = FLIPPED_AXES[orientation]
    shape = detector.shape or detector.max_shape
    if (flip_slow or flip_fast) and not shape:
        raise ValueError(f"The detector shape is needed to convert orientation {orientation}")
    id11 = {"o11": o11, "o12": 0, "o21": 0, "o22": o22}
    id11["distance"] = (f2d.directDist or 0) * 1e-3 * distance_unit.scale
    # Fit2D counts the beam center half a pixel further than ImageD11 does
    z_center = (f2d.centerY or 0) - 0.5  # in pixel
    y_center = (f2d.centerX or 0) - 0.5  # in pixel
    if flip_slow:
        z_center = shape[0] - 1 - z_center
    if flip_fast:
        y_center = shape[1] - 1 - y_center
    id11["y_center"] = y_center
    id11["z_center"] = z_center
    id11["tilt_x"] = poni.rot3
    id11["tilt_y"] = poni.rot2
    id11["tilt_z"] = -poni.rot1
    if poni.wavelength:
        id11["wavelength"] = poni.wavelength * wavelength_unit.scale
    id11["y_size"] = detector.pixel2 * distance_unit.scale
    id11["z_size"] = detector.pixel1 * distance_unit.scale
    id11["shape"] = detector.shape or detector.max_shape
    id11["spline"] = detector.splinefile

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

    if o12 or o21 or abs(o11) != 1 or abs(o22) != 1:
        raise RuntimeError("Transposed orientations are not supported")
    try:
        orientation = next(key for key, value in ORIENTATION_TO_FLIP_MATRIX.items()
                           if value == (o11, o22))
    except StopIteration:
        raise RuntimeError(f"No orientation matches the flip matrix ({o11}, {o22})") from None
    flipped = FLIPPED_AXES[orientation]

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
    poni._rot1 = -(id11.tilt_z or 0)
    distance = (id11.distance or 0) / len_scale
    poni._dist = distance * cos(poni.rot2) * cos(poni.rot1)
    pixel_v = (id11.z_size or 0) / len_scale
    pixel_h = (id11.y_size or 0) / len_scale
    shape = id11.shape
    z_center = id11.z_center or 0.0
    y_center = id11.y_center or 0.0
    if flipped[0] or flipped[1]:
        if not shape:
            raise ValueError(f"The detector shape is needed to convert orientation {orientation}")
        if flipped[0]:
            z_center = shape[0] - 1 - z_center
        if flipped[1]:
            y_center = shape[1] - 1 - y_center
    # ImageD11 counts the beam center half a pixel before Fit2D and pyFAI do
    poni._poni1 = -distance * sin(poni.rot2) + pixel_v * (z_center + 0.5)
    poni._poni2 = +distance * cos(poni.rot2) * sin(poni.rot1) + pixel_h * (y_center + 0.5)
    spline = id11.spline
    poni._detector = Detector(pixel1=pixel_v, pixel2=pixel_h, splinefile=spline, max_shape=shape, orientation=orientation)
    wl = id11.wavelength
    if wl:
        poni._wavelength = wl / wl_scale

    return poni
