# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2025 European Synchrotron Radiation Facility, Grenoble, France
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

"""This modules contains helper function to convert to/from FIT2D geometry
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "15/11/2025"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import typing
import logging
from ..utils.dataclasses import case_insensitive_dataclass
from math import pi, cos, sin, sqrt, acos, asin
from ..detectors import Detector
from ..io.ponifile import PoniFile
logger = logging.getLogger(__name__)

def degrees(rad:float) -> float:
    return 180 * rad / pi

def radians(deg:float) -> float:
    return deg * pi / 180


@case_insensitive_dataclass(slots=True)
class Fit2dGeometry:
    """ This object represents the geometry as configured in Fit2D.

    It behaves like a dataclass, is case insensitive and can behave like a dict as well but cannnot be extended.

    :param directDist: Distance from sample to the detector along the incident beam in mm. The detector may be extrapolated when tilted.
    :param centerX: Position of the beam-center on the detector in pixels, along the fastest axis of the image.
    :param centerY: Position of the beam-center on the detector in pixels, along the slowest axis of the image.
    :param tilt: Angle of tilt of the detector in degrees
    :param tiltPlanRotation: Direction of the tilt (undefined when tilt is 0)
    :param detector: Detector definition as is pyFAI.
    :param wavelength: Wavelength of the beam in Angstrom
    """
    directDist: float = None
    centerX: float = None
    centerY: float = None
    tilt: float = 0.0
    tiltPlanRotation: float = 0.0
    pixelX: float = None
    pixelY: float = None
    splinefile: str = None
    detector: Detector = None
    wavelength: float = None

    @classmethod
    def _fromdict(cls, dico):
        "Mirror of _asdict: take the dict and populate the tuple to be returned"
        obj = cls(**dico)
        return obj

    def _asdict(self):
        "Mirror of _asdict method from NamedTuple"
        return {k: self.__getattr__(k) for k in typing.get_type_hints(self.__class__)}

    def __repr__(self):
        return f"DirectBeamDist= {self.directDist:.3f} mm\tCenter: x={self.centerX:.3f}, y={self.centerY:.3f} pix\t"\
               f"Tilt= {self.tilt:.3f}° tiltPlanRotation= {self.tiltPlanRotation:.3f}°" + \
               (f" \N{GREEK SMALL LETTER LAMDA}= {self.wavelength:.3f}\N{LATIN CAPITAL LETTER A WITH RING ABOVE}" if self.wavelength else "")

    # dict-like interface:
    def __getitem__(self, key:str):
        return self.__getattr__(key)
    def __setitem__(self, key:str, value):
        self.__setattr__(key, value)
    def get(self, key:str, default=None):
        if key.lower() in self._ci_map:
            return self.__getattr__(key)
        return default
    def __contains__(self, key:str):
        return key.lower() in self._ci_map
    def __iter__(self):
        yield from self._ci_map.values()
    def keys(self):
        return self._ci_map.values()
    def values(self):
        return [self.__getattr__(i) for i in self._ci_map.values()]
    def items(self):
        return [(i, self.__getattr__(i)) for i in self._ci_map.values()]
    def __len__(self):
        return self._ci_map.__len__()


def convert_to_Fit2d(poni):
    """Convert a Geometry|PONI object to the geometry of Fit2D
    Please see the doc from Fit2dGeometry

    :param poni: azimuthal integrator, geometry or poni
    :return: same geometry as a Fit2dGeometry named-tuple
    """
    poni = PoniFile(poni)

    cos_tilt = cos(poni._rot1) * cos(poni._rot2)

    sin_tilt = sqrt(1.0 - cos_tilt * cos_tilt)
    tan_tilt = sin_tilt / cos_tilt
    # This is tilt plane rotation
    if sin_tilt == 0:
        # tilt plan rotation is undefined when there is no tilt!, does not matter
        cos_tilt = 1.0
        sin_tilt = 0.0
        cos_tpr = 1.0
        sin_tpr = 0.0

    else:
        cos_tpr = max(-1.0, min(1.0, -cos(poni._rot2) * sin(poni._rot1) / sin_tilt))
        sin_tpr = sin(poni._rot2) / sin_tilt
    directDist = 1.0e3 * poni._dist / cos_tilt
    tilt = degrees(acos(cos_tilt))
    if sin_tpr < 0:
        tpr = -degrees(acos(cos_tpr))
    else:
        tpr = degrees(acos(cos_tpr))

    centerX = (poni._poni2 + poni.dist * tan_tilt * cos_tpr) / poni.detector.pixel2
    if abs(tilt) < 1e-5:  # in degree
        centerY = (poni.poni1) / poni.detector.pixel1
    else:
        centerY = (poni._poni1 + poni.dist * tan_tilt * sin_tpr) / poni.detector.pixel1
    out = poni.detector.getFit2D()
    out["directDist"] = directDist
    out["centerX"] = centerX
    out["centerY"] = centerY
    out["tilt"] = tilt
    out["tiltPlanRotation"] = tpr
    out["detector"] = poni.detector
    out["pixelX"] = poni.detector.pixel2 * 1e6
    out["pixelY"] = poni.detector.pixel1 * 1e6
    out["splineFile"] = poni.detector.splinefile
    if poni.wavelength:
        out["wavelength"] = poni.wavelength * 1e10
    return Fit2dGeometry._fromdict(out)


def convert_from_Fit2d(f2d):
    """Import the geometry from Fit2D

    :param f2d: instance of Fit2dGeometry
    :return: PoniFile instance
    """
    if not isinstance(f2d, Fit2dGeometry):
        if isinstance(f2d, dict):
            f2d = Fit2dGeometry._fromdict(f2d)
        else:
            f2d = Fit2dGeometry(f2d)
    res = PoniFile()
    try:
        cos_tilt = cos(radians(f2d.tilt))
        sin_tilt = sin(radians(f2d.tilt))
        cos_tpr = cos(radians(f2d.tiltPlanRotation))
        sin_tpr = sin(radians(f2d.tiltPlanRotation))
    except AttributeError as error:
        logger.error(("Got strange results with tilt=%s"
                      " and tiltPlanRotation=%s: %s"),
                     f2d.tilt, f2d.tiltPlanRotation, error)
    if f2d.detector is None or f2d.detector.pixel1 is None:
        cfg = {}
        if f2d.splineFile:
            cfg["splinefile"] = os.path.abspath(f2d.splineFile)
        if f2d.pixelX and f2d.pixelY:
            cfg["pixel1"] = f2d.pixelY * 1.0e-6
            cfg["pixel2"] = f2d.pixelX * 1.0e-6
        detector = Detector(**cfg)
    elif isinstance(f2d.detector, Detector):
        detector = f2d.detector
    else:
        detector = Detector.factory(f2d.detector)

    res._detector = detector
    if f2d.wavelength:
        res._wavelength = f2d.wavelength * 1e-10
    res._dist = f2d.directDist * cos_tilt * 1.0e-3
    res._poni1 = f2d.centerY * detector.pixel1 - f2d.directDist * sin_tilt * sin_tpr * 1.0e-3
    res._poni2 = f2d.centerX * detector.pixel2 - f2d.directDist * sin_tilt * cos_tpr * 1.0e-3
    res._rot2 = rot2 = asin(sin_tilt * sin_tpr)  # or pi-
    rot1 = acos(min(1.0, max(-1.0, (cos_tilt / sqrt(1.0 - (sin_tpr * sin_tilt) ** 2)))))  # + or -
    if cos_tpr * sin_tilt > 0:
        rot1 = -rot1
    res._rot1 = rot1
    if abs(cos_tilt - cos(rot1) * cos(rot2)) >= 1e-6:
        raise RuntimeError("Inconsistency in geometry conversion")
    if f2d.tilt == 0.0:
        rot3 = 0
    else:
        rot3 = acos(min(1.0, max(-1.0, (cos_tilt * cos_tpr * sin_tpr - cos_tpr * sin_tpr) / sqrt(1.0 - sin_tpr * sin_tpr * sin_tilt * sin_tilt))))  # + or -
        rot3 = pi / 2.0 - rot3
    res._rot3 = rot3
    return res
