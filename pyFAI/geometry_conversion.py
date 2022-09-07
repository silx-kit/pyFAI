# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2022 European Synchrotron Radiation Facility, Grenoble, France
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

"""This modules contains only one (large) class in charge of:

* calculating the geometry, i.e. the position in the detector space of each pixel of the detector
* manages caches to store intermediate results

NOTA: The Geometry class is not a "transformation class" which would take a
detector and transform it. It is rather a description of the experimental setup.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/09/2022"
__status__ = "production"
__docformat__ = 'restructuredtext'

import os
import logging
logger = logging.getLogger(__name__)
from collections import namedtuple
from math import pi, cos, sin, sqrt, acos, asin
degrees = lambda  x: 180*x/pi
radians = lambda  x: x*pi/180 
from .detectors import Detector
from .io.ponifile import PoniFile

Fit2dGeometry = namedtuple("Fit2dGeometry", "directDist centerX centerY tilt tiltPlanRotation pixelX pixelY splineFile",
                           defaults=[None,]*8)
Fit2dGeometry.__doc__ = "distance is in mm, angles in degree, beam-center in pixels and pixel size in micron"


def convert_to_Fit2d(poni):
    """Convert a Geometry|PONI object to the geometry of Fit2D
    Please see the doc from Fit2dGeometry
    
    :param poni: azimuthal integrator, geometry or poni
    :return: same geometry as a Fit2dGeometry named-tuple
    """
    print(poni.detector)
    poni = PoniFile(poni)
    print(poni.detector)

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
    return Fit2dGeometry(**out)


def convert_from_Fit2d(f2d):
    """Import the geometry from Fit2D
    
    :param f2d: instance of Fit2dGeometry
    :return: PoniFile instance
    """  
    if not isinstance(f2d, Fit2dGeometry):
        if isinstance(f2d, dict):
            f2d = Fit2dGeometry(**f2d)
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
    
    detector = Detector()
    if f2d.splineFile:
        detector.splineFile = os.path.abspath(f2d.splineFile)
    elif f2d.pixelX and f2d.pixelY:
        detector.pixel1 = f2d.pixelY * 1.0e-6
        detector.pixel2 = f2d.pixelX * 1.0e-6

    print(detector)
    res._detector = detector
    print(res.detector)
    res._dist = f2d.directDist * cos_tilt * 1.0e-3
    res._poni1 = f2d.centerY * detector.pixel1 - f2d.directDist * sin_tilt * sin_tpr * 1.0e-3
    res._poni2 = f2d.centerX * detector.pixel2 - f2d.directDist * sin_tilt * cos_tpr * 1.0e-3
    res._rot2 = rot2 = asin(sin_tilt * sin_tpr)  # or pi-
    rot1 = acos(min(1.0, max(-1.0, (cos_tilt / sqrt(1.0 - (sin_tpr * sin_tilt) ** 2)))))  # + or -
    if cos_tpr * sin_tilt > 0:
        rot1 = -rot1
    res._rot1 = rot1
    assert abs(cos_tilt - cos(rot1) * cos(rot2)) < 1e-6
    if f2d.tilt == 0.0:
        rot3 = 0
    else:
        rot3 = acos(min(1.0, max(-1.0, (cos_tilt * cos_tpr * sin_tpr - cos_tpr * sin_tpr) / sqrt(1.0 - sin_tpr * sin_tpr * sin_tilt * sin_tilt))))  # + or -
        rot3 = pi / 2.0 - rot3
    res._rot3 = rot3
    return res
    
    