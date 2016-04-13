# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/04/2016"
__status__ = "production"
__docformat__ = 'restructuredtext'

import logging
from numpy import radians, degrees, arccos, arctan2, sin, cos, sqrt
import numpy
import os
import threading
import time

from . import detectors
from . import units
from .decorators import deprecated
from .utils import expand2d
try:
    from .third_party import six
except ImportError:
    import six
StringTypes = (six.binary_type, six.text_type)

logger = logging.getLogger("pyFAI.geometry")


try:
    from .ext import _geometry
except ImportError:
    _geometry = None

try:
    from .ext import bilinear
except ImportError:
    bilinear = None

try:
    from .ext.fastcrc import crc32
except ImportError:
    from zlib import crc32


class Geometry(object):
    """
    This class is an azimuthal integrator based on P. Boesecke's geometry and
    histogram algorithm by Manolo S. del Rio and V.A Sole

    Detector is assumed to be corrected from "raster orientation" effect.
    It is not addressed here but rather in the Detector object or at read time.
    Considering there is no tilt:

    - Detector fast dimension (dim2) is supposed to be horizontal
      (dimension X of the image)

    - Detector slow dimension (dim1) is supposed to be vertical, upwards
      (dimension Y of the image)

    - The third dimension is chose such as the referential is
      orthonormal, so dim3 is along incoming X-ray beam


    Demonstration of the equation done using Mathematica.
    -----------------------------------------------------

    Axis 1 is along first dimension of detector (when not tilted),
    this is the slow dimension of the image array in C or Y
    x1={1,0,0}

    Axis 2 is along second dimension of detector (when not tilted),
    this is the fast dimension of the image in C or X
    x2={0,1,0}

    Axis 3 is along the incident X-Ray beam
    x3={0,0,1}

    We define the 3 rotation around axis 1, 2 and 3:

    rotM1 = RotationMatrix[rot1,x1] =  {{1,0,0},{0,cos[rot1],-sin[rot1]},{0,sin[rot1],cos[rot1]}}
    rotM2 =  RotationMatrix[rot2,x2] = {{cos[rot2],0,sin[rot2]},{0,1,0},{-sin[rot2],0,cos[rot2]}}
    rotM3 =  RotationMatrix[rot3,x3] = {{cos[rot3],-sin[rot3],0},{sin[rot3],cos[rot3],0},{0,0,1}}

    Rotations of the detector are applied first Rot around axis 1,
    then axis 2 and finally around axis 3:

    R = rotM3.rotM2.rotM1

    R = {{cos[rot2] cos[rot3],cos[rot3] sin[rot1] sin[rot2]-cos[rot1] sin[rot3],cos[rot1] cos[rot3] sin[rot2]+sin[rot1] sin[rot3]},
          {cos[rot2] sin[rot3],cos[rot1] cos[rot3]+sin[rot1] sin[rot2] sin[rot3],-cos[rot3] sin[rot1]+cos[rot1] sin[rot2] sin[rot3]},
          {-sin[rot2],cos[rot2] sin[rot1],cos[rot1] cos[rot2]}}

    In Python notation:

    R.x1 = [cos(rot2)*cos(rot3),cos(rot2)*sin(rot3),-sin(rot2)]

    R.x2 = [cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3),cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3), cos(rot2)*sin(rot1)]

    R.x3 = [cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3),-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3), cos(rot1)*cos(rot2)]

    * Coordinates of the Point of Normal Incidence:

      PONI = R.{0,0,L}

      PONI = [L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3)),
                   L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)),L*cos(rot1)*cos(rot2)]

    * Any pixel on detector plan at coordinate (d1, d2) in
      meters. Detector is at z=L

      P={d1,d2,L}

      R.P = [t1, t2, t3]
      t1 = R.P.x1 = d1*cos(rot2)*cos(rot3) + d2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) + L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
      t2 = R.P.x2 = d1*cos(rot2)*sin(rot3)  + d2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3)) + L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3))
      t3 = R.P.x3 = d2*cos(rot2)*sin(rot1) - d1*sin(rot2) + L*cos(rot1)*cos(rot2)

    * Distance sample (origin) to detector point (d1,d2)

      |R.P| = sqrt(pow(Abs(L*cos(rot1)*cos(rot2) + d2*cos(rot2)*sin(rot1) - d1*sin(rot2)),2) +
                        pow(Abs(d1*cos(rot2)*cos(rot3) + d2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) +
                        L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))),2) +
                        pow(Abs(d1*cos(rot2)*sin(rot3) + L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +
                        d2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))),2))

    *  cos(2theta) is defined as (R.P component along x3) over the distance from origin to data point |R.P|

    tth = ArcCos [-(R.P).x3/|R.P|]

    tth = Arccos((-(L*cos(rot1)*cos(rot2)) - d2*cos(rot2)*sin(rot1) + d1*sin(rot2))/
                        sqrt(pow(Abs(L*cos(rot1)*cos(rot2) + d2*cos(rot2)*sin(rot1) - d1*sin(rot2)),2) +
                          pow(Abs(d1*cos(rot2)*cos(rot3) + d2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) +
                         L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))),2) +
                          pow(Abs(d1*cos(rot2)*sin(rot3) + L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +
                         d2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))),2)))

    * tan(2theta) is defined as sqrt(t1**2 + t2**2) / t3

    tth = ArcTan2 [sqrt(t1**2 + t2**2) , t3 ]

    Getting 2theta from it's tangeant seems both more precise (around
    beam stop very far from sample) and faster by about 25% Currently
    there is a swich in the method to follow one path or the other.

    * Tangeant of angle chi is defined as (R.P component along x1)
      over (R.P component along x2). Arctan2 should be used in actual
      calculation

     chi = ArcTan[((R.P).x1) / ((R.P).x2)]

     chi = ArcTan2(d1*cos(rot2)*cos(rot3) + d2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) +
                            L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3)),
                          d1*cos(rot2)*sin(rot3) + L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +
                            d2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3)))

    """

    def __init__(self, dist=1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0,
                 pixel1=None, pixel2=None, splineFile=None, detector=None, wavelength=None):
        """
        @param dist: distance sample - detector plan (orthogonal distance, not along the beam), in meter.
        @param poni1: coordinate of the point of normal incidence along the detector's first dimension, in meter
        @param poni2: coordinate of the point of normal incidence along the detector's second dimension, in meter
        @param rot1: first rotation from sample ref to detector's ref, in radians
        @param rot2: second rotation from sample ref to detector's ref, in radians
        @param rot3: third rotation from sample ref to detector's ref, in radians
        @param pixel1: pixel size of the fist dimension of the detector,  in meter
        @param pixel2: pixel size of the second dimension of the detector,  in meter
        @param splineFile: file containing the geometric distortion of the detector. Overrides the pixel size.
        """
        self._dist = dist
        self._poni1 = poni1
        self._poni2 = poni2
        self._rot1 = rot1
        self._rot2 = rot2
        self._rot3 = rot3
        self.param = [self._dist, self._poni1, self._poni2,
                      self._rot1, self._rot2, self._rot3]
        self.chiDiscAtPi = True  # chi discontinuity (radians), pi by default
        self._cached_array = {}  # dict for caching all arrays
        self._dssa = None
        self._dssa_crc = None  # checksum associated with _dssa
        self._dssa_order = 3  # by default we correct for 1/cos(2th), fit2d corrects for 1/cos^3(2th)
        self._corner4Da = None  # actual 4d corner array
        self._corner4Ds = None  # space for the corner array, 2th, q, r, ...
        self._wavelength = wavelength
        self._oversampling = None
        self._correct_solid_angle_for_spline = True
        self._sem = threading.Semaphore()
        self._polarization_factor = 0
        self._polarization_axis_offset = 0
        self._polarization = None
        self._polarization_crc = None  # checksum associated with _polarization
        self._cosa = None  # cosine of the incidance angle
        self._transmission_normal = None
        self._transmission_corr = None
        self._transmission_crc = None

        if detector:
            if isinstance(detector, StringTypes):
                self.detector = detectors.detector_factory(detector)
            else:
                self.detector = detector
        else:
            self.detector = detectors.Detector()
        if splineFile:
            self.detector.splineFile = os.path.abspath(splineFile)
        elif pixel1 and pixel2:
            self.detector.pixel1 = pixel1
            self.detector.pixel2 = pixel2

    def __repr__(self, dist_unit="m", ang_unit="rad", wl_unit="m"):
        """Nice representation of the class

        @param dist_unit: units for distances
        @param ang_unit: units used for angles
        @param wl_unit: units used for wavelengths
        @return: nice string representing the configuration in use
        """
        dist_unit = units.to_unit(dist_unit, units.LENGTH_UNITS) or units.l_m
        ang_unit = units.to_unit(ang_unit, units.ANGLE_UNITS) or units.A_rad
        wl_unit = units.to_unit(wl_unit, units.LENGTH_UNITS) or units.l_m
        self.param = [self._dist, self._poni1, self._poni2,
                      self._rot1, self._rot2, self._rot3]
        lstTxt = [self.detector.__repr__()]
        if self._wavelength:
            lstTxt.append("Wavelength= %.6e%s" %
                          (self._wavelength * wl_unit.scale, wl_unit.REPR))
        lstTxt.append(("SampleDetDist= %.6e%s\tPONI= %.6e, %.6e%s\trot1=%.6f"
                       "  rot2= %.6f  rot3= %.6f %s") %
                      (self._dist * dist_unit.scale, dist_unit.REPR, self._poni1 * dist_unit.scale,
                       self._poni2 * dist_unit.scale, dist_unit.REPR,
                      self._rot1 * ang_unit.scale, self._rot2 * ang_unit.scale,
                      self._rot3 * ang_unit.scale, ang_unit.REPR))
        if self.detector.pixel1:
            f2d = self.getFit2D()
            lstTxt.append(("DirectBeamDist= %.3fmm\tCenter: x=%.3f, y=%.3f pix"
                           "\tTilt=%.3f deg  tiltPlanRotation= %.3f deg") %
                          (f2d["directDist"], f2d["centerX"], f2d["centerY"],
                           f2d["tilt"], f2d["tiltPlanRotation"]))
        return os.linesep.join(lstTxt)

    def _calc_cartesian_positions(self, d1, d2, poni1=None, poni2=None):
        """
        Calculate the position in cartesian coordinate (centered on the PONI)
        and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        @param d1: ndarray of dimention 1/2 containing the Y pixel positions
        @param d2: ndarray of dimention 1/2 containing the X pixel positions
        @param poni1: value in the Y direction of the poni coordinate (meter)
        @param poni2: value in the X direction of the poni coordinate (meter)
        @return: 2-arrays of same shape as d1 & d2 with the position in meter

        d1 and d2 must have the same shape, returned array will have
        the same shape.
        """
        if poni1 is None:
            poni1 = self.poni1
        if poni2 is None:
            poni2 = self.poni2

        p1, p2, p3 = self.detector.calc_cartesian_positions(d1, d2)
        return p1 - poni1, p2 - poni2, p3

    def calc_pos_zyx(self, d0=None, d1=None, d2=None, param=None, corners=False, use_cython=True):
        """Calculate the position of a set of points in space in the sample's centers referential.

        This is usually used for calculating the pixel position in space.


        @param d0: altitude on the point compared to the detector (i.e. z), may be None
        @param d1: position on the detector along the slow dimention (i.e. y)
        @param d2: position on the detector along the fastest dimention (i.e. x)
        @param corners: return positions on the corners (instead of center)
        @return 3-tuple of nd-array,  with  dim0=along the beam,
                                            dim1=along slowest dimension
                                            dim2=along fastest dimension
        """
        if param is None:
            dist, poni1, poni2, rot1, rot2, rot3 = self._dist, self._poni1, self._poni2, self._rot1, self._rot2, self._rot3
        else:
            dist, poni1, poni2, rot1, rot2, rot3 = param[:6]

        if (not corners) and ((d1 is None) or (d2 is None)):
            raise RuntimeError("input corrdiate d1 and d2 are mandatory")
        if d0 is None:
            L = dist
        else:
            L = dist + d0
        if corners:
            tmp = self.detector.get_pixel_corners()
            p1 = tmp[..., 1]
            p2 = tmp[..., 2]
            p3 = tmp[..., 0]
        else:
            p1, p2, p3 = self.detector.calc_cartesian_positions(d1, d2)
        if use_cython and _geometry:
            t3, t1, t2 = _geometry.calc_pos_zyx(L, poni1, poni2, rot1, rot2, rot3, p1, p2, p3)
        else:
            p1 = p1 - poni1
            p2 = p2 - poni2
            # we make copies
            if p3 is not None:
                L = L + p3

            cosRot1 = cos(rot1)
            cosRot2 = cos(rot2)
            cosRot3 = cos(rot3)
            sinRot1 = sin(rot1)
            sinRot2 = sin(rot2)
            sinRot3 = sin(rot3)
            t1 = p1 * cosRot2 * cosRot3 + \
                 p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) - \
                 L * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3)
            t2 = p1 * cosRot2 * sinRot3 + \
                 p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3) - \
                 L * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3)
            t3 = p1 * sinRot2 - p2 * cosRot2 * sinRot1 + L * cosRot1 * cosRot2
        return (t3, t1, t2)

    def tth(self, d1, d2, param=None, path="cython"):
        """
        Calculates the 2theta value for the center of a given pixel
        (or set of pixels)

        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @param path: can be "cos", "tan" or "cython"
        @return: 2theta in radians
        @rtype: float or array of floats.
        """

        if path == "cython" and _geometry:
            if param is None:
                dist, poni1, poni2, rot1, rot2, rot3 = self._dist, self._poni1, \
                                self._poni2, self._rot1, self._rot2, self._rot3
            else:
                dist, poni1, poni2, rot1, rot2, rot3 = param[:6]
            p1, p2, p3 = self._calc_cartesian_positions(d1, d2, poni1, poni2)
            tmp = _geometry.calc_tth(L=dist,
                                     rot1=rot1,
                                     rot2=rot2,
                                     rot3=rot3,
                                     pos1=p1,
                                     pos2=p2,
                                     pos3=p3)
        else:
            t3, t1, t2 = self.calc_pos_zyx(d0=None, d1=d1, d2=d2, param=param)
            if path == "cos":
                tmp = arccos(t3 / sqrt(t1 ** 2 + t2 ** 2 + t3 ** 2))
            else:
                tmp = arctan2(sqrt(t1 ** 2 + t2 ** 2), t3)
        return tmp

    def qFunction(self, d1, d2, param=None, path="cython"):
        """
        Calculates the q value for the center of a given pixel (or set
        of pixels) in nm-1

        q = 4pi/lambda sin( 2theta / 2 )

        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: q in in nm^(-1)
        @rtype: float or array of floats.
        """
        if not self.wavelength:
            raise RuntimeError(("Scattering vector q cannot be calculated"
                                " without knowing wavelength !!!"))

        if _geometry and path == "cython":
            if param is None:
                dist, poni1, poni2, rot1, rot2, rot3 = self._dist, self._poni1, \
                                self._poni2, self._rot1, self._rot2, self._rot3
            else:
                dist, poni1, poni2, rot1, rot2, rot3 = param[:6]

            p1, p2, p3 = self._calc_cartesian_positions(d1, d2, poni1, poni2)
            out = _geometry.calc_q(L=dist,
                                   rot1=rot1,
                                   rot2=rot2,
                                   rot3=rot3,
                                   pos1=p1,
                                   pos2=p2,
                                   pos3=p3,
                                   wavelength=self.wavelength)
        else:
            out = 4.0e-9 * numpy.pi / self.wavelength * \
                numpy.sin(self.tth(d1=d1, d2=d2, param=param, path=path) / 2.0)
        return out

    def rFunction(self, d1, d2, param=None, path="cython"):
        """
        Calculates the radius value for the center of a given pixel
        (or set of pixels) in m

          r = distance to the incident beam

        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: r in in m
        @rtype: float or array of floats.
        """

        if _geometry and path == "cython":
            if param is None:
                dist, poni1, poni2, rot1, rot2, rot3 = self._dist, self._poni1, \
                                self._poni2, self._rot1, self._rot2, self._rot3
            else:
                dist, poni1, poni2, rot1, rot2, rot3 = param[:6]

            p1, p2, p3 = self._calc_cartesian_positions(d1, d2, poni1, poni2)
            out = _geometry.calc_r(L=dist,
                                   rot1=rot1,
                                   rot2=rot2,
                                   rot3=rot3,
                                   pos1=p1,
                                   pos2=p2,
                                   pos3=p3)
        else:
            # Before 03/2016 it was the distance at beam-center
            # cosTilt = cos(self._rot1) * cos(self._rot2)
            # directDist = self._dist / cosTilt  # in m
            # out = directDist * numpy.tan(self.tth(d1=d1, d2=d2, param=param))
            _, t1, t2 = self.calc_pos_zyx(d0=None, d1=d1, d2=d2, param=param)
            out = numpy.sqrt(t1 * t1 + t2 * t2)
        return out

    def qArray(self, shape=None):
        """
        Generate an array of the given shape with q(i,j) for all
        elements.
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)
        if self._cached_array.get("q_center") is None:
            with self._sem:
                if self._cached_array.get("q_center") is None:
                    qa = numpy.fromfunction(self.qFunction, shape,
                                            dtype=numpy.float32)
                    self._cached_array["q_center"] = qa

        return self._cached_array["q_center"]

    def rArray(self, shape=None):
        """Generate an array of the given shape with r(i,j) for all elements;
        The radius r being  in meters.

        @param shape: expected shape of the detector
        @return: 2d array of the given shape with radius in m from beam center on detector.
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if self._cached_array.get("r_center") is None:
            with self._sem:
                if self._cached_array.get("r_center") is None:
                    self._cached_array["r_center"] = numpy.fromfunction(self.rFunction, shape,
                                                                        dtype=numpy.float32)
        return self._cached_array.get("r_center")

    def rd2Array(self, shape=None):
        """Generate an array of the given shape with (d*(i,j))^2 for all pixels.

        d*^2 is the reciprocal spacing squared in inverse nm squared

        @param shape: expected shape of the detector
        @return:2d array of the given shape with reciprocal spacing squared
        """
        qArray = self.qArray(shape)
        if self._cached_array.get("d*2_center") is None:
            with self._sem:
                if self._cached_array.get("d*2_center") is None:
                    self._cached_array["d*2_center"] = (qArray / (2.0 * numpy.pi)) ** 2
        return self._cached_array["d*2_center"]

    @deprecated
    def qCornerFunct(self, d1, d2):
        """Calculate the q_vector for any pixel corner (in nm^-1)

        @param shape: expected shape of the detector
        """
        return self.qFunction(d1 - 0.5, d2 - 0.5)

    @deprecated
    def rCornerFunct(self, d1, d2):
        """
        Calculate the radius array for any pixel corner (in m)
        """
        return self.rFunction(d1 - 0.5, d2 - 0.5)

    @deprecated
    def tth_corner(self, d1, d2):
        """
        Calculates the 2theta value for the corner of a given pixel
        (or set of pixels)

        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: 2theta in radians
        @rtype: floar or array of floats.
        """
        return self.tth(d1 - 0.5, d2 - 0.5)

    def twoThetaArray(self, shape=None):
        """Generate an array of two-theta(i,j) in radians for each pixel in detector

        the 2theta array values are in radians

        @param shape: shape of the detector
        @return: array of 2theta position in radians
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if self._cached_array.get("2th_center") is None:
            with self._sem:
                if self._cached_array.get("2th_center") is None:
                    ttha = numpy.fromfunction(self.tth,
                                              shape,
                                              dtype=numpy.float32)
                    self._cached_array["2th_center"] = ttha
        return self._cached_array["2th_center"]

    def chi(self, d1, d2, path="cython"):
        """
        Calculate the chi (azimuthal angle) for the centre of a pixel
        at coordinate d1,d2 which in the lab ref has coordinate:

        X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
        X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
        X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
        hence tan(Chi) =  X2 / X1

        @param d1: pixel coordinate along the 1st dimention (C convention)
        @type d1: float or array of them
        @param d2: pixel coordinate along the 2nd dimention (C convention)
        @type d2: float or array of them
        @param path: can be "tan" (i.e via numpy) or "cython"
        @return: chi, the azimuthal angle in rad
        """
        p1, p2, p3 = self._calc_cartesian_positions(d1, d2, self._poni1, self._poni2)

        if path == "cython" and _geometry:
            tmp = _geometry.calc_chi(L=self._dist,
                                     rot1=self._rot1, rot2=self._rot2, rot3=self._rot3,
                                     pos1=p1, pos2=p2, pos3=p3)
            tmp.shape = d1.shape
        else:
            cosRot1 = cos(self._rot1)
            cosRot2 = cos(self._rot2)
            cosRot3 = cos(self._rot3)
            sinRot1 = sin(self._rot1)
            sinRot2 = sin(self._rot2)
            sinRot3 = sin(self._rot3)
            L = self._dist
            if p3 is not None:
                L = L + p3
            num = p1 * cosRot2 * cosRot3 \
                + p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) \
                - L * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3)
            den = p1 * cosRot2 * sinRot3 \
                - L * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3) \
                + p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3)
            tmp = numpy.arctan2(num, den)
        return tmp

    def chi_corner(self, d1, d2):
        """
        Calculate the chi (azimuthal angle) for the corner of a pixel
        at coordinate d1,d2 which in the lab ref has coordinate:

        X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
        X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
        X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
        hence tan(Chi) =  X2 / X1

        @param d1: pixel coordinate along the 1st dimention (C convention)
        @type d1: float or array of them
        @param d2: pixel coordinate along the 2nd dimention (C convention)
        @type d2: float or array of them
        @return: chi, the azimuthal angle in rad
        """
        return self.chi(d1 - 0.5, d2 - 0.5)

    def chiArray(self, shape=None):
        """Generate an array of azimuthal angle chi(i,j) for all elements in the detector.

        Azimuthal angles are in radians

        Nota: Refers to the pixel centers !

        @param shape: the shape of the chi array
        @return: the chi array as numpy.ndarray
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if self._cached_array.get("chi_center") is None:
            chia = numpy.fromfunction(self.chi, shape,
                                      dtype=numpy.float32)
            if not self.chiDiscAtPi:
                chia = chia % (2.0 * numpy.pi)
            self._cached_array["chi_center"] = chia
        return self._cached_array["chi_center"]

    def positionArray(self, shape=None, corners=False, dtype=numpy.float64):
        """Generate an array for the pixel position given the shape of the detector.

        if corners is False, the coordinates of the center of the pixel
        is returned in an array of shape: (shape[0], shape[1], 3)
        where the 3 coordinates are:
        * z: along incident beam,
        * y: to the top/sky,
        * x: towards the center of the ring

        If is True, the corner of each pixels are then returned.
        the output shape is then (shape[0], shape[1], 4, 3)

        @param shape: shape of the array expected
        @param corners: set to true to receive a (...,4,3) array of corner positions
        @param dtype: output format requested
        @return: 3D coodinates as nd-array of size (...,3) or (...,3) (default)

        Nota: this value is not cached and actually generated on demand (costly)
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        pos = numpy.fromfunction(lambda d1, d2: self.calc_pos_zyx(None, d1, d2, corners=corners),
                                 shape,
                                 dtype=dtype)
        outshape = pos[0].shape + (3,)
        tpos = numpy.empty(outshape, dtype=dtype)
        for idx in range(3):
            tpos[..., idx] = pos[idx]
        return tpos

    def corner_array(self, shape=None, unit="2th", use_cython=True):
        """
        Generate a 3D array of the given shape with (i,j) (radial
        angle 2th, azimuthal angle chi ) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are:
           * dim3[0]: radial angle 2th, q, r, ...
           * dim3[1]: azimuthal angle chi
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if unit:
            unit = units.to_unit(unit)
            space = unit.REPR.split("_")[0]
        else:
            unit = None
            space = None
            if (self._corner4Da is not None) and (shape == self._corner4Da.shape[:2]):
                return self._corner4Da
        if self._corner4Da is None or self._corner4Ds != space or shape != self._corner4Da.shape[:2]:
            with self._sem:
                if self._corner4Da is None or self._corner4Ds != space or shape != self._corner4Da.shape[:2]:
                    corners = None
                    if use_cython:
                        if self.detector.IS_CONTIGUOUS:
                            d1 = expand2d(numpy.arange(shape[0] + 1.0), shape[1] + 1.0, False)
                            d2 = expand2d(numpy.arange(shape[1] + 1.0), shape[0] + 1.0, True)
                            p1, p2, p3 = self.detector.calc_cartesian_positions(d1, d2, center=False, use_cython=True)
                        else:
                            det_corners = self.detector.get_pixel_corners()
                            p1 = det_corners[..., 1]
                            p2 = det_corners[..., 2]
                            p3 = det_corners[..., 0]
                        try:
                            res = _geometry.calc_rad_azim(self.dist, self.poni1, self.poni2,
                                                          self.rot1, self.rot2, self.rot3,
                                                          p1, p2, p3,
                                                          space, self._wavelength)
                        except KeyError:
                            logger.warning("No fast path for space: %s", space)
                        else:
                            if self.detector.IS_CONTIGUOUS:
                                if bilinear:
                                    # convert_corner_2D_to_4D needs contiguous arrays as input
                                    radi = numpy.ascontiguousarray(res[..., 0], numpy.float32)
                                    azim = numpy.ascontiguousarray(res[..., 1], numpy.float32)
                                    corners = bilinear.convert_corner_2D_to_4D(2, radi, azim)
                                else:
                                    corners = numpy.zeros((shape[0], shape[1], 4, 2),
                                                          dtype=numpy.float32)
                                    corners[:, :, 0, :] = res[:-1, :-1, :]
                                    corners[:, :, 1, :] = res[1:, :-1, :]
                                    corners[:, :, 2, :] = res[1:, 1:, :]
                                    corners[:, :, 3, :] = res[:-1, 1:, :]
                            else:
                                corners = res

                    if corners is None:
                        # In case the fast-path is not implemented
                        pos = self.positionArray(shape, corners=True)
                        x = pos[..., 2]
                        y = pos[..., 1]
                        z = pos[..., 0]
                        chi = numpy.arctan2(y, x)
                        corners = numpy.zeros((shape[0], shape[1], 4, 2),
                                              dtype=numpy.float32)
                        if chi.shape[:2] == shape:
                            corners[..., 1] = chi
                        else:
                            corners[:shape[0], :shape[1], :, 1] = chi[:shape[0], :shape[1], :]
                        if space is not None:
                            rad = unit.equation(x, y, z, self._wavelength)
                            if rad.shape[:2] == shape:
                                corners[..., 0] = rad
                            else:
                                corners[:shape[0], :shape[1], :, 0] = rad[:shape[0], :shape[1], :]
                    self._corner4Da = corners
                    self._corner4Ds = space
        return self._corner4Da

    @deprecated
    def cornerArray(self, shape=None):
        """Generate a 4D array of the given shape with (i,j) (radial
        angle 2th, azimuthal angle chi ) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are:
           * dim3[0]: radial angle 2th
           * dim3[1]: azimuthal angle chi
        """
        return self.corner_array(shape, unit=units.TTH_RAD)

    @deprecated
    def cornerQArray(self, shape=None):
        """
        Generate a 3D array of the given shape with (i,j) (azimuthal
        angle) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are (scattering vector q, azimuthal angle chi)
        """
        return self.corner_array(shape, unit=units.Q, use_cython=False)

    @deprecated
    def cornerRArray(self, shape=None):
        """
        Generate a 3D array of the given shape with (i,j) (azimuthal
        angle) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are (radial distance, azimuthal angle chi)
        """
        return self.corner_array(shape, unit=units.R, use_cython=False)

    @deprecated
    def cornerRd2Array(self, shape=None):
        """
        Generate a 3D array of the given shape with (i,j) (azimuthal
        angle) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are (reciprocal spacing squared, azimuthal angle chi)
        """
        return self.corner_array(shape, unit=units.RecD2_NM)

    def center_array(self, shape=None, unit="2th"):
        """
        Generate a 2D array of the given shape with (i,j) (radial
        angle ) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are:
           * dim3[0]: radial angle 2th, q, r, ...
           * dim3[1]: azimuthal angle chi
        """
        space_name_map = {  # space -> array name
                           # "2th_center": "_ttha",
                           # "chi_center":"_chia",
                           # "q_center":"_qa",
                           # "r_center": "_ra",
                           # "d*2_center": "_rd2a"
                           }

        unit = units.to_unit(unit)
        space = unit.REPR.split("_")[0] + "_center"
        ary = None
        if (space in space_name_map):
            ary = self.__getattribute__(space_name_map[space])
        elif space in self._cached_array:
            ary = self._cached_array[space]

        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if (ary is not None) and (ary.shape == shape):
            return ary

        pos = self.positionArray(shape, corners=False)
        x = pos[..., 2]
        y = pos[..., 1]
        z = pos[..., 0]
        ary = unit.equation(x, y, z, self.wavelength)

        if (space in space_name_map):
            self.__setattr__(space_name_map[space], ary)
        else:
            self._cached_array[space] = ary

        return ary

    def delta_array(self, shape=None, unit="2th"):
        """
        Generate a 2D array of the given shape with (i,j) (delta-radial
        angle) for all elements.

        @param shape: expected shape
        @type shape: 2-tuple of integer
        @return: 3d array with shape=(*shape,4,2) the two elements are:
           * dim3[0]: radial angle 2th, q, r, ...
           * dim3[1]: azimuthal angle chi
        """

        unit = units.to_unit(unit)
        space = unit.REPR.split("_")[0] + "_delta"
        ary = self._cached_array.get(space)

        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if (ary is not None) and (ary.shape == shape):
            return ary
        center = self.center_array(shape, unit=unit)
        corners = self.corner_array(shape, unit=unit)
        delta = abs(corners[..., 0] - numpy.atleast_3d(center))
        ary = delta.max(axis=-1)
        self._cached_array[space] = ary
        return ary

    def delta2Theta(self, shape=None):
        """
        Generate a 3D array of the given shape with (i,j) with the max
        distance between the center and any corner in 2 theta

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2D-array containing the max delta angle between a pixel center and any corner in 2theta-angle (rad)
        """

        if self._cached_array.get("2th_delta") is None:
            center = self.twoThetaArray(shape)
            corners = self.corner_array(shape, unit=units.TTH)
            with self._sem:
                if self._cached_array.get("2th_delta") is None:
                    delta = abs(corners[..., 0] - numpy.atleast_3d(center))
                    self._cached_array["2th_delta"] = delta.max(axis=-1)
        return self._cached_array["2th_delta"]

    def deltaChi(self, shape=None):
        """
        Generate a 3D array of the given shape with (i,j) with the max
        distance between the center and any corner in chi-angle (rad)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2D-array  containing the max delta angle between a pixel center and any corner in chi-angle (rad)
        """
        if self._cached_array.get("chi_delta") is None:
            center = numpy.atleast_3d(self.chiArray(shape))
            corner = self.corner_array(shape, None)
            with self._sem:
                if self._cached_array.get("chi_delta") is None:
                    twoPi = 2.0 * numpy.pi
                    delta = numpy.minimum(((corner[:, :, :, 1] - center) % twoPi),
                                          ((center - corner[:, :, :, 1]) % twoPi))
                    self._cached_array["chi_delta"] = delta.max(axis=-1)
        return self._cached_array["chi_delta"]

    def deltaQ(self, shape=None):
        """
        Generate a 2D array of the given shape with (i,j) with the max
        distance between the center and any corner in q_vector unit
        (nm^-1)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: array 2D containing the max delta Q between a pixel center and any corner in q_vector unit (nm^-1)
        """
        if self._cached_array.get("q_delta") is None:
            center = self.qArray(shape)
            corners = self.corner_array(shape, unit=units.Q)
            with self._sem:
                if self._cached_array.get("q_delta") is None:
                    delta = abs(corners[..., 0] - numpy.atleast_3d(center))
                    self._cached_array["q_delta"] = delta.max(axis=-1)
        return self._cached_array["q_delta"]

    def deltaR(self, shape=None):
        """
        Generate a 2D array of the given shape with (i,j) with the max
        distance between the center and any corner in radius unit (mm)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: array 2D containing the max delta Q between a pixel center and any corner in q_vector unit (nm^-1)
        """
        if self._cached_array.get("r_delta") is None:
            center = self.rArray(shape)
            corners = self.corner_array(shape, unit=units.R)
            with self._sem:
                if self._cached_array.get("r_delta") is None:
                    delta = abs(corners[..., 0] - numpy.atleast_3d(center))
                    self._cached_array["r_delta"] = delta.max(axis=-1)
        return self._cached_array["r_delta"]

    def deltaRd2(self, shape=None):
        """
        Generate a 2D array of the given shape with (i,j) with the max
        distance between the center and any corner in unit: reciprocal spacing squarred (1/nm^2)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: array 2D containing the max delta (d*)^2 between a pixel center and any corner in reciprocal spacing squarred (1/nm^2)
        """
        if self._cached_array.get("d*2_delta") is None:
            center = self.center_array(shape, unit=units.RecD2_NM)
            corners = self.corner_array(shape, unit=units.RecD2_NM)
            with self._sem:
                if self._cached_array.get("d*2_delta") is None:
                    delta = abs(corners[..., 0] - numpy.atleast_3d(center))
                    self._cached_array["d*2_delta"] = delta.max(axis=-1)
        return self._cached_array.get("d*2_delta")

    def array_from_unit(self, shape=None, typ="center", unit=units.TTH):
        """
        Generate an array of position in different dimentions (R, Q,
        2Theta)

        @param shape: shape of the expected array
        @type shape: ndarray.shape
        @param typ: "center", "corner" or "delta"
        @type typ: str
        @param unit: can be Q, TTH, R for now
        @type unit: pyFAI.units.Enum

        @return: R, Q or 2Theta array depending on unit
        @rtype: ndarray
        """
        shape = self.get_shape(shape)
        if shape is None:
            logger.error("Shape is neither specified in the method call, "
                         "neither in the detector: %s", self.detector)

        if not typ in ("center", "corner", "delta"):
            logger.warning("Unknown type of array %s,"
                           " defaulting to 'center'" % typ)
            typ = "center"
        unit = units.to_unit(unit)
        meth_name = unit.get(typ)
        if meth_name and meth_name in dir(Geometry):
            # fast path may be available
            out = Geometry.__dict__[meth_name](self, shape)
        else:
            # fast path is definitely not available, use the generic formula
            if typ == "center":
                out = self.center_array(shape, unit)
            elif typ == "corner":
                out = self.corner_array(shape, unit)
            else:  # typ == "delta":
                out = self.delta_array(shape, unit)
        return out

    def cosIncidance(self, d1, d2, path="cython"):
        """
        Calculate the incidence angle (alpha) for current pixels (P).
        The poni being the point of normal incidence,
        it's incidence angle is $\{alpha} = 0$ hence $cos(\{alpha}) = 1$

        @param d1: 1d or 2d set of points in pixel coord
        @param d2:  1d or 2d set of points in pixel coord
        @return: cosine of the incidence angle
        """
        p1, p2, p3 = self._calc_cartesian_positions(d1, d2)
        if p3 is not None:
            # case for non-planar detector ...

            # Calculate the sample-pixel vector (center of pixel) and norm it
            z, y, x = self.calc_pos_zyx(None, d1, d2)
            t = numpy.zeros((z.size, 3))
            for i, v in enumerate((z, y, x)):
                t[..., i] = v.ravel()
            length = numpy.sqrt((t * t).sum(axis=-1))
            length.shape = (length.size, 1)
            length.strides = (length.strides[0], 0)
            t /= length
            # extract the 4 corners of each pixel and calculate the cross product of the diagonal to get the normal
            z, y, x = self.calc_pos_zyx(None, d1, d2, corners=True)
            corners = numpy.zeros(z.shape + (3,))
            for i, v in enumerate((z, y, x)):
                corners[..., i] = v
            A = corners[..., 0, :]
            B = corners[..., 1, :]
            C = corners[..., 2, :]
            D = corners[..., 3, :]
            A.shape = -1, 3
            B.shape = -1, 3
            C.shape = -1, 3
            D.shape = -1, 3
            orth = numpy.cross(C - A, D - B)
            # normalize the normal vector
            length = numpy.sqrt((orth * orth).sum(axis=-1))
            length.shape = length.shape + (1,)
            length.strides = length.strides[:-1] + (0,)
            orth /= length
            # calculate the cosine of the vector using the dot product
            return (t * orth).sum(axis=-1).reshape(d1.shape)
        if path == "cython":
            cosa = _geometry.calc_cosa(self._dist, p1, p2)
        else:
            cosa = self._dist / numpy.sqrt(self._dist * self._dist + p1 * p1 + p2 * p2)
        return cosa

    def diffSolidAngle(self, d1, d2):
        """
        Calculate the solid angle of the current pixels (P) versus the PONI (C)

                  Omega(P)    A cos(a)     SC^2         3       SC^3
        dOmega = --------- = --------- x --------- = cos (a) = ------
                  Omega(C)    SP^2        A cos(0)              SP^3

        cos(a) = SC/SP

        @param d1: 1d or 2d set of points
        @param d2: 1d or 2d set of points (same size&shape as d1)
        @return: solid angle correction array
        """
        ds = 1.0

        # #######################################################
        # Nota: the solid angle correction should be done in flat
        # field correction. Here is dual-correction
        # #######################################################

        if self.spline and self._correct_solid_angle_for_spline:
            max1 = d1.max() + 1
            max2 = d2.max() + 1
            sX = self.spline.splineFuncX(numpy.arange(max2 + 1),
                                         numpy.arange(max1) + 0.5)
            sY = self.spline.splineFuncY(numpy.arange(max2) + 0.5,
                                         numpy.arange(max1 + 1))
            dX = sX[:, 1:] - sX[:, :-1]
            dY = sY[1:, :] - sY[:-1, :]
            ds = (dX + 1.0) * (dY + 1.0)

        if self._cosa is None:
            self._cosa = self.cosIncidance(d1, d2)
        dsa = ds * self._cosa ** self._dssa_order

        return dsa

    def solidAngleArray(self, shape=None, order=3, absolute=False):
        """Generate an array for the solid angle correction
        given the shape of the detector.

        solid_angle = cos(incidence)^3

        @param shape: shape of the array expected
        @param order: should be 3, power of the formula just obove
        @param absolute: the absolute solid angle is calculated as:

        SA = pix1*pix2/dist^2 * cos(incidence)^3

        """
        if self._dssa is None:
            if order is True:
                self._dssa_order = 3.0
            else:
                self._dssa_order = float(order)
            self._dssa = numpy.fromfunction(self.diffSolidAngle,
                                            shape, dtype=numpy.float32)
            self._dssa_crc = crc32(self._dssa)
        if absolute:
            return self._dssa * self.pixel1 * self.pixel2 / self._dist / self._dist
        else:
            return self._dssa

    def save(self, filename):
        """
        Save the geometry parameters.

        @param filename: name of the file where to save the parameters
        @type filename: string
        """
        try:
            with open(filename, "a") as f:
                f.write(("# Nota: C-Order, 1 refers to the Y axis,"
                         " 2 to the X axis \n"))
                f.write("# Calibration done at %s\n" % time.ctime())
                detector = self.detector
                if isinstance(detector, detectors.NexusDetector) and detector._filename:
                    f.write("Detector: %s\n" % os.path.abspath(detector._filename))
                elif detector.name != "Detector":
                    f.write("Detector: %s\n" % detector.__class__.__name__)
                f.write("PixelSize1: %s\n" % detector.pixel1)
                f.write("PixelSize2: %s\n" % detector.pixel2)
                if detector.splineFile:
                    f.write("SplineFile: %s\n" % detector.splineFile)

                f.write("Distance: %s\n" % self._dist)
                f.write("Poni1: %s\n" % self._poni1)
                f.write("Poni2: %s\n" % self._poni2)
                f.write("Rot1: %s\n" % self._rot1)
                f.write("Rot2: %s\n" % self._rot2)
                f.write("Rot3: %s\n" % self._rot3)
                if self._wavelength is not None:
                    f.write("Wavelength: %s\n" % self._wavelength)
        except IOError:
            logger.error("IOError while writing to file %s" % filename)
    write = save

    @classmethod
    def sload(cls, filename):
        """
        A static method combining the constructor and the loader from
        a file

        @param filename: name of the file to load
        @type filename: string
        @return: instance of Gerometry of AzimuthalIntegrator set-up with the parameter from the file.
        """
        inst = cls()
        inst.load(filename)
        return inst

    def load(self, filename):
        """
        Load the refined parameters from a file.

        @param filename: name of the file to load
        @type filename: string
        """
        data = {}
        for line in open(filename):
            if line.startswith("#") or (":" not in line):
                continue
            words = line.split(":", 1)

            key = words[0].strip().lower()
            try:
                value = words[1].strip()
            except Exception as error:  # IGNORE:W0703:
                logger.error("Error %s with line: %s" % (error, line))
            data[key] = value
        if "detector" in data:
            self.detector = detectors.detector_factory(data["detector"])
        else:
            self.detector = detectors.Detector()
        if self.detector.force_pixel and ("pixelsize1" in data) and ("pixelsize2" in data):
            pixel1 = float(data["pixelsize1"])
            pixel2 = float(data["pixelsize2"])
            self.detector = self.detector.__class__(pixel1=pixel1, pixel2=pixel2)
        else:
            if "pixelsize1" in data:
                self.detector.pixel1 = float(data["pixelsize1"])
            if "pixelsize2" in data:
                self.detector.pixel2 = float(data["pixelsize2"])
        if "distance" in data:
            self._dist = float(data["distance"])
        if "poni1" in data:
            self._poni1 = float(data["poni1"])
        if "poni2" in data:
            self._poni2 = float(data["poni2"])
        if "rot1" in data:
            self._rot1 = float(data["rot1"])
        if "rot2" in data:
            self._rot2 = float(data["rot2"])
        if "rot3" in data:
            self._rot3 = float(data["rot3"])
        if "wavelength" in data:
            self._wavelength = float(data["wavelength"])
        if "splinefile" in data:
            if data["splinefile"].lower() != "none":
                self.detector.set_splineFile(data["splinefile"])
        self.reset()
    read = load

    def getPyFAI(self):
        """
        Export geometry setup with the geometry of PyFAI

        @return: dict with the parameter-set of the PyFAI geometry
        """
        with self._sem:
            out = self.detector.getPyFAI()
            out["dist"] = self._dist
            out["poni1"] = self._poni1
            out["poni2"] = self._poni2
            out["rot1"] = self._rot1
            out["rot2"] = self._rot2
            out["rot3"] = self._rot3
            if self._wavelength:
                out["wavelength"] = self._wavelength
        return out

    def setPyFAI(self, **kwargs):
        """
        set the geometry from a pyFAI-like dict
        """
        with self._sem:
            if "detector" in kwargs:
                self.detector = detectors.detector_factory(kwargs["detector"])
            else:
                self.detector = detectors.Detector()
            for key in ["dist", "poni1", "poni2",
                        "rot1", "rot2", "rot3",
                        "pixel1", "pixel2", "splineFile", "wavelength"]:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            self.param = [self._dist, self._poni1, self._poni2,
                          self._rot1, self._rot2, self._rot3]
            self.chiDiscAtPi = True  # position of the discontinuity of chi in radians, pi by default
            self.reset()
#            self._wavelength = None
            self._oversampling = None
            if self.splineFile:
                self.detector.set_splineFile(self.splineFile)
        return self

    def getFit2D(self):
        """
        Export geometry setup with the geometry of Fit2D

        @return: dict with parameters compatible with fit2D geometry
        """
        with self._sem:
            cosTilt = cos(self._rot1) * cos(self._rot2)
            sinTilt = sqrt(1 - cosTilt * cosTilt)
            # This is tilt plane rotation
            if sinTilt != 0:
                cosTpr = max(-1, min(1, -cos(self._rot2) * sin(self._rot1) / sinTilt))
                sinTpr = sin(self._rot2) / sinTilt
            else:  # undefined, does not seem to matter as not tilted
                cosTpr = 1.0
                sinTpr = 0.0
            directDist = 1.0e3 * self._dist / cosTilt
            tilt = degrees(arccos(cosTilt))
            if sinTpr < 0:
                tpr = -degrees(arccos(cosTpr))
            else:
                tpr = degrees(arccos(cosTpr))

            centerX = (self._poni2 + self._dist * sinTilt / cosTilt * cosTpr)\
                / self.pixel2
            if abs(tilt) < 1e-5:
                centerY = (self._poni1) / self.pixel1
            else:
                centerY = (self._poni1 + self._dist * sinTilt / cosTilt * sinTpr) / self.pixel1
            out = self.detector.getFit2D()
            out["directDist"] = directDist
            out["centerX"] = centerX
            out["centerY"] = centerY
            out["tilt"] = tilt
            out["tiltPlanRotation"] = tpr
        return out

    def setFit2D(self, directDist, centerX, centerY,
                 tilt=0., tiltPlanRotation=0.,
                 pixelX=None, pixelY=None, splineFile=None):
        """
        Set the Fit2D-like parameter set: For geometry description see
        HPR 1996 (14) pp-240

        Warning: Fit2D flips automatically images depending on their file-format.
        By reverse engineering we noticed this behavour for Tiff and Mar345 images (at least).
        To obtaine correct result you will have to flip images using numpy.flipud.

        @param direct: direct distance from sample to detector along the incident beam (in millimeter as in fit2d)
        @param tilt: tilt in degrees
        @param tiltPlanRotation: Rotation (in degrees) of the tilt plan arround the Z-detector axis
                * 0deg -> Y does not move, +X goes to Z<0
                * 90deg -> X does not move, +Y goes to Z<0
                * 180deg -> Y does not move, +X goes to Z>0
                * 270deg -> X does not move, +Y goes to Z>0

        @param pixelX,pixelY: as in fit2d they ar given in micron, not in meter
        @param centerX, centerY: pixel position of the beam center
        @param splineFile: name of the file containing the spline
        """
        with self._sem:
            try:
                cosTilt = cos(radians(tilt))
                sinTilt = sin(radians(tilt))
                cosTpr = cos(radians(tiltPlanRotation))
                sinTpr = sin(radians(tiltPlanRotation))
            except AttributeError as error:
                logger.error(("Got strange results with tilt=%s"
                              " and tiltPlanRotation=%s: %s") %
                             (tilt, tiltPlanRotation, error))
            if splineFile is None:
                if pixelX is not None:
                    self.detector.pixel1 = pixelY * 1.0e-6
                if pixelY is not None:
                    self.detector.pixel2 = pixelX * 1.0e-6
            else:
                self.detector.set_splineFile(splineFile)
            self._dist = directDist * cosTilt * 1.0e-3
            self._poni1 = centerY * self.pixel1\
                - directDist * sinTilt * sinTpr * 1.0e-3
            self._poni2 = centerX * self.pixel2\
                - directDist * sinTilt * cosTpr * 1.0e-3
            rot2 = numpy.arcsin(sinTilt * sinTpr)  # or pi-
            rot1 = numpy.arccos(min(1.0, max(-1.0, (cosTilt / numpy.sqrt(1 - sinTpr * sinTpr * sinTilt * sinTilt)))))  # + or -
            if cosTpr * sinTilt > 0:
                rot1 = -rot1
            assert abs(cosTilt - cos(rot1) * cos(rot2)) < 1e-6
            if tilt == 0:
                rot3 = 0
            else:
                rot3 = numpy.arccos(min(1.0, max(-1.0, (cosTilt * cosTpr * sinTpr - cosTpr * sinTpr) / numpy.sqrt(1 - sinTpr * sinTpr * sinTilt * sinTilt))))  # + or -
                rot3 = numpy.pi / 2.0 - rot3
            self._rot1 = rot1
            self._rot2 = rot2
            self._rot3 = rot3
            self.reset()
            return self

    def setSPD(self, SampleDistance, Center_1, Center_2, Rot_1=0, Rot_2=0, Rot_3=0,
               PSize_1=None, PSize_2=None, splineFile=None, BSize_1=1, BSize_2=1,
               WaveLength=None):
        """
        Set the SPD like parameter set: For geometry description see
        Peter Boesecke J.Appl.Cryst.(2007).40, s423–s427

        Basically the main difference with pyFAI is the order of the axis which are flipped

        @param SampleDistance: distance from sample to detector at the PONI (orthogonal projection)
        @param Center_1, pixel position of the PONI along fastest axis
        @param Center_2: pixel position of the PONI along slowest axis
        @param Rot_1: rotation around the fastest axis (x)
        @param Rot_2: rotation around the slowest axis (y)
        @param Rot_3: rotation around the axis ORTHOGONAL to the detector plan
        @param PSize_1: pixel size in meter along the fastest dimention
        @param PSize_2: pixel size in meter along the slowst dimention
        @param splineFile: name of the file containing the spline
        @param BSize_1: pixel binning factor along the fastest dimention
        @param BSize_2: pixel binning factor along the slowst dimention
        @param WaveLength: wavelength used
        """
        # first define the detector
        if splineFile:
            # let's assume the spline file is for unbinned detectors ...
            self.detector = detectors.FReLoN(splineFile)
            self.detector.binning = (int(BSize_2), int(BSize_1))
        elif PSize_1 and PSize_2:
            self.detector = detectors.Detector(PSize_2, PSize_1)
            if BSize_2 > 1 or BSize_1 > 1:
                # set binning factor without changing pixel size
                self.detector._binning = (int(BSize_2), int(BSize_1))

        # then the geometry
        self._dist = float(SampleDistance)
        self._poni1 = float(Center_2) * self.detector.pixel1
        self._poni2 = float(Center_1) * self.detector.pixel2
        # This is WRONG ... correct it
        self._rot1 = Rot_2 or 0
        self._rot2 = Rot_1 or 0
        self._rot3 = -(Rot_3 or 0)
        if Rot_1 or Rot_2 or Rot_3:
            # TODO: one-day
            raise NotImplementedError("rotation axis not yet implemented for SPD")
        # and finally the wavelength
        if WaveLength:
            self.wavelength = float(WaveLength)
        self.reset()
        return self

    def getSPD(self):
        """
        get the SPD like parameter set: For geometry description see
        Peter Boesecke J.Appl.Cryst.(2007).40, s423–s427

        Basically the main difference with pyFAI is the order of the axis which are flipped

        @return: dictionnary with those parameters:
            SampleDistance: distance from sample to detector at the PONI (orthogonal projection)
            Center_1, pixel position of the PONI along fastest axis
            Center_2: pixel position of the PONI along slowest axis
            Rot_1: rotation around the fastest axis (x)
            Rot_2: rotation around the slowest axis (y)
            Rot_3: rotation around the axis ORTHOGONAL to the detector plan
            PSize_1: pixel size in meter along the fastest dimention
            PSize_2: pixel size in meter along the slowst dimention
            splineFile: name of the file containing the spline
            BSize_1: pixel binning factor along the fastest dimention
            BSize_2: pixel binning factor along the slowst dimention
            WaveLength: wavelength used in meter
        """
        res = {"PSize_1": self.detector.pixel2,
               "PSize_2": self.detector.pixel1,
               "BSize_1": self.detector.binning[1],
               "BSize_2": self.detector.binning[0],
               "splineFile": self.detector.splineFile,
               "Rot_3": None,
               "Rot_2": None,
               "Rot_1": None,
               "Center_2": self._poni1 / self.detector.pixel1,
               "Center_1": self._poni2 / self.detector.pixel2,
               "SampleDistance": self.dist
               }
        if self._wavelength:
            res["WaveLength"] = self._wavelength
        if abs(self.rot1) > 1e-6 or abs(self.rot2) > 1e-6 or abs(self.rot3) > 1e-6:
            logger.warning("Rotation conversion from pyFAI to SPD is not yet implemented")
        return res

    def setChiDiscAtZero(self):
        """
        Set the position of the discontinuity of the chi axis between
        0 and 2pi.  By default it is between pi and -pi
        """
        with self._sem:
            self.chiDiscAtPi = False
            self._cached_array["chi_center"] = None
            self._corner4Da = None
            self._corner4Ds = None
#             self._corner4Dqa = None
#             self._corner4Dra = None
#             self._corner4Drd2a = None

    def setChiDiscAtPi(self):
        """
        Set the position of the discontinuity of the chi axis between
        -pi and +pi.  This is the default behavour
        """
        with self._sem:
            self.chiDiscAtPi = True
            self._cached_array["chi_center"] = None
            self._corner4Da = None
            self._corner4Ds = None
#             self._corner4Dqa = None
#             self._corner4Dra = None
#             self._corner4Drd2a = None

    @deprecated
    def setOversampling(self, iOversampling):
        """
        set the oversampling factor
        """
        if self._oversampling is None:
            lastOversampling = 1.0
        else:
            lastOversampling = float(self._oversampling)

        self._oversampling = iOversampling
        self._cached_arrays["2th_center"] = None
        self._cached_arrays["q_center"] = None
        self._dssa = None
        self._cached_array["chi_center"] = None

        self.pixel1 /= self._oversampling / lastOversampling
        self.pixel2 /= self._oversampling / lastOversampling

    def oversampleArray(self, myarray):
        origShape = myarray.shape
        origType = myarray.dtype
        new = numpy.zeros((origShape[0] * self._oversampling,
                           origShape[1] * self._oversampling)).astype(origType)
        for i in range(self._oversampling):
            for j in range(self._oversampling):
                new[i::self._oversampling, j::self._oversampling] = myarray
        return new

    def polarization(self, shape=None, factor=None, axis_offset=0):
        """
        Calculate the polarization correction accoding to the
        polarization factor:

        * If the polarization factor is None, the correction is not applied (returns 1)
        * If the polarization factor is 0 (circular polarization), the correction correspond to (1+(cos2θ)^2)/2
        * If the polarization factor is 1 (linear horizontal polarization), there is no correction in the vertical plane  and a node at 2th=90, chi=0
        * If the polarization factor is -1 (linear vertical polarization), there is no correction in the horizontal plane and a node at 2th=90, chi=90
        * If the polarization is elliptical, the polarization factor varies between -1 and +1.

        The axis_offset parameter allows correction for the misalignement of the polarization plane (or ellipse main axis) and the the detector's X axis.

        @param factor: (Ih-Iv)/(Ih+Iv): varies between 0 (no polarization) and 1 (where division by 0 could occure at 2th=90, chi=0)
        @param axis_offset: Angle between the polarization main axis and detector X direction (in radians !!!)
        @return: 2D array with polarization correction array (intensity/polarisation)

        """
        shape = self.get_shape(shape)
        if shape is None:
            raise RuntimeError(("You should provide a shape if the"
                                " geometry is not yet initiallized"))

        if factor is None:
            return numpy.ones(shape, dtype=numpy.float32)
        else:
            factor = float(factor)

        if self._polarization is not None:
            with self._sem:
                if ((factor == self._polarization_factor)
                   and (shape == self._polarization.shape)
                   and (axis_offset == self._polarization_axis_offset)):
                    return self._polarization

        tth = self.twoThetaArray(shape)
        chi = self.chiArray(shape) + axis_offset
        with self._sem:
                cos2_tth = numpy.cos(tth) ** 2
                self._polarization = ((1 + cos2_tth - factor * numpy.cos(2 * chi) * (1 - cos2_tth)) / 2.0)  # .astype(numpy.float32)
                self._polarization_factor = factor
                self._polarization_axis_offset = axis_offset
                self._polarization_crc = crc32(self._polarization)
                return self._polarization

    def calc_transmission(self, t0, shape=None):
        """
        Defines the absorption correction for a phosphor screen or a scintillator
        from t0, the normal transmission of the screen.

        Icor = Iobs(1-t0)/(1-exp(ln(t0)/cos(incidence)))
                 1-exp(ln(t0)/cos(incidence)
        let t = -----------------------------
                          1 - t0
        See reference on:
        J. Appl. Cryst. (2002). 35, 356–359 G. Wu et al.  CCD phosphor

        @param t0: value of the normal transmission (from 0 to 1)
        @param shape: shape of the array
        @return: actual
        """
        shape = self.get_shape(shape)
        if t0 < 0 or t0 > 1:
            logger.error("Impossible value for normal transmission: %s" % t0)
            return

        with self._sem:
            if (t0 == self._transmission_normal) \
                and (shape is None
                     or (shape == self._transmission_corr.shape)):
                return self._transmission_corr

            if shape is None:
                raise RuntimeError(("You should provide a shape if the"
                                    " geometry is not yet initiallized"))

        with self._sem:
            self._transmission_normal = t0
            if self._cosa is None:
                self._cosa = numpy.fromfunction(self.cosIncidance, shape, dtype=numpy.float32)
            self._transmission_corr = (1.0 - numpy.exp(numpy.log(t0) / self._cosa)) / (1 - t0)
            self._transmission_crc = crc32(self._transmission_corr)
        return self._transmission_corr

    def reset(self):
        """
        reset most arrays that are cached: used when a parameter
        changes.
        """
        self.param = [self._dist, self._poni1, self._poni2,
                      self._rot1, self._rot2, self._rot3]
        self._dssa = None
        self._corner4Da = None
        self._corner4Ds = None
        self._polarization = None
        self._polarization_factor = None
        self._transmission_normal = None
        self._transmission_corr = None
        self._transmission_crc = None
        self._cosa = None
        self._cached_array = {}

    def calcfrom1d(self, tth, I, shape=None, mask=None,
                   dim1_unit=units.TTH, correctSolidAngle=True,
                   dummy=0.0,
                   polarization_factor=None, dark=None, flat=None,
                   ):
        """
        Computes a 2D image from a 1D integrated profile

        @param tth: 1D array with radial unit
        @param I: scattering intensity
        @param shape: shape of the image (if not defined by the detector)
        @param dim1_unit: unit for the "tth" array
        @param correctSolidAngle:
        @param dummy: value for masked pixels
        @param polarization_factor: set to true to use previously used value
        @param dark: dark current correction
        @param flat: flatfield corrction
        @return: 2D image reconstructed

        """
        dim1_unit = units.to_unit(dim1_unit)
        tth = tth.copy() / dim1_unit.scale

        if shape is None:
            shape = self.detector.max_shape
        try:
            ttha = self.__getattribute__(dim1_unit.center)(shape)
        except:
            raise RuntimeError("in pyFAI.Geometry.calcfrom1d: " +
                               str(dim1_unit) + " not (yet?) Implemented")
        calcimage = numpy.interp(ttha.ravel(), tth, I)
        calcimage.shape = shape
        if correctSolidAngle:
            calcimage *= self.solidAngleArray(shape)
        if polarization_factor is not None:
            if (polarization_factor is True) and (self._polarization is not None):
                polarization = self._polarization
            else:
                polarization = self.polarization(shape, polarization_factor,
                                                 axis_offset=0)
            assert polarization.shape == tuple(shape)
            calcimage *= polarization
        if flat is not None:
            assert dark.shape == tuple(shape)
            calcimage *= flat
        if dark is not None:
            assert dark.shape == tuple(shape)
            calcimage += dark
        if mask is not None:
            assert mask.shape == tuple(shape)
            calcimage[numpy.where(mask)] = dummy
        return calcimage

    def __copy__(self):
        """@return a shallow copy of itself.
        """
        new = self.__class__(detector=self.detector)
        # transfer numerical values:
        numerical = ["_dist", "_poni1", "_poni2", "_rot1", "_rot2", "_rot3",
                     "chiDiscAtPi", "_dssa_crc", "_dssa_order", "_wavelength",
                     '_oversampling', '_correct_solid_angle_for_spline',
                     '_polarization_factor', '_polarization_axis_offset',
                     '_polarization_crc', '_transmission_crc', '_transmission_normal',
                     "_corner4Ds"]
        array = [ "_dssa",
                 "_corner4Da",
                 '_polarization', '_cosa', '_transmission_normal', '_transmission_corr']
        for key in numerical + array:
            new.__setattr__(key, self.__getattribute__(key))
        new.param = [new._dist, new._poni1, new._poni2,
                     new._rot1, new._rot2, new._rot3]
        new._cached_array = self._cached_array.copy()
        return new

    def __deepcopy__(self, memo=None):
        """deep copy
        @param memo: dict with modified objects
        @return: a deep copy of itself."""
        numerical = ["_dist", "_poni1", "_poni2", "_rot1", "_rot2", "_rot3",
                     "chiDiscAtPi", "_dssa_crc", "_dssa_order", "_wavelength",
                     '_oversampling', '_correct_solid_angle_for_spline',
                     '_polarization_factor', '_polarization_axis_offset',
                     '_polarization_crc', '_transmission_crc', '_transmission_normal',
                     "_corner4Ds"]
        array = [ "_dssa",
                 "_corner4Da",
                 '_polarization', '_cosa', '_transmission_normal', '_transmission_corr']
        if memo is None:
            memo = {}
        new = self.__class__()
        memo[id(self)] = new
        new_det = self.detector.__deepcopy__(memo)
        new.detector = new_det

        for key in numerical:
            old_value = self.__getattribute__(key)
            memo[id(old_value)] = old_value
            new.__setattr__(key, old_value)
        for key in array:
            value = self.__getattribute__(key)
            if value is not None:
                new.__setattr__(key, 1 * value)
            else:
                new.__setattr__(key, None)
        new_param = [new._dist, new._poni1, new._poni2,
                     new._rot1, new._rot2, new._rot3]
        memo[id(self.param)] = new_param
        new.param = new_param
        cached = {}
        memo[id(self._cached_array)] = cached
        for key, old_value in self._cached_array.items():
            if "copy" in dir(old_value):
                new_value = old_value.copy()
                memo[id(old_value)] = new_value
        new._cached_array = cached
        return new

# ############################################
# Accessors and public properties of the class
# ############################################
    def get_shape(self, shape=None):
        """Guess what is the best shape ....
        @param shape: force this value (2-tuple of int)
        @return: 2-tuple of int
        """
        if shape is None:
            shape = self.detector.shape
        if shape is None:
            for ary in self._cached_array.values():
                if ary is not None:
                    shape = ary.shape[:2]
                    break
        return shape

    def set_dist(self, value):
        if isinstance(value, float):
            self._dist = value
        else:
            self._dist = float(value)
        self.reset()

    def get_dist(self):
        return self._dist

    dist = property(get_dist, set_dist)

    def set_poni1(self, value):
        if isinstance(value, float):
            self._poni1 = value
        elif isinstance(value, (tuple, list)):
            self._poni1 = float(value[0])
        else:
            self._poni1 = float(value)
        self.reset()

    def get_poni1(self):
        return self._poni1

    poni1 = property(get_poni1, set_poni1)

    def set_poni2(self, value):
        if isinstance(value, float):
            self._poni2 = value
        elif isinstance(value, (tuple, list)):
            self._poni2 = float(value[0])
        else:
            self._poni2 = float(value)
        self.reset()

    def get_poni2(self):
        return self._poni2

    poni2 = property(get_poni2, set_poni2)

    def set_rot1(self, value):
        if isinstance(value, float):
            self._rot1 = value
        elif isinstance(value, (tuple, list)):
            self._rot1 = float(value[0])
        else:
            self._rot1 = float(value)
        self.reset()

    def get_rot1(self):
        return self._rot1

    rot1 = property(get_rot1, set_rot1)

    def set_rot2(self, value):
        if isinstance(value, float):
            self._rot2 = value
        elif isinstance(value, (tuple, list)):
            self._rot2 = float(value[0])
        else:
            self._rot2 = float(value)
        self.reset()

    def get_rot2(self):
        return self._rot2

    rot2 = property(get_rot2, set_rot2)

    def set_rot3(self, value):
        if isinstance(value, float):
            self._rot3 = value
        elif isinstance(value, (tuple, list)):
            self._rot3 = float(value[0])
        else:
            self._rot3 = float(value)
        self.reset()

    def get_rot3(self):
        return self._rot3

    rot3 = property(get_rot3, set_rot3)

    def set_wavelength(self, value):
        old_wl = self._wavelength
        if isinstance(value, float):
            self._wavelength = value
        elif isinstance(value, (tuple, list)):
            self._wavelength = float(value[0])
        else:
            self._wavelength = float(value)
        qa = dqa = corner4Da = corner4Ds = None
        if old_wl and self._wavelength:
            if self._cached_array.get("q_center") is not None:
                qa = self._cached_array["q_center"] * old_wl / self._wavelength
            if self._corner4Ds and ("d" in self._corner4Ds or "q" in self._corner4Ds):
                corner4Da = self._corner4Da.copy()
                corner4Da[..., 0] = self._corner4Da[..., 0] * old_wl / self._wavelength
                corner4Ds = self._corner4Ds
        self.reset()
        # restore updated values
        self._cached_array["q_delta"] = dqa
        self._cached_array["q_center"] = qa
        self._corner4Da = corner4Da
        self._corner4Ds = corner4Ds

    def get_wavelength(self):
        if self._wavelength is None:
            raise RuntimeWarning("Using wavelength without having defined"
                                 " it previously ... excepted to fail !")
        return self._wavelength

    wavelength = property(get_wavelength, set_wavelength)

    def get_ttha(self):
        return self._cached_array.get("2th_center")

    def set_ttha(self, _):
        logger.error("You are not allowed to modify 2theta array")

    def del_ttha(self):
        self._cached_array["2th_center"] = None
    ttha = property(get_ttha, set_ttha, del_ttha, "2theta array in cache")

    def get_chia(self):
        return self._cached_array.get("chi_center")

    def set_chia(self, _):
        logger.error("You are not allowed to modify chi array")

    def del_chia(self):
        self._cached_array["chi_center"] = None
    chia = property(get_chia, set_chia, del_chia, "chi array in cache")

    def get_dssa(self):
        return self._dssa

    def set_dssa(self, _):
        logger.error("You are not allowed to modify solid angle array")

    def del_dssa(self):
        self._dssa = None
    dssa = property(get_dssa, set_dssa, del_dssa, "solid angle array in cache")

    def get_qa(self):
        return self._cached_array.get("q_center")

    def set_qa(self, _):
        logger.error("You are not allowed to modify Q array")

    def del_qa(self):
        self._cached_array["q_center"] = None
    qa = property(get_qa, set_qa, del_qa, "Q array in cache")

    def get_ra(self):
        return self._cached_array.get("r_center")

    def set_ra(self, _):
        logger.error("You are not allowed to modify R array")

    def del_ra(self):
        self.self._cached_array["r_center"] = None
    ra = property(get_ra, set_ra, del_ra, "R array in cache")

    def get_pixel1(self):
        return self.detector.pixel1

    def set_pixel1(self, pixel1):
        self.detector.pixel1 = pixel1

    pixel1 = property(get_pixel1, set_pixel1)

    def get_pixel2(self):
        return self.detector.pixel2

    def set_pixel2(self, pixel2):
        self.detector.pixel2 = pixel2

    pixel2 = property(get_pixel2, set_pixel2)

    def get_splineFile(self):
        return self.detector.splineFile

    def set_splineFile(self, splineFile):
        self.detector.splineFile = splineFile

    splineFile = property(get_splineFile, set_splineFile)

    def get_spline(self):
        return self.detector.spline

    def set_spline(self, spline):
        self.detector.spline = spline

    spline = property(get_spline, set_spline)

    def get_correct_solid_angle_for_spline(self):
        return self._correct_solid_angle_for_spline

    def set_correct_solid_angle_for_spline(self, value):
        v = bool(value)
        with self._sem:
            if v != self._correct_solid_angle_for_spline:
                self._dssa = None
                self._correct_solid_angle_for_spline = v

    correct_SA_spline = property(get_correct_solid_angle_for_spline,
                                 set_correct_solid_angle_for_spline)

    def set_maskfile(self, maskfile):
        self.detector.set_maskfile(maskfile)

    def get_maskfile(self):
        return self.detector.get_maskfile()

    maskfile = property(get_maskfile, set_maskfile)

    def set_mask(self, mask):
        self.detector.set_mask(mask)

    def get_mask(self):
        return self.detector.get_mask()

    mask = property(get_mask, set_mask)
