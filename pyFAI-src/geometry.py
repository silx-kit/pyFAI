# !/usr/bin/env python
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "20/04/2013"
__status__ = "beta"
__docformat__ = 'restructuredtext'

import os
import time
import threading
import logging
import types
import numpy

from numpy import radians, degrees, arccos, arctan2, sin, cos, sqrt
from pyFAI import detectors

try:
    from pyFAI import _geometry
except ImportError:
    _geometry = None
logger = logging.getLogger("pyFAI.geometry")
try:
    from pyFAI.fastcrc import crc32
except ImportError:
    from zlib import crc32
from . import units


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
        self._ttha = None
        self._dttha = None
        self._dssa = None
        self._dssa_crc = None  # checksum associated with _dssa
        self._dssa_order = 3  # by default we correct for 1/cos(2th), fit2d corrects for 1/cos^3(2th)
        self._chia = None
        self._dchia = None
        self._qa = None
        self._dqa = None
        self._ra = None
        self._dra = None
        self._corner4Da = None
        self._corner4Dqa = None
        self._corner4Dra = None
        self._wavelength = wavelength
        self._oversampling = None
        self._correct_solid_angle_for_spline = True
        self._sem = threading.Semaphore()
        self._polarization_factor = 0
        self._polarization_axis_offset = 0
        self._polarization = None
        self._polarization_crc = None  # checksum associated with _polarization
        self._cosa = None      #cosine of the incidance angle
        self._transmission_normal = None
        self._transmission_corr = None
        self._transmission_crc = None

        if detector:
            if type(detector) in types.StringTypes:
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

    def __repr__(self):
        self.param = [self._dist, self._poni1, self._poni2,
                      self._rot1, self._rot2, self._rot3]
        lstTxt = [self.detector.__repr__()]
        if self._wavelength:
            lstTxt.append("Wavelength= %.6em" % self._wavelength)
        lstTxt.append(("SampleDetDist= %.6em\tPONI= %.6e, %.6em\trot1=%.6f"
                       "  rot2= %.6f  rot3= %.6f rad") % tuple(self.param))
        if self.detector.pixel1:
            f2d = self.getFit2D()
            lstTxt.append(("DirectBeamDist= %.3fmm\tCenter: x=%.3f, y=%.3f pix"
                           "\tTilt=%.3f deg  tiltPlanRotation= %.3f deg") %
                          (f2d["directDist"], f2d["centerX"], f2d["centerY"],
                           f2d["tilt"], f2d["tiltPlanRotation"]))
        return os.linesep.join(lstTxt)

    def _calcCartesianPositions(self, d1, d2, poni1=None, poni2=None):
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

        p1, p2 = self.detector.calc_cartesian_positions(d1, d2)
        return p1 - poni1, p2 - poni2

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
        @rtype: floar or array of floats.
        """
        if param is None:
            param = self.param
        p1, p2 = self._calcCartesianPositions(d1, d2, param[1], param[2])

        if path == "cython" and _geometry:
            tmp = _geometry.calc_tth(L=param[0],
                                     rot1=param[3],
                                     rot2=param[4],
                                     rot3=param[5],
                                     pos1=p1,
                                     pos2=p2)
            tmp.shape = p1.shape
        else:
            L = param[0]
            cosRot1 = cos(param[3])
            cosRot2 = cos(param[4])
            cosRot3 = cos(param[5])
            sinRot1 = sin(param[3])
            sinRot2 = sin(param[4])
            sinRot3 = sin(param[5])
            t1 = p1 * cosRot2 * cosRot3 + \
                p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) - \
                L * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3)
            t2 = p1 * cosRot2 * sinRot3 + \
                p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3) - \
                L * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3)
            t3 = p1 * sinRot2 - p2 * cosRot2 * sinRot1 + L * cosRot1 * cosRot2
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
            p1, p2 = self._calcCartesianPositions(d1, d2,
                                                  self._poni1, self.poni2)
            out = _geometry.calc_q(L=self._dist,
                                   rot1=self._rot1,
                                   rot2=self._rot2,
                                   rot3=self._rot3,
                                   pos1=p1,
                                   pos2=p2,
                                   wavelength=self.wavelength)
            out.shape = p1.shape
        else:
            out = 4.0e-9 * numpy.pi / self.wavelength * \
                numpy.sin(self.tth(d1=d1, d2=d2, param=param) / 2.0)
        return out

    def rFunction(self, d1, d2, param=None, path="numpy"):
        """
        Calculates the radius value for the center of a given pixel
        (or set of pixels) in mm

        r = direct_distance  * tan( 2theta )

        @param d1: position(s) in pixel in first dimension (c order)
        @type d1: scalar or array of scalar
        @param d2: position(s) in pixel in second dimension (c order)
        @type d2: scalar or array of scalar
        @return: r in in mm
        @rtype: float or array of floats.
        """
        cosTilt = cos(self._rot1) * cos(self._rot2)
        directDist = self._dist / cosTilt  # in m

        if _geometry and path == "cython":
            p1, p2 = self._calcCartesianPositions(d1, d2, self._poni1, self.poni2)
            out = _geometry.calc_r(L=self._dist,
                                   rot1=self._rot1,
                                   rot2=self._rot2,
                                   rot3=self._rot3,
                                   pos1=p1,
                                   pos2=p2)
            out.shape = p1.shape
        else:
            out = directDist * numpy.tan(self.tth(d1=d1, d2=d2, param=param))
        return out

    def qArray(self, shape):
        """
        Generate an array of the given shape with q(i,j) for all
        elements.
        """
        if self._qa is None:
            with self._sem:
                if self._qa is None:
                    self._qa = numpy.fromfunction(self.qFunction, shape,
                                                  dtype=numpy.float32)
        return self._qa

    def rArray(self, shape):
        """
        Generate an array of the given shape with r(i,j) for all
        elements; r in mm.

        @param shape: expected shape
        @return: 2d array of the given shape with radius in mm from beam stop.
        """
        if self._ra is None:
            with self._sem:
                if self._ra is None:
                    self._ra = numpy.fromfunction(self.rFunction, shape,
                                                  dtype=numpy.float32)
        return self._ra

    def qCornerFunct(self, d1, d2):
        """
        Calculate the q_vector for any pixel corner (in nm^-1)
        """
        return self.qFunction(d1 - 0.5, d2 - 0.5)

    def rCornerFunct(self, d1, d2):
        """
        Calculate the radius array for any pixel corner (in mm)
        """
        return self.rFunction(d1 - 0.5, d2 - 0.5)

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

    def twoThetaArray(self, shape):
        """
        Generate an array of the given shape with two-theta(i,j) for
        all elements.
        """
        if self._ttha is None:
            with self._sem:
                if self._ttha is None:
                    self._ttha = numpy.fromfunction(self.tth,
                                                    shape,
                                                    dtype=numpy.float32)
        return self._ttha

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
        p1, p2 = self._calcCartesianPositions(d1, d2, self._poni1, self.poni2)

        if path == "cython" and _geometry:
            tmp = _geometry.calc_chi(
                L=self._dist,
                rot1=self._rot1, rot2=self._rot2, rot3=self._rot3,
                pos1=p1, pos2=p2)
            tmp.shape = p1.shape
        else:
            cosRot1 = cos(self._rot1)
            cosRot2 = cos(self._rot2)
            cosRot3 = cos(self._rot3)
            sinRot1 = sin(self._rot1)
            sinRot2 = sin(self._rot2)
            sinRot3 = sin(self._rot3)
            L = self._dist
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

    def chiArray(self, shape):
        """
        Generate an array of the given shape with chi(i,j) (azimuthal
        angle) for all elements.

        @param shape: the shape of the chi array
        @type shape: ndarray.shape
        @return: the chi array
        @rtype: ndarray
        """
        if self._chia is None:
            self._chia = numpy.fromfunction(self.chi, shape,
                                            dtype=numpy.float32)
            if not self.chiDiscAtPi:
                self._chia = self._chia % (2 * numpy.pi)
        return self._chia

    def cornerArray(self, shape):
        """
        Generate a 3D array of the given shape with (i,j) (radial
        angle 2th, azimuthal angle chi ) for all elements.

        @param shape: expected shape
        @type shape: ndarray.shape
        @return: 3d array with shape=(*shape,2) the two elements are (radial angle 2th, azimuthal angle chi)
        """
        if self._corner4Da is None:
            with self._sem:
                if self._corner4Da is None:
                    self._corner4Da = numpy.zeros((shape[0], shape[1], 4, 2),
                                                  dtype=numpy.float32)
                    chi = numpy.fromfunction(self.chi_corner,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    tth = numpy.fromfunction(self.tth_corner,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    self._corner4Da[:, :, 0, 0] = tth[:-1, :-1]
                    self._corner4Da[:, :, 0, 1] = chi[:-1, :-1]
                    self._corner4Da[:, :, 1, 0] = tth[1:, :-1]
                    self._corner4Da[:, :, 1, 1] = chi[1:, :-1]
                    self._corner4Da[:, :, 2, 0] = tth[1:, 1:]
                    self._corner4Da[:, :, 2, 1] = chi[1:, 1:]
                    self._corner4Da[:, :, 3, 0] = tth[:-1, 1:]
                    self._corner4Da[:, :, 3, 1] = chi[:-1, 1:]
        return self._corner4Da

    def cornerQArray(self, shape):
        """
        Generate a 3D array of the given shape with (i,j) (azimuthal
        angle) for all elements.
        """
        if self._corner4Dqa is None:
            with self._sem:
                if self._corner4Dqa is None:
                    self._corner4Dqa = numpy.zeros((shape[0], shape[1], 4, 2),
                                                   dtype=numpy.float32)
                    qar = numpy.fromfunction(self.qCornerFunct,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    chi = numpy.fromfunction(self.chi_corner,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    self._corner4Dqa[:, :, 0, 0] = qar[:-1, :-1]
                    self._corner4Dqa[:, :, 0, 1] = chi[:-1, :-1]
                    self._corner4Dqa[:, :, 1, 0] = qar[1:, :-1]
                    self._corner4Dqa[:, :, 1, 1] = chi[1:, :-1]
                    self._corner4Dqa[:, :, 2, 0] = qar[1:, 1:]
                    self._corner4Dqa[:, :, 2, 1] = chi[1:, 1:]
                    self._corner4Dqa[:, :, 3, 0] = qar[:-1, 1:]
                    self._corner4Dqa[:, :, 3, 1] = chi[:-1, 1:]
        return self._corner4Dqa

    def cornerRArray(self, shape):
        """
        Generate a 3D array of the given shape with (i,j) (azimuthal
        angle) for all elements.
        """
        if self._corner4Dra is None:
            with self._sem:
                if self._corner4Dra is None:
                    self._corner4Dra = numpy.zeros((shape[0], shape[1], 4, 2),
                                                   dtype=numpy.float32)
                    rar = numpy.fromfunction(self.rCornerFunct,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    chi = numpy.fromfunction(self.chi_corner,
                                             (shape[0] + 1, shape[1] + 1),
                                             dtype=numpy.float32)
                    self._corner4Dra[:, :, 0, 0] = rar[:-1, :-1]
                    self._corner4Dra[:, :, 0, 1] = chi[:-1, :-1]
                    self._corner4Dra[:, :, 1, 0] = rar[1:, :-1]
                    self._corner4Dra[:, :, 1, 1] = chi[1:, :-1]
                    self._corner4Dra[:, :, 2, 0] = rar[1:, 1:]
                    self._corner4Dra[:, :, 2, 1] = chi[1:, 1:]
                    self._corner4Dra[:, :, 3, 0] = rar[:-1, 1:]
                    self._corner4Dra[:, :, 3, 1] = chi[:-1, 1:]
        return self._corner4Dra

    def delta2Theta(self, shape):
        """
        Generate a 3D array of the given shape with (i,j) with the max
        distance between the center and any corner in 2 theta

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2D-array containing the max delta angle between a pixel center and any corner in 2theta-angle (rad)
        """
        tth_center = self.twoThetaArray(shape)
        if self._dttha is None:
            with self._sem:
                if self._dttha is None:
                    delta = numpy.zeros([shape[0], shape[1], 4],
                                        dtype=numpy.float32)
                    if self._corner4Da is not None\
                            and self._corner4Da.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                abs(self._corner4Da[:, :, i, 0] - tth_center)
                    else:
                        tth_corner = numpy.fromfunction(
                            self.tth_corner,
                            (shape[0] + 1, shape[1] + 1), dtype=numpy.float32)
                        delta[:, :, 0] = abs(tth_corner[:-1, :-1] - tth_center)
                        delta[:, :, 1] = abs(tth_corner[1:, :-1] - tth_center)
                        delta[:, :, 2] = abs(tth_corner[1:, 1:] - tth_center)
                        delta[:, :, 3] = abs(tth_corner[:-1, 1:] - tth_center)
                    self._dttha = delta.max(axis=2)
        return self._dttha

    def deltaChi(self, shape):
        """
        Generate a 3D array of the given shape with (i,j) with the max
        distance between the center and any corner in chi-angle (rad)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: 2D-array  containing the max delta angle between a pixel center and any corner in chi-angle (rad)
        """
        chi_center = self.chiArray(shape)
        if self._dchia is None:
            with self._sem:
                if self._dchia is None:
                    twoPi = 2 * numpy.pi
                    delta = numpy.zeros([shape[0], shape[1], 4],
                                        dtype=numpy.float32)
                    if self._corner4Da is not None\
                            and self._corner4Da.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                numpy.minimum(((self._corner4Da[:, :, i, 1] - chi_center) % twoPi),
                                              ((chi_center - self._corner4Da[:, :, i, 1]) % twoPi))
                    if self._corner4Dqa is not None\
                            and self._corner4Dqa.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                numpy.minimum(((self._corner4Dqa[:, :, i, 1] - chi_center) % twoPi),
                                              ((chi_center - self._corner4Dqa[:, :, i, 1]) % twoPi))
                    if self._corner4Dra is not None\
                            and self._corner4Dra.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                numpy.minimum(((self._corner4Dra[:, :, i, 1] - chi_center) % twoPi),
                                              ((chi_center - self._corner4Dra[:, :, i, 1]) % twoPi))
                    else:
                        chi_corner = \
                            numpy.fromfunction(self.chi_corner,
                                               (shape[0] + 1, shape[1] + 1),
                                               dtype=numpy.float32)
                        delta[:, :, 0] = \
                            numpy.minimum(((chi_corner[:-1, :-1] - chi_center) % twoPi),
                                          ((chi_center - chi_corner[:-1, :-1]) % twoPi))
                        delta[:, :, 1] = \
                            numpy.minimum(((chi_corner[1: , :-1] - chi_center) % twoPi),
                                          ((chi_center - chi_corner[1: , :-1]) % twoPi))
                        delta[:, :, 2] = \
                            numpy.minimum(((chi_corner[1: , 1: ] - chi_center) % twoPi),
                                          ((chi_center - chi_corner[1: , 1: ]) % twoPi))
                        delta[:, :, 3] = \
                            numpy.minimum(((chi_corner[:-1, 1: ] - chi_center) % twoPi),
                                          ((chi_center - chi_corner[:-1, 1: ]) % twoPi))
                    self._dchia = delta.max(axis= -1)
        return self._dchia

    def deltaQ(self, shape):
        """
        Generate a 2D array of the given shape with (i,j) with the max
        distance between the center and any corner in q_vector unit
        (nm^-1)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: array 2D containing the max delta Q between a pixel center and any corner in q_vector unit (nm^-1)
        """
        q_center = self.qArray(shape)
        if self._dqa is None:
            with self._sem:
                if self._dqa is None:
                    delta = numpy.zeros([shape[0], shape[1], 4],
                                        dtype=numpy.float32)
                    if self._corner4Dqa is not None\
                            and self._corner4Dqa.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                abs(self._corner4Dqa[:, :, i, 0] - q_center)
                    else:
                        q_corner = \
                            numpy.fromfunction(self.qCornerFunct,
                                               (shape[0] + 1, shape[1] + 1),
                                               dtype=numpy.float32)
                        delta[:, :, 0] = abs(q_corner[:-1, :-1] - q_center)
                        delta[:, :, 1] = abs(q_corner[1:, :-1] - q_center)
                        delta[:, :, 2] = abs(q_corner[1:, 1:] - q_center)
                        delta[:, :, 3] = abs(q_corner[:-1, 1:] - q_center)
                    self._dqa = delta.max(axis= -1)
        return self._dqa

    def deltaR(self, shape):
        """
        Generate a 2D array of the given shape with (i,j) with the max
        distance between the center and any corner in radius unit (mm)

        @param shape: The shape of the detector array: 2-tuple of integer
        @return: array 2D containing the max delta Q between a pixel center and any corner in q_vector unit (nm^-1)
        """
        q_center = self.rArray(shape)
        if self._dra is None:
            with self._sem:
                if self._dra is None:
                    delta = numpy.zeros([shape[0], shape[1], 4],
                                        dtype=numpy.float32)
                    if self._corner4Dra is not None\
                            and self._corner4Dra.shape[:2] == tuple(shape):
                        for i in range(4):
                            delta[:, :, i] = \
                                abs(self._corner4Dra[:, :, i, 0] - q_center)
                    else:
                        q_corner = \
                            numpy.fromfunction(self.rCornerFunct,
                                               (shape[0] + 1, shape[1] + 1),
                                               dtype=numpy.float32)
                        delta[:, :, 0] = abs(q_corner[:-1, :-1] - q_center)
                        delta[:, :, 1] = abs(q_corner[1:, :-1] - q_center)
                        delta[:, :, 2] = abs(q_corner[1:, 1:] - q_center)
                        delta[:, :, 3] = abs(q_corner[:-1, 1:] - q_center)
                    self._dra = delta.max(axis= -1)
        return self._dra

    def cosIncidance(self, d1, d2):
        """
        Calculate the incidence angle (alpha) for current pixels (P). 
        The poni is at incidence angle=1 so cos(alpha) = 1
        
        """
        p1, p2 = self._calcCartesianPositions(d1, d2)
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
        # field correction Here is dual-correction
        # #######################################################

        if self.spline and self._correct_solid_angle_for_spline:
            max1 = d1.max() + 1
            max2 = d2.max() + 1
            sX = self.spline.splineFuncX(numpy.arange(max2 + 1),
                                         numpy.arange(max1) + 0.5)
            sY = self.spline.splineFuncY(numpy.arange(max2) + 0.5,
                                         numpy.arange(max1 + 1))
            dX = sX[:, 1:] - sX[:, :-1]
            dY = sY[1:, : ] - sY[:-1, :]
            ds = (dX + 1.0) * (dY + 1.0)

        if self._cosa is None:
            self._cosa = self.cosIncidance(d1, d2)
        dsa = ds * self._cosa ** self._dssa_order

        return dsa

    def solidAngleArray(self, shape, order=3):
        """
        Generate an array of the given shape with the solid angle of
        the current element two-theta(i,j) for all elements.



        @param shape: shape of the array expected
        @param order:
        """
        if self._dssa is None:
            if order is True:
                self._dssa_order = 3.0
            else:
                self._dssa_order = float(order)
            self._dssa = numpy.fromfunction(self.diffSolidAngle,
                                            shape, dtype=numpy.float32)
            self._dssa_crc = crc32(self._dssa)
        return self._dssa

    def save(self, filename):
        """
        Save the refined parameters.

        @param filename: name of the file where to save the parameters
        @type filename: string
        """
        try:
            with open(filename, "a") as f:
                f.write(("# Nota: C-Order, 1 refers to the Y axis,"
                         " 2 to the X axis %s") % os.linesep)
                f.write("# Calibration done at %s%s" % (time.ctime(), os.linesep))
                if self.detector.name != "Detector":
                    f.write("Detector: %s%s" % (self.detector.__class__.__name__,
                                                os.linesep))
                f.write("PixelSize1: %s%s" % (self.pixel1, os.linesep))
                f.write("PixelSize2: %s%s" % (self.pixel2, os.linesep))
                f.write("Distance: %s%s" % (self._dist, os.linesep))
                f.write("Poni1: %s%s" % (self._poni1, os.linesep))
                f.write("Poni2: %s%s" % (self._poni2, os.linesep))
                f.write("Rot1: %s%s" % (self._rot1, os.linesep))
                f.write("Rot2: %s%s" % (self._rot2, os.linesep))
                f.write("Rot3: %s%s" % (self._rot3, os.linesep))
                f.write("SplineFile: %s%s" % (self.splineFile, os.linesep))
                if self._wavelength is not None:
                    f.write("Wavelength: %s%s" % (self._wavelength,
                                                  os.linesep))
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
                        "pixel1", "pixel2", "splineFile"]:
                if key in kwargs:
                    setattr(self, key, kwargs[key])
            self.param = [self._dist, self._poni1, self._poni2,
                          self._rot1, self._rot2, self._rot3]
            self.chiDiscAtPi = True  # position of the discontinuity of chi in radians, pi by default
            self.reset()
            self._wavelength = None
            self._oversampling = None
            if self.splineFile:
                self.detector.set_splineFile(self.splineFile)

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
                rot3 = numpy.pi / 2 - rot3
            self._rot1 = rot1
            self._rot2 = rot2
            self._rot3 = rot3
            self.reset()

    def setChiDiscAtZero(self):
        """
        Set the position of the discontinuity of the chi axis between
        0 and 2pi.  By default it is between pi and -pi
        """
        self.chiDiscAtPi = False
        self._chia = None
        self._corner4Da = None
        self._corner4Dqa = None
        self._corner4Dra = None

    def setChiDiscAtPi(self):
        """
        Set the position of the discontinuity of the chi axis between
        -pi and +pi.  This is the default behavour
        """
        self.chiDiscAtPi = True
        self._chia = None
        self._corner4Da = None
        self._corner4Dqa = None
        self._corner4Dra = None

    def setOversampling(self, iOversampling):
        """
        set the oversampling factor
        """
        if self._oversampling is None:
            lastOversampling = 1.0
        else:
            lastOversampling = float(self._oversampling)

        self._oversampling = iOversampling
        self._ttha = None
        self._dssa = None
        self._chia = None
        self._qa = None
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
        if shape is None:
            for i in ["_ttha", "_dttha", "_dssa", "_chia", "_dchia", "_qa", "_dqa", "_ra", "_dra"]:
                ary = self.__getattribute__(i)
                if ary is not None:
                    shape = ary.shape
                    break
        if shape is None:
            raise RuntimeError(("You should provide a shape if the"
                                " geometry is not yet initiallized"))


        if factor is None:
            return numpy.ones(shape, dtype=numpy.float32)
        else:
            factor = float(factor)

        if self._polarization is not None:
            with self._sem:
                if (factor == self._polarization_factor) \
                    and (shape == self._polarization.shape)\
                    and (axis_offset == self._polarization_axis_offset):
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
        if t0 < 0 or t0 > 1:
            logger.error("Impossible value for normal transmission: %s" % t0)
            return

        with self._sem:
            if (t0 == self._transmission_normal) \
                and (shape is None \
                     or (shape == self._transmission_corr.shape)):
                return self._transmission_corr

        if self._cosa is None:
            if shape is None:
                for i in ["_ttha", "_dttha", "_dssa", "_chia", "_dchia", "_qa", "_dqa", "_ra", "_dra"]:
                    ary = self.__getattribute__(i)
                    if ary is not None:
                        shape = ary.shape
                        break
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
        self._ttha = None
        self._dttha = None
        self._dssa = None
        self._chia = None
        self._dchia = None
        self._qa = None
        self._dqa = None
        self._ra = None
        self._dra = None
        self._corner4Da = None
        self._corner4Dqa = None
        self._corner4Dra = None
        self._polarization = None
        self._polarization_factor = None
        self._transmission_normal = None
        self._transmission_corr = None
        self._transmission_crc = None
        self._cosa = None


    def calcfrom1d(self, tth, I, shape=None, mask=None,
                   dim1_unit=units.TTH, correctSolidAngle=True):
        """
        Computes a 2D image from a 1D integrated profile

        @param tth: 1D array with 2theta in degrees
        @param I: scattering intensity
        @return: 2D image reconstructed

        """
        dim1_unit = units.to_unit(dim1_unit)
        tth /= dim1_unit.scale
        if dim1_unit == units.TTH:
            if shape is None:
                ttha = self._ttha
                shape = self._ttha.shape
            else:
                ttha = self.twoThetaArray(shape)
        elif dim1_unit == units.Q:
            if shape is None:
                ttha = self._qa
                shape = ttha.shape
            else:
                ttha = self.qArray(shape)
        elif dim1_unit == units.R:
            if shape is None:
                ttha = self._ra
                shape = ttha.shape
            else:
                ttha = self.rArray(shape)
        else:
#            TODO
            raise RuntimeError("in pyFAI.Geometry.calcfrom1d: "
                               "Not (yet?) Implemented")
        calcimage = numpy.interp(ttha.ravel(), tth, I)
        calcimage.shape = shape
        if correctSolidAngle:
            calcimage *= self.solidAngleArray(shape)
        if mask is not None:
            assert mask.shape == tuple(shape)
            calcimage[numpy.where(mask)] = 0
        return calcimage

# ############################################
# Accessors and public properties of the class
# ############################################

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
        if isinstance(value, float):
            self._wavelength = value
        elif isinstance(value, (tuple, list)):
            self._wavelength = float(value[0])
        else:
            self._wavelength = float(value)
        self._qa = None
        self._dqa = None
        self._corner4Dqa = None

    def get_wavelength(self):
        if self._wavelength is None:
            raise RuntimeWarning("Using wavelength without having defined"
                                 " it previously ... excepted to fail !")
        return self._wavelength

    wavelength = property(get_wavelength, set_wavelength)

    def get_ttha(self):
        return self._ttha

    def set_ttha(self, _):
        logger.error("You are not allowed to modify 2theta array")

    def del_ttha(self):
        self._ttha = None

    ttha = property(get_ttha, set_ttha, del_ttha, "2theta array in cache")

    def get_chia(self):
        return self._chia

    def set_chia(self, _):
        logger.error("You are not allowed to modify chi array")

    def del_chia(self):
        self._chia = None

    chia = property(get_chia, set_chia, del_chia, "chi array in cache")

    def get_dssa(self):
        return self._dssa

    def set_dssa(self, _):
        logger.error("You are not allowed to modify solid angle array")

    def del_dssa(self):
        self._dssa = None

    dssa = property(get_dssa, set_dssa, del_dssa, "solid angle array in cache")

    def get_qa(self):
        return self._qa

    def set_qa(self, _):
        logger.error("You are not allowed to modify Q array")

    def del_qa(self):
        self._qa = None

    qa = property(get_qa, set_qa, del_qa, "Q array in cache")

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
