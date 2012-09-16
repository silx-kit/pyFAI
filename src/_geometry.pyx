#!/usr/bin/env python
# -*- coding: utf8 -*-
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


import cython
cimport numpy
import numpy
from cython.parallel cimport prange
from libc.math cimport sin, cos, atan2, sqrt, M_PI

@cython.cdivision(True)
cdef double tth(double p1, double p2, double L, double sinRot1, double cosRot1, double sinRot2, double cosRot2, double sinRot3, double cosRot3) nogil:
    """
    calculate 2 theta for 1 pixel
    @param p1:distances in meter along dim1 from PONI
    @param p2: distances in meter along dim2 from PONI
    @param sinRot1,sinRot2,sinRot3: sine of the angles
    @param cosRot1,cosRot2,cosRot3: cosine of the angles
    """
    cdef double t1=0, t2=0, t3=0
    t1 = p1 * cosRot2 * cosRot3 + p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) - L * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3)
    t2 = p1 * cosRot2 * sinRot3 + p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3) - L * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3)
    t3 = (p1 * sinRot2 - p2 * cosRot2 * sinRot1 + L * cosRot1 * cosRot2)
    return  atan2(sqrt(t1 ** 2 + t2 ** 2), t3)

@cython.cdivision(True)
cdef double q(double p1, double p2, double L, double sinRot1, double cosRot1, double sinRot2, double cosRot2, double sinRot3, double cosRot3, double wavelength) nogil:
    """
    calculate the scattering vector q for 1 pixel
    @param p1:distances in meter along dim1 from PONI
    @param p2: distances in meter along dim2 from PONI
    @param sinRot1,sinRot2,sinRot3: sine of the angles
    @param cosRot1,cosRot2,cosRot3: cosine of the angles
    """
    return 4.0e-9 * M_PI / wavelength * sin(tth(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)/2.0)


@cython.cdivision(True)
cdef double chi(double p1, double p2, double L, double sinRot1, double cosRot1, double sinRot2, double cosRot2, double sinRot3, double cosRot3) nogil:
    """
    calculate chi for 1 pixel
    @param p1:distances in meter along dim1 from PONI
    @param p2: distances in meter along dim2 from PONI
    @param sinRot1,sinRot2,sinRot3: sine of the angles
    @param cosRot1,cosRot2,cosRot3: cosine of the angles
    """
    cdef double num=0, den=0
    num = p1 * cosRot2 * cosRot3 + p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) - L * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3)
    den = p1 * cosRot2 * sinRot3 - L * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3) + p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3)
    return  atan2(num, den)

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_tth(double L, double rot1, double rot2, double rot3,
             numpy.ndarray pos1 not None, numpy.ndarray pos2 not None):
    """
    Calculate the 2theta array (radial angle) in parallel

    @param L: distance sample - PONI
    @param rot1: angle1
    @param rot2: angle2
    @param rot3: angle3
    @param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    @param pos2: numpy array with distances in meter along dim2 from PONI (X)
    """
    cdef double sinRot1 = sin(rot1)
    cdef double cosRot1 = cos(rot1)
    cdef double sinRot2 = sin(rot2)
    cdef double cosRot2 = cos(rot2)
    cdef double sinRot3 = sin(rot3)
    cdef double cosRot3 = cos(rot3)
    cdef size_t  size = pos1.size, i=0
    assert pos2.size == size
    cdef double[:] c1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float64)
    cdef double[:] c2 = numpy.ascontiguousarray(pos2.ravel(),dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out = numpy.empty(size, dtype=numpy.float64)
    for i in prange(size, nogil=True, schedule="static"):
        out[i] = tth(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_chi(double L, double rot1, double rot2, double rot3,
             numpy.ndarray pos1 not None, numpy.ndarray pos2 not None):
    """
    Calculate the chi array (azimuthal angles) in parallel

    X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
    X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
    X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
    tan(Chi) =  X2 / X1


    @param L: distance sample - PONI
    @param rot1: angle1
    @param rot2: angle2
    @param rot3: angle3
    @param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    @param pos2: numpy array with distances in meter along dim2 from PONI (X)
    """
    cdef double sinRot1 = sin(rot1)
    cdef double cosRot1 = cos(rot1)
    cdef double sinRot2 = sin(rot2)
    cdef double cosRot2 = cos(rot2)
    cdef double sinRot3 = sin(rot3)
    cdef double cosRot3 = cos(rot3)
    cdef size_t  size = pos1.size, i=0
    assert pos2.size == size
    cdef double[:] c1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float64)
    cdef double[:] c2 = numpy.ascontiguousarray(pos2.ravel(),dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out = numpy.empty(size, dtype=numpy.float64)
    for i in prange(size, nogil=True, schedule="static"):
        out[i] = chi(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_q(double L, double rot1, double rot2, double rot3,
             numpy.ndarray pos1 not None, numpy.ndarray pos2 not None, double wavelength):
    """
    Calculate the q (scattering vector) array in parallel

    X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
    X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
    X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
    tan(Chi) =  X2 / X1


    @param L: distance sample - PONI
    @param rot1: angle1
    @param rot2: angle2
    @param rot3: angle3
    @param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    @param pos2: numpy array with distances in meter along dim2 from PONI (X)
    @param wavelength: in meter to get q in nm-1
    """
    cdef double sinRot1 = sin(rot1)
    cdef double cosRot1 = cos(rot1)
    cdef double sinRot2 = sin(rot2)
    cdef double cosRot2 = cos(rot2)
    cdef double sinRot3 = sin(rot3)
    cdef double cosRot3 = cos(rot3)
    cdef size_t  size = pos1.size, i=0
    assert pos2.size == size
    cdef double[:] c1 = numpy.ascontiguousarray(pos1.ravel(),dtype=numpy.float64)
    cdef double[:] c2 = numpy.ascontiguousarray(pos2.ravel(),dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim = 1] out = numpy.empty(size, dtype=numpy.float64)
    for i in prange(size, nogil=True, schedule="static"):
        out[i] = q(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, wavelength)
    return out
