# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3, binding=True
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping
## cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""
This extension is a fast-implementation for calculating the geometry, i.e. where
every pixel of an array stays in space (x,y,z) or its (r, :math:`\\chi`)
coordinates.
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "07/01/2025"
__copyright__ = "2011-2020, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

include "math_common.pxi"

cimport cython
import os
import numpy
from cython.parallel cimport prange
from libc.math cimport sin, cos, atan2, sqrt, M_PI

cdef:
    Py_ssize_t MIN_SIZE = 1024  # Minimum size of the array to go parallel.
    Py_ssize_t MAX_THREADS = 8  # Limit to 8 cores at maximum, avoids using multiple sockets usually. The value comes from numexpr
    double twopi = 2.0 * M_PI

# We declare a second cython.floating so that it behaves like an actual template
ctypedef fused float_or_double:
    cython.double
    cython.float

try:
    MAX_THREADS = min(MAX_THREADS, len(os.sched_getaffinity(os.getpid()))) # Limit to the actual number of threads
except Exception:
    MAX_THREADS = min(MAX_THREADS, os.cpu_count() or 1)


cdef inline double f_t1(double p1, double p2, double p3,
                        double sinRot1, double cosRot1,
                        double sinRot2, double cosRot2,
                        double sinRot3, double cosRot3,
                        int orientation=0) noexcept nogil:
    """Calculate t1 (aka y) for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param p3: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles
    :param orientation: value 1-4
    """
    cdef double orient = -1.0 if (orientation==1 or orientation==2) else 1.0
    return orient*(p1 * cosRot2 * cosRot3 +
            p2 * (cosRot3 * sinRot1 * sinRot2 - cosRot1 * sinRot3) -
            p3 * (cosRot1 * cosRot3 * sinRot2 + sinRot1 * sinRot3))


cdef inline double f_t2(double p1, double p2, double p3,
                        double sinRot1, double cosRot1,
                        double sinRot2, double cosRot2,
                        double sinRot3, double cosRot3,
                        int orientation=0) noexcept nogil:
    """Calculate t2 (aka x) for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param p3: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles
    :param orientation: value 1-4
    """
    cdef double orient = -1.0 if (orientation==1 or orientation==4) else 1.0
    return orient*(p1 * cosRot2 * sinRot3 +
            p2 * (cosRot1 * cosRot3 + sinRot1 * sinRot2 * sinRot3) -
            p3 * (-(cosRot3 * sinRot1) + cosRot1 * sinRot2 * sinRot3))


cdef inline double f_t3(double p1, double p2, double p3,
                        double sinRot1, double cosRot1,
                        double sinRot2, double cosRot2,
                        double sinRot3, double cosRot3,
                        int orientation=0) noexcept nogil:
    """Calculate t3 (aka -z) for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param p3: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles
    :param orientation: unused
    """

    return p1 * sinRot2 - p2 * cosRot2 * sinRot1 + p3 * cosRot1 * cosRot2


cdef inline double f_tth(double p1, double p2, double L,
                         double sinRot1, double cosRot1,
                         double sinRot2, double cosRot2,
                         double sinRot3, double cosRot3,
                         int orientation=0) noexcept nogil:
    """Calculate 2 theta for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles
    :param orientation: unused
    :return: 2 theta
    """
    cdef:
        double t1 = f_t1(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
        double t2 = f_t2(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
        double t3 = f_t3(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    return atan2(sqrt(t1 * t1 + t2 * t2), t3)


cdef inline double f_q(double p1, double p2, double L,
                       double sinRot1, double cosRot1,
                       double sinRot2, double cosRot2,
                       double sinRot3, double cosRot3,
                       double wavelength, int orientation=0) noexcept nogil:
    """
    Calculate the scattering vector q for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles,
    :param orientation: unused
    """
    return 4.0e-9 * M_PI / wavelength * sin(f_tth(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3) / 2.0)


cdef inline double f_chi(double p1, double p2, double L,
                         double sinRot1, double cosRot1,
                         double sinRot2, double cosRot2,
                         double sinRot3, double cosRot3,
                         int orientation=0) noexcept nogil:
    """
    calculate chi for 1 pixel
    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles,
    :param orientation: value 1-4
    """
    cdef:
        double t1 = f_t1(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
        double t2 = f_t2(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
    return atan2(t1, t2)


cdef inline double f_r(double p1, double p2, double L,
                       double sinRot1, double cosRot1,
                       double sinRot2, double cosRot2,
                       double sinRot3, double cosRot3,
                       int orientation=0) noexcept nogil:
    """
    calculate r for 1 pixel, radius from beam center to current
    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    :param sinRot1,sinRot2,sinRot3: sine of the angles
    :param cosRot1,cosRot2,cosRot3: cosine of the angles
    :param orientation: unused
    """
    cdef:
        double t1 = f_t1(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
        double t2 = f_t2(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
        # double t3 = f_t3(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    return sqrt(t1 * t1 + t2 * t2)
    # Changed 10/03/2016 ... the radius is in the pixel position.
    # return L * sqrt(t1 * t1 + t2 * t2) / (t3 * cosRot1 * cosRot2)


cdef inline double f_cosa(double p1, double p2, double L) noexcept nogil:
    """
    calculate cosine of the incidence angle for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    """
    return L / sqrt((L * L) + ((p1 * p1) + (p2 * p2)))

cdef inline double f_sina(double p1, double p2, double L) noexcept nogil:
    """
    calculate sinus of the incidence angle for 1 pixel

    :param p1:distances in meter along dim1 from PONI
    :param p2: distances in meter along dim2 from PONI
    :param L: distance sample - PONI
    """
    cdef double r2 = p1 * p1 + p2 * p2
    return sqrt(r2 / (L * L + r2))


################################################################################
# End of pure cython function declaration
################################################################################


def calc_pos_zyx(double L, double poni1, double poni2,
                 double rot1, double rot2, double rot3,
                 pos1 not None,
                 pos2 not None,
                 pos3=None,
                 int orientation=0):
    """Calculate the 3D coordinates in the sample's referential

    :param L: distance sample - PONI
    :param poni1: PONI coordinate along y axis
    :param poni2: PONI coordinate along x axis
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param orientation: value 1-4
    :return: 3-tuple of ndarray of double with same shape and size as pos1

    """
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        Py_ssize_t  size = pos1.size, i = 0
        double p1, p2, p3
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] t1 = numpy.empty(size, dtype=numpy.float64)
        double[::1] t2 = numpy.empty(size, dtype=numpy.float64)
        double[::1] t3 = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            p1 = c1[i] - poni1
            p2 = c2[i] - poni2
            t1[i] = f_t1(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t2[i] = f_t2(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t3[i] = f_t3(p1, p2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            p1 = c1[i] - poni1
            p2 = c2[i] - poni2
            p3 = c3[i] + L
            t1[i] = f_t1(p1, p2, p3, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t2[i] = f_t2(p1, p2, p3, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t3[i] = f_t3(p1, p2, p3, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)

    r1 = numpy.asarray(t1)
    r2 = numpy.asarray(t2)
    r3 = numpy.asarray(t3)

    if pos1.ndim == 3:
        return (r3.reshape(pos1.shape[0], pos1.shape[1], pos1.shape[2]),
                r1.reshape(pos1.shape[0], pos1.shape[1], pos1.shape[2]),
                r2.reshape(pos1.shape[0], pos1.shape[1], pos1.shape[2]))

    if pos1.ndim == 2:
        return (r3.reshape(pos1.shape[0], pos1.shape[1]),
                r1.reshape(pos1.shape[0], pos1.shape[1]),
                r2.reshape(pos1.shape[0], pos1.shape[1]))
    else:
        return r3, r1, r2


def calc_tth(double L, double rot1, double rot2, double rot3,
             pos1 not None,
             pos2 not None,
             pos3=None,
             int orientation=0):
    """
    Calculate the 2theta array (radial angle) in parallel

    :param L: distance sample - PONI
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param orientation: unused
    :return: ndarray of double with same shape and size as pos1
    """
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_tth(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_tth(c1[i], c2[i], L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)

    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_chi(double L, double rot1, double rot2, double rot3,
             pos1 not None,
             pos2 not None,
             pos3=None,
             int orientation=0,
             bint chi_discontinuity_at_pi=True):
    """Calculate the chi array (azimuthal angles) using OpenMP

    X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
    X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
    X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
    tan(Chi) =  X2 / X1


    :param L: distance sample - PONI
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param orientation: orientation of the detector, values 1-4
    :param chi_discontinuity_at_pi: set to False to obtain chi in the range [0, 2pi[ instead of [-pi, pi[
    :return: ndarray of double with same shape and size as pos1
    """
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        double chi
        Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_chi(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            chi = f_chi(c1[i], c2[i], L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            if chi_discontinuity_at_pi:
                out[i] = chi
            else:
                out[i] = (chi + twopi) % twopi
    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_q(double L, double rot1, double rot2, double rot3,
           pos1 not None,
           pos2 not None,
           double wavelength, pos3=None,
           int orientation=0):
    """
    Calculate the q (scattering vector) array using OpenMP

    X1 = p1*cos(rot2)*cos(rot3) + p2*(cos(rot3)*sin(rot1)*sin(rot2) - cos(rot1)*sin(rot3)) -  L*(cos(rot1)*cos(rot3)*sin(rot2) + sin(rot1)*sin(rot3))
    X2 = p1*cos(rot2)*sin(rot3) - L*(-(cos(rot3)*sin(rot1)) + cos(rot1)*sin(rot2)*sin(rot3)) +  p2*(cos(rot1)*cos(rot3) + sin(rot1)*sin(rot2)*sin(rot3))
    X3 = -(L*cos(rot1)*cos(rot2)) + p2*cos(rot2)*sin(rot1) - p1*sin(rot2)
    tan(Chi) =  X2 / X1


    :param L: distance sample - PONI
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param wavelength: in meter to get q in nm-1
    :param orientation: unused
    :return: ndarray of double with same shape and size as pos1
    """
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_q(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, wavelength)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_q(c1[i], c2[i], L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, wavelength)

    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_r(double L, double rot1, double rot2, double rot3,
           pos1 not None, pos2 not None,
           pos3=None,
           int orientation=0):
    """
    Calculate the radius array (radial direction) in parallel

    :param L: distance sample - PONI
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param orientation: unused
    :return: ndarray of double with same shape and size as pos1
    """
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_r(c1[i], c2[i], L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_r(c1[i], c2[i], L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)

    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_cosa(double L,
              pos1 not None,
              pos2 not None,
              pos3=None):
    """Calculate the cosine of the incidence angle using OpenMP.
    Used for sensors thickness effect corrections

    :param L: distance sample - PONI
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :return: ndarray of double with same shape and size as pos1
    """
    cdef Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_cosa(c1[i], c2[i], L)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            out[i] = f_cosa(c1[i], c2[i], L + c3[i])

    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_sina(double L,
              pos1 not None,
              pos2 not None,
              pos3=None):
    """Calculate the sine of the incidence angle using OpenMP.
    Used for sensors thickness effect corrections

    :param L: distance sample - PONI
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :return: ndarray of double with same shape and size as pos1
    """
    cdef ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        double[::1] out = numpy.empty(size, dtype=numpy.float64)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static"):
            out[i] = f_sina(c1[i], c2[i], L)
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static"):
            out[i] = f_sina(c1[i], c2[i], L + c3[i])

    if pos1.ndim == 2:
        return numpy.asarray(out).reshape(pos1.shape[0], pos1.shape[1])
    else:
        return numpy.asarray(out)


def calc_rad_azim(double L,
                  double poni1,
                  double poni2,
                  double rot1,
                  double rot2,
                  double rot3,
                  pos1 not None,
                  pos2 not None,
                  pos3=None,
                  space="2th",
                  wavelength=None,
                  int orientation=0,
                  bint chi_discontinuity_at_pi=True
                  ):
    """Calculate the radial & azimutal position for each pixel from pos1, pos2, pos3.

    :param L: distance sample - PONI
    :param poni1: PONI coordinate along y axis
    :param poni2: PONI coordinate along x axis
    :param rot1: angle1
    :param rot2: angle2
    :param rot3: angle3
    :param pos1: numpy array with distances in meter along dim1 from PONI (Y)
    :param pos2: numpy array with distances in meter along dim2 from PONI (X)
    :param pos3: numpy array with distances in meter along Sample->PONI (Z), positive behind the detector
    :param space: can be "2th", "q" or "r" for radial units. Azimuthal units are radians
    :param orientation: values from 1 to 4
    :param chi_discontinuity_at_pi: set to False to obtain chi in the range [0, 2pi[ instead of [-pi, pi[
    :return: ndarray of double with same shape and size as pos1 + (2,),
    :raise: KeyError when space is bad !
            ValueError when wavelength is missing

    """
    cdef Py_ssize_t  size = pos1.size, i = 0
    assert pos2.size == size, "pos2.size == size"
    cdef:
        double sinRot1 = sin(rot1)
        double cosRot1 = cos(rot1)
        double sinRot2 = sin(rot2)
        double cosRot2 = cos(rot2)
        double sinRot3 = sin(rot3)
        double cosRot3 = cos(rot3)
        int cspace = 0
        double[::1] c1 = numpy.ascontiguousarray(pos1.ravel(), dtype=numpy.float64)
        double[::1] c2 = numpy.ascontiguousarray(pos2.ravel(), dtype=numpy.float64)
        double[::1] c3
        float[:, ::1] out = numpy.empty((size, 2), dtype=numpy.float32)
        double t1, t2, t3, chi, fwavelength=0.0

    if space == "2th":
        cspace = 1
    elif space == "q":
        cspace = 2
        if not wavelength:
            raise ValueError("wavelength is needed for q calculation")
        else:
            fwavelength = float(wavelength)
    elif space == "r":
        cspace = 3
    else:
        raise KeyError("Not implemented space %s in cython" % space)

    if pos3 is None:
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            t1 = f_t1(c1[i] - poni1, c2[i] - poni2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t2 = f_t2(c1[i] - poni1, c2[i] - poni2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t3 = f_t3(c1[i] - poni1, c2[i] - poni2, L, sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
            if cspace == 1:
                out[i, 0] = atan2(sqrt(t1 * t1 + t2 * t2), t3)
            elif cspace == 2:
                out[i, 0] = 4.0e-9 * M_PI / fwavelength * sin(atan2(sqrt(t1 * t1 + t2 * t2), t3) / 2.0)
            elif cspace == 3:
                out[i, 0] = sqrt(t1 * t1 + t2 * t2)
            chi = atan2(t1, t2)
            if chi_discontinuity_at_pi:
                out[i, 1] = chi
            else:
                out[i, 1] = (chi + twopi) % twopi
    else:
        assert pos3.size == size, "pos3.size == size"
        c3 = numpy.ascontiguousarray(pos3.ravel(), dtype=numpy.float64)
        for i in prange(size, nogil=True, schedule="static", num_threads=1 if size<MIN_SIZE else MAX_THREADS):
            t1 = f_t1(c1[i] - poni1, c2[i] - poni2, L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t2 = f_t2(c1[i] - poni1, c2[i] - poni2, L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3, orientation)
            t3 = f_t3(c1[i] - poni1, c2[i] - poni2, L + c3[i], sinRot1, cosRot1, sinRot2, cosRot2, sinRot3, cosRot3)
            if cspace == 1:
                out[i, 0] = atan2(sqrt(t1 * t1 + t2 * t2), t3)
            elif cspace == 2:
                out[i, 0] = 4.0e-9 * M_PI / fwavelength * sin(atan2(sqrt(t1 * t1 + t2 * t2), t3) / 2.0)
            elif cspace == 3:
                out[i, 0] = sqrt(t1 * t1 + t2 * t2)
            chi = atan2(t1, t2)
            if chi_discontinuity_at_pi:
                out[i, 1] = chi
            else:
                out[i, 1] = (chi + twopi) % twopi

    nout = numpy.asarray(out)
    if pos1.ndim == 3:
        return nout.reshape(pos1.shape[0], pos1.shape[1], pos1.shape[2], 2)
    if pos1.ndim == 2:
        return nout.reshape(pos1.shape[0], pos1.shape[1], 2)
    else:
        return nout


def calc_delta_chi(floating[:, ::1] centers,
                   float_or_double[:, :, :, ::1] corners):
    """Calculate the delta chi array (azimuthal angles) using OpenMP

    :param centers: numpy array with chi angles of the center of the pixels
    :param corners: numpy array with chi angles of the corners of the pixels
    :return: ndarray of double with same shape and size as centers with the delta chi per pixel
    """
    cdef:
        Py_ssize_t width, height, row, col, corn, nbcorn
        double co, ce, delta0, delta1, delta2, delta
        double[:, ::1] res

    height = centers.shape[0]
    width = centers.shape[1]
    assert corners.shape[0] == height, "height match"
    assert corners.shape[1] == width, "width match"
    nbcorn = corners.shape[2]

    res = numpy.empty((height, width), dtype=numpy.float64)
    with nogil:
        for row in prange(height, num_threads=MAX_THREADS):
            for col in range(width):
                ce = centers[row, col]
                delta = 0.0
                for corn in range(nbcorn):
                    co = corners[row, col, corn, 1]
                    delta1 = (co - ce + twopi) % twopi
                    delta2 = (ce - co + twopi) % twopi
                    delta0 = min(delta1, delta2)
                    if delta0 > delta:
                        delta = delta0
                res[row, col] = delta
    return numpy.asarray(res)
