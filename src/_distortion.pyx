#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
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
from cython.parallel import prange
from libc.math cimport floor,ceil, fabs
from libc.string cimport memset
import logging, threading
import types, os, sys, time
logger = logging.getLogger("pyFAI._distortion")
from .detectors import detector_factory
from .utils import timeit
import fabio

cdef struct lut_point:
    numpy.uint32_t idx
    numpy.float32_t coef

cdef inline float min4f(float a, float b, float c, float d) nogil:
    """Calculates the min of 4 float numbers"""
    if (a <= b) and (a <= c) and (a <= d):
        return a
    if (b <= a) and (b <= c) and (b <= d):
        return b
    if (c <= a) and (c <= b) and (c <= d):
        return c
    else:
        return d

cdef inline float max4f(float a, float b, float c, float d) nogil:
    """Calculates the max of 4 float numbers"""
    if (a >= b) and (a >= c) and (a >= d):
        return a
    if (b >= a) and (b >= c) and (b >= d):
        return b
    if (c >= a) and (c >= b) and (c >= d):
        return c
    else:
        return d


cpdef inline float calc_area(float I1, float I2, float slope, float intercept) nogil:
    "Calculate the area between I1 and I2 of a line with a given slope & intercept"
    return 0.5 * (I2 - I1) * (slope * (I2 + I1) + 2 * intercept)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline void integrate(float[:,:] box, float start, float stop, float slope, float intercept) nogil:
    "Integrate in a box a line between start and stop, line defined by its slope & intercept "
    cdef int i, h = 0
    cdef float P,dP, A, AA, dA, sign
    if start < stop:  # positive contribution
        P = ceil(start)
        dP = P - start
        if P > stop:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (stop - start)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            if dP > 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = dP
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) - 1, h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> floor(P)), (<int> floor(stop))):
                A = calc_area(i, i + 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA

                    h = 0
                    dA = 1.0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i , h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = floor(stop)
            dP = stop - P
            if dP > 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)), h] += sign * dA
                        AA -= dA
                        h += 1
    elif    start > stop:  # negative contribution. Nota is start=stop: no contribution
        P = floor(start)
        if stop > P:  # start and stop are in the same unit
            A = calc_area(start, stop, slope, intercept)
            if A != 0:
                AA = fabs(A)
                sign = A / AA
                dA = (start - stop)  # always positive
                h = 0
                while AA > 0:
                    if dA > AA:
                        dA = AA
                        AA = -1
                    box[(<int> floor(start)), h] += sign * dA
                    AA -= dA
                    h += 1
        else:
            dP = P - start
            if dP < 0:
                A = calc_area(start, P, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[(<int> floor(P)) , h] += sign * dA
                        AA -= dA
                        h += 1
            # subsection P1->Pn
            for i in range((<int> start), (<int> ceil(stop)), -1):
                A = calc_area(i, i - 1, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = 1
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        box[i - 1 , h] += sign * dA
                        AA -= dA
                        h += 1
            # Section Pn->B
            P = ceil(stop)
            dP = stop - P
            if dP < 0:
                A = calc_area(P, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    h = 0
                    dA = fabs(dP)
                    while AA > 0:
                        if dA > AA:
                            dA = AA; AA = -1
                        box[(<int> floor(stop)), h] += sign * dA
                        AA -= dA
                        h += 1

cdef class Quad:
    """
    Basic quadrilatere object

                                     |
                                     |
                                     |                       xxxxxA
                                     |      xxxxxxxI'xxxxxxxx     x
                             xxxxxxxxIxxxxxx       |               x
                Bxxxxxxxxxxxx        |             |               x
                x                    |             |               x
                x                    |             |               x
                 x                   |             |                x
                 x                   |             |                x
                 x                   |             |                x
                 x                   |             |                x
                 x                   |             |                x
                  x                  |             |                 x
                  x                  |             |                 x
                  x                  |             |                 x
                  x                 O|             P              A'  x
 -----------------J------------------+--------------------------------L-----------------------
                  x                  |                                 x
                  x                  |                                  x
                  x                  |                                  x
                   x                 |     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxD
                   CxxxxxxxxxxxxxxxxxKxxxxx
                                     |
                                     |
                                     |
                                     |
        """
    cdef float[:,:] box
    cdef float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area
    cdef int offset0, offset1, box_size0, box_size1
    cdef bint has_area, has_slope

    def __cinit__(self, float[:,:] buffer):
        self.box = buffer
        self.A0 = self.A1 = 0
        self.B0 = self.B1 = 0
        self.C0 = self.C1 = 0
        self.D0 = self.D1 = 0
        self.offset0 = self.offset1 = 0
        self.box_size0 = self.box_size1 = 0
        self.pAB = self.pBC = self.pCD = self.pDA = 0
        self.cAB = self.cBC = self.cCD = self.cDA = 0
        self.area = 0
        self.has_area=0
        self.has_slope=0


    def reinit(self, A0, A1, B0, B1, C0, C1, D0, D1):
        self.box[:, :] = 0.0
        self.A0 = A0
        self.A1 = A1
        self.B0 = B0
        self.B1 = B1
        self.C0 = C0
        self.C1 = C1
        self.D0 = D0
        self.D1 = D1
        self.offset0 = (<int> floor(min(self.A0, self.B0, self.C0, self.D0)))
        self.offset1 = (<int> floor(min(self.A1, self.B1, self.C1, self.D1)))
        self.box_size0 = (<int> ceil(max(self.A0, self.B0, self.C0, self.D0))) - self.offset0
        self.box_size1 = (<int> ceil(max(self.A1, self.B1, self.C1, self.D1))) - self.offset1
        self.A0 -= self.offset0
        self.A1 -= self.offset1
        self.B0 -= self.offset0
        self.B1 -= self.offset1
        self.C0 -= self.offset0
        self.C1 -= self.offset1
        self.D0 -= self.offset0
        self.D1 -= self.offset1
        self.pAB = self.pBC = self.pCD = self.pDA = 0
        self.cAB = self.cBC = self.cCD = self.cDA = 0
        self.area = 0
        self.has_area = 0
        self.has_slope = 0

    def __repr__(self):
        res = ["offset %i,%i size %i, %i" % (self.offset0, self.offset1, self.box_size0, self.box_size1), "box:"]
        for i in range(self.box_size0):
            line=""
            for j in range(self.box_size1):
                line+="\t%.3f"%self.box[i,j]
            res.append(line)
        return os.linesep.join(res)

    cpdef float get_box(self, int i, int j):
        return self.box[i,j]
    cpdef int get_offset0(self):
        return self.offset0
    cpdef int  get_offset1(self):
        return self.offset1
    cpdef int  get_box_size0(self):
        return self.box_size0
    cpdef int  get_box_size1(self):
        return self.box_size1

    cpdef init_slope(self):
        if not self.has_slope:
            if self.B0 != self.A0:
                self.pAB = (self.B1 - self.A1) / (self.B0 - self.A0)
                self.cAB = self.A1 - self.pAB * self.A0
            if self.C0 != self.B0:
                self.pBC = (self.C1 - self.B1) / (self.C0 - self.B0)
                self.cBC = self.B1 - self.pBC * self.B0
            if self.D0 != self.C0:
                self.pCD = (self.D1 - self.C1) / (self.D0 - self.C0)
                self.cCD = self.C1 - self.pCD * self.C0
            if self.A0 != self.D0:
                self.pDA = (self.A1 - self.D1) / (self.A0 - self.D0)
                self.cDA = self.D1 - self.pDA * self.D0
            self.has_slope = 1

    cpdef float calc_area_AB(self, float I1,float I2):
        if self.B0 != self.A0:
            return 0.5 * (I2 - I1) * (self.pAB * (I2 + I1) + 2 * self.cAB)
        else:
            return 0.0

    cpdef float calc_area_BC(self, float J1, float J2):
        if self.B0 != self.C0:
            return 0.5 * (J2 - J1) * (self.pBC * (J1 + J2) + 2 * self.cBC)
        else:
            return 0.0

    cpdef float calc_area_CD(self, float K1, float K2):
        if self.C0 != self.D0:
            return 0.5 * (K2 - K1) * (self.pCD * (K2 + K1) + 2 * self.cCD)
        else:
            return 0.0

    cpdef float calc_area_DA(self,float L1,float L2):
        if self.D0 != self.A0:
            return 0.5 * (L2 - L1) * (self.pDA * (L1 + L2) + 2 * self.cDA)
        else:
            return 0.0



    cpdef float calc_area(self):
        if not self.has_area:
            self.area = 0.5*((self.C0 - self.A0)*(self.D1 - self.B1)-(self.C1 - self.A1)*(self.D0 - self.B0))
            self.has_area = 1
        return self.area

    def populate_box(self):
        cdef int i0, i1
        cdef float area,value
        if not self.has_slope:
            self.init_slope()
        integrate(self.box, self.B0, self.A0, self.pAB, self.cAB)
        integrate(self.box, self.A0, self.D0, self.pDA, self.cDA)
        integrate(self.box, self.D0, self.C0, self.pCD, self.cCD)
        integrate(self.box, self.C0, self.B0, self.pBC, self.cBC)
        area = self.calc_area()
        for i0 in range(self.box_size0):
            for i1 in range(self.box_size1):
                value = self.box[i0,i1] / area
                self.box[i0,i1] = value
                if value < 0.0:
                    print self.box
                    self.box[:, :] = 0
                    print "AB"
                    self.integrate(self.B0, self.A0, self.pAB, self.cAB)
                    print self.box
                    self.box[:, :] = 0
                    print "DA"
                    self.integrate(self.A0, self.D0, self.pDA, self.cDA)
                    print self.box
                    self.box[:, :] = 0
                    print "CD"
                    self.integrate(self.D0, self.C0, self.pCD, self.cCD)
                    print self.box
                    self.box[:, :] = 0
                    print "BC"
                    self.integrate(self.C0, self.B0, self.pBC, self.cBC)
                    print self.box
                    print self
                    raise RuntimeError()

    def integrate(self, float start, float stop, float slope, float intercept):
        cdef int i, h = 0
        cdef float P,dP, A, AA, dA, sign
#        print start, stop, calc_area(start, stop)
        if start < stop:  # positive contribution
            P = ceil(start)
            dP = P - start
#            print "Integrate", start, P, stop, calc_area(start, stop)
            if P > stop:  # start and stop are in the same unit
                A = calc_area(start, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    dA = (stop - start)  # always positive
#                    print AA, sign, dA
                    h = 0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        self.box[(<int> floor(start)), h] += sign * dA
                        AA -= dA
                        h += 1
            else:
                if dP > 0:
                    A = calc_area(start, P, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA
                        h = 0
                        dA = dP
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[(<int> floor(P)) - 1, h] += sign * dA
                            AA -= dA
                            h += 1
                # subsection P1->Pn
                for i in range((<int> floor(P)), (<int> floor(stop))):
                    A = calc_area(i, i + 1, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA

                        h = 0
                        dA = 1.0
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[i , h] += sign * dA
                            AA -= dA
                            h += 1
                # Section Pn->B
                P = floor(stop)
                dP = stop - P
                if dP > 0:
                    A = calc_area(P, stop, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA
                        h = 0
                        dA = fabs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[(<int> floor(P)), h] += sign * dA
                            AA -= dA
                            h += 1
        elif    start > stop:  # negative contribution. Nota is start=stop: no contribution
            P = floor(start)
            if stop > P:  # start and stop are in the same unit
                A = calc_area(start, stop, slope, intercept)
                if A != 0:
                    AA = fabs(A)
                    sign = A / AA
                    dA = (start - stop)  # always positive
                    h = 0
                    while AA > 0:
                        if dA > AA:
                            dA = AA
                            AA = -1
                        self.box[(<int> floor(start)), h] += sign * dA
                        AA -= dA
                        h += 1
            else:
                dP = P - start
                if dP < 0:
                    A = calc_area(start, P, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA
                        h = 0
                        dA = fabs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[(<int> floor(P)) , h] += sign * dA
                            AA -= dA
                            h += 1
                # subsection P1->Pn
                for i in range((<int> start), (<int> ceil(stop)), -1):
                    A = calc_area(i, i - 1, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA
                        h = 0
                        dA = 1
                        while AA > 0:
                            if dA > AA:
                                dA = AA
                                AA = -1
                            self.box[i - 1 , h] += sign * dA
                            AA -= dA
                            h += 1
                # Section Pn->B
                P = ceil(stop)
                dP = stop - P
                if dP < 0:
                    A = calc_area(P, stop, slope, intercept)
                    if A != 0:
                        AA = fabs(A)
                        sign = A / AA
                        h = 0
                        dA = fabs(dP)
                        while AA > 0:
                            if dA > AA:
                                dA = AA; AA = -1
                            self.box[(<int> floor(stop)), h] += sign * dA
                            AA -= dA
                            h += 1

class Distortion(object):
    """

    This class applies a distortion correction on an image.

    It is also able to apply an inversion of the correction.

    """
    def __init__(self, detector="detector", shape=None):
        """
        @param detector: detector instance or detector name
        """
        if type(detector) in types.StringTypes:
            self.detector = detector_factory(detector)
        else:  # we assume it is a Detector instance
            self.detector = detector
        if "max_shape" in dir(self.detector):
            self.shape = self.detector.max_shape
        else:
            self.shape = shape
        self.shape = tuple([int(i) for i in self.shape])
        self._sem = threading.Semaphore()
        self.lut_size = None
        self.pos = None
        self.LUT = None
        self.delta0 = self.delta1 = None  # max size of an pixel on a regular grid ...


    def __repr__(self):
        return os.linesep.join(["Distortion correction for detector:",
                                self.detector.__repr__()])
    def calc_pos(self):
        if self.pos is None:
            with self._sem:
                if self.pos is None:
                    pos_corners = numpy.empty((self.shape[0] + 1, self.shape[1] + 1, 2), dtype=numpy.float64)
                    d1 = numpy.outer(numpy.arange(self.shape[0] + 1, dtype=numpy.float64), numpy.ones(self.shape[1] + 1, dtype=numpy.float64)) - 0.5
                    d2 = numpy.outer(numpy.ones(self.shape[0] + 1, dtype=numpy.float64), numpy.arange(self.shape[1] + 1, dtype=numpy.float64)) - 0.5
                    pos_corners[:, :, 0], pos_corners[:, :, 1] = self.detector.calc_cartesian_positions(d1, d2)
                    pos_corners[:, :, 0] /= self.detector.pixel1
                    pos_corners[:, :, 1] /= self.detector.pixel2
                    pos = numpy.empty((self.shape[0], self.shape[1], 4, 2), dtype=numpy.float32)
                    pos[:, :, 0, :] = pos_corners[:-1, :-1]
                    pos[:, :, 1, :] = pos_corners[:-1, 1: ]
                    pos[:, :, 2, :] = pos_corners[1: , 1: ]
                    pos[:, :, 3, :] = pos_corners[1: , :-1]
                    self.pos = pos
                    self.delta0 = int((numpy.ceil(pos_corners[1:, :, 0]) - numpy.floor(pos_corners[:-1, :, 0])).max())
                    self.delta1 = int((numpy.ceil(pos_corners[:, 1:, 1]) - numpy.floor(pos_corners[:, :-1, 1])).max())
        return self.pos

    @timeit
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def calc_LUT_size(self):
        """
        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        We chose to discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity

        We keep self.pos: pos_corners will not be compatible with systems showing non adjacent pixels (like some xpads)

        """
        cdef int i, j, k, l, shape0, shape1
        cdef numpy.ndarray[numpy.float32_t, ndim = 4] pos
        cdef int[:,:] pos0min, pos1min, pos0max, pos1max
        cdef numpy.ndarray[numpy.uint32_t, ndim = 2] lut_size
        if self.lut_size is None:
            if self.pos is None:
                pos = self.calc_pos()
            else:
                pos = self.pos
            with self._sem:
                if self.lut_size is None:
                    shape0, shape1 = self.shape
                    pos0min = numpy.floor(pos[:, :, :, 0].min(axis= -1)).astype(numpy.int32).clip(0, self.shape[0])
                    pos1min = numpy.floor(pos[:, :, :, 1].min(axis= -1)).astype(numpy.int32).clip(0, self.shape[1])
                    pos0max = (numpy.ceil(pos[:, :, :, 0].max(axis= -1)).astype(numpy.int32) + 1).clip(0, self.shape[0])
                    pos1max = (numpy.ceil(pos[:, :, :, 1].max(axis= -1)).astype(numpy.int32) + 1).clip(0, self.shape[1])
                    lut_size = numpy.zeros(self.shape, dtype=numpy.uint32)
                    with nogil:
                        for i in range(shape0):
                            for j in range(shape1):
                                for k in range(pos0min[i, j],pos0max[i, j]):
                                    for l in range(pos1min[i, j],pos1max[i, j]):
                                        lut_size[k,l] += 1
                    self.lut_size = lut_size.max()
        return lut_size

    @timeit
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def calc_LUT(self):
        cdef int i, j, ms, ml, ns, nl, shape0, shape1, delta0, delta1, buffer_size, i0, i1, size
        cdef int offset0, offset1, box_size0, box_size1
        cdef numpy.uint32_t k, idx=0
        cdef float A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area, value
#        cdef numpy.ndarray[numpy.float32_t, ndim = 3]  pos
        cdef float[:,:,:,:] pos
        cdef numpy.ndarray[lut_point, ndim = 3] lut
        cdef numpy.ndarray[numpy.uint32_t, ndim = 2] outMax = numpy.zeros(self.shape, dtype=numpy.uint32)
        cdef  float[:,:] buffer
        #cdef float[:,:,:] pos
        shape0, shape1 = self.shape

        if self.lut_size is None:
            self.calc_LUT_size()
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    pos = self.pos#.reshape(shape0,shape1,4*sizeof(float))
                    lut = numpy.recarray(shape=(self.shape[0] , self.shape[1], self.lut_size), dtype=[("idx", numpy.uint32), ("coef", numpy.float32)])
                    size = self.shape[0]*self.shape[1]*self.lut_size*sizeof(lut_point)
                    memset(&lut[0,0,0], 0, size)
                    logger.info("LUT shape: (%i,%i,%i) %.3f MByte"%(lut.shape[0], lut.shape[1],lut.shape[2],size/1.0e6))
                    buffer = numpy.empty((self.delta0, self.delta1),dtype=numpy.float32)
                    buffer_size = self.delta0 * self.delta1 * sizeof(float)
                    logger.info("Max pixel size: %ix%i; Max source pixel in target: %i"%(buffer.shape[1],buffer.shape[0], self.lut_size))
                    with nogil:
                        # i,j, idx are indexes of the raw image uncorrected
                        for i in range(shape0):
                            for j in range(shape1):
                                #reinit of buffer
                                memset(&buffer[0,0], 0, buffer_size)
                                A0 = pos[i, j, 0, 0]
                                A1 = pos[i, j, 0, 1]
                                B0 = pos[i, j, 1, 0]
                                B1 = pos[i, j, 1, 1]
                                C0 = pos[i, j, 2, 0]
                                C1 = pos[i, j, 2, 1]
                                D0 = pos[i, j, 3, 0]
                                D1 = pos[i, j, 3, 1]
                                offset0 = (<int> floor(min4f(A0, B0, C0, D0)))
                                offset1 = (<int> floor(min4f(A1, B1, C1, D1)))
                                box_size0 = (<int> ceil(max4f(A0, B0, C0, D0))) - offset0
                                box_size1 = (<int> ceil(max4f(A1, B1, C1, D1))) - offset1
                                A0 -= <float> offset0
                                A1 -= <float> offset1
                                B0 -= <float> offset0
                                B1 -= <float> offset1
                                C0 -= <float> offset0
                                C1 -= <float> offset1
                                D0 -= <float> offset0
                                D1 -= <float> offset1
                                if B0 != A0:
                                    pAB = (B1 - A1) / (B0 - A0)
                                    cAB = A1 - pAB * A0
                                else:
                                    pAB = cAB = 0.0
                                if C0 != B0:
                                    pBC = (C1 - B1) / (C0 - B0)
                                    cBC = B1 - pBC * B0
                                else:
                                    pBC = cBC = 0.0
                                if D0 != C0:
                                    pCD = (D1 - C1) / (D0 - C0)
                                    cCD = C1 - pCD * C0
                                else:
                                    pCD = cCD = 0.0
                                if A0 != D0:
                                    pDA = (A1 - D1) / (A0 - D0)
                                    cDA = D1 - pDA * D0
                                else:
                                    pDA = cDA = 0.0
                                integrate(buffer, B0, A0, pAB, cAB)
                                integrate(buffer, A0, D0, pDA, cDA)
                                integrate(buffer, D0, C0, pCD, cCD)
                                integrate(buffer, C0, B0, pBC, cBC)
                                area = 0.5*((C0 - A0)*(D1 - B1)-(C1 - A1)*(D0 - B0))
                                for ms in range(box_size0):
                                    ml = ms + offset0
                                    if ml < 0 or ml >= shape0:
                                        continue
                                    for ns in range(box_size1):
                                        # ms,ns are indexes of the corrected image in short form, ml & nl are the same
                                        nl = ns + offset1
                                        if nl < 0 or nl >= shape1:
                                            continue
                                        value = buffer[ms, ns] / area
                                        if value <= 0:
                                            continue
                                        k = outMax[ml, nl]
                                        lut[ml, nl, k].idx = idx
                                        lut[ml, nl, k].coef = value
                                        outMax[ml, nl] = k + 1
                                idx += 1
                    self.LUT = lut.reshape(self.shape[0] * self.shape[1], self.lut_size)

    @timeit
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def correct(self, image):
        """
        Correct an image based on the look-up table calculated ...

        @param image: 2D-array with the image
        @return: corrected 2D image
        """
        cdef int i,j, lshape0, lshape1, idx, size
        cdef float coef
        cdef lut_point[:,:] LUT
        cdef float[:] lout, lin
#        if self.integrator is None:
        if self.LUT is None:
                with self._sem:
                    if self.LUT is None:
                        self.calc_LUT()
        LUT = self.LUT
        lshape0 = LUT.shape[0]
        lshape1 = LUT.shape[1]
        img_shape = image.shape
        if (img_shape[0]<self.shape[0]) or (img_shape[1]<self.shape[1]):
            new_image = numpy.zeros(self.shape, dtype=numpy.float32)
            new_image[:img_shape[0],:img_shape[1]] = image
            image = new_image
            logger.warning("Patching image as image is %ix%i and spline is %ix%i"%(img_shape[1],img_shape[0],self.shape[1],self.shape[0]))

        out = numpy.zeros(self.shape, dtype=numpy.float32)
        lout = out.ravel()
        lin = numpy.ascontiguousarray(image.ravel(),dtype=numpy.float32)
        size = lin.size
        for i in prange(lshape0, nogil=True, schedule="static"):
            for j in range(lshape1):
                idx = LUT[i,j].idx
                coef = LUT[i,j].coef
                if coef<=0:
                    continue
                if idx>=size:
                    with gil:
                        logger.warning("Accessing %i >= %i !!!"%(idx,size))
                        continue
                lout[i] += lin[idx] * coef
        return out[:img_shape[0],:img_shape[1]]

    @timeit
    def uncorrect(self, image):
        """
        Take an image which has been corrected and transform it into it's raw (with loose of information)

        @param image: 2D-array with the image
        @return: uncorrected 2D image and a mask (pixels in raw image
        """
        if self.LUT is None:
            with self._sem:
                if self.LUT is None:
                    self.calc_LUT()
        out = numpy.zeros(self.shape,dtype=numpy.float32)
        mask = numpy.zeros(self.shape, dtype=numpy.int8)
        lmask = mask.ravel()
        lout = out.ravel()
        lin = image.ravel()
        tot = self.LUT.coef.sum(axis= -1)
        for idx in range(self.LUT.shape[0]):
            t = tot[idx]
            if t <= 0:
                lmask[idx] = 1
                continue
            val = lin[idx]/t
            lout[self.LUT[idx].idx] += val * self.LUT[idx].coef
        return out, mask
