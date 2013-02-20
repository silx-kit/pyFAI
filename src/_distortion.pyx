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
from libc.math cimport floor,ceil, fabs
import os

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
    cdef double[:,:] box
    cdef double A0, A1, B0, B1, C0, C1, D0, D1, pAB, pBC, pCD, pDA, cAB, cBC, cCD, cDA, area
    cdef int offset0, offset1, box_size0, box_size1
    cdef bint has_area, has_slope

    def __cinit__(self, double[:,:] buffer):
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
        return os.linesep.join(["offset %i,%i size %i, %i" % (self.offset0, self.offset1, self.box_size0, self.box_size1), "box: %s" % self.box[:self.box_size0, :self.box_size1]])

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

    cpdef double calc_area_AB(self, double I1,double I2):
        if self.B0 != self.A0:
            return 0.5 * (I2 - I1) * (self.pAB * (I2 + I1) + 2 * self.cAB)
        else:
            return 0.0
        
    cpdef double calc_area_BC(self, double J1, double J2):
        if self.B0 != self.C0:
            return 0.5 * (J2 - J1) * (self.pBC * (J1 + J2) + 2 * self.cBC)
        else:
            return 0.0
        
    cpdef double calc_area_CD(self, double K1, double K2):
        if self.C0 != self.D0:
            return 0.5 * (K2 - K1) * (self.pCD * (K2 + K1) + 2 * self.cCD)
        else:
            return 0.0
        
    cpdef double calc_area_DA(self,double L1,double L2):
        if self.D0 != self.A0:
            return 0.5 * (L2 - L1) * (self.pDA * (L1 + L2) + 2 * self.cDA)
        else:
            return 0.0
        
    cpdef double  calc_area_old(self):
        if not self.area:
            if not self.has_slope:
                self.init_slope()
            self.area = -self.calc_area_AB(self.A0, self.B0) - \
                        self.calc_area_BC(self.B0, self.C0) - \
                        self.calc_area_CD(self.C0, self.D0) - \
                        self.calc_area_DA(self.D0, self.A0)
            self.has_area = 1
        return self.area

    cpdef double calc_area(self):
        if not self.has_area:
            self.area = 0.5*(self.C0 - self.A0)*(self.D1 - self.B1)-(self.C1 - self.A1)*(self.D0 - self.B0)
            self.has_area = 1
        return self.area

    def populate_box(self):
        cdef int i0, i1
        cdef double area,value
        if self.pAB is None:
            self.init_slope()
        self.integrateAB(self.B0, self.A0, self.calc_area_AB)
        self.integrateAB(self.A0, self.D0, self.calc_area_DA)
        self.integrateAB(self.D0, self.C0, self.calc_area_CD)
        self.integrateAB(self.C0, self.B0, self.calc_area_BC)
        area = self.calc_area()
        for i0 in range(self.box_size0):
            for i1 in range(self.box_size1):
                value = self.box[i0,i1] / area
                self.box[i0,i1] = value
                if value < 0.0:
                    print self.box
                    self.box[:, :] = 0
                    print "AB"
                    self.integrateAB(self.B0, self.A0, self.calc_area_AB)
                    print self.box
                    self.box[:, :] = 0
                    print "DA"
                    self.integrateAB(self.A0, self.D0, self.calc_area_DA)
                    print self.box
                    self.box[:, :] = 0
                    print "CD"
                    self.integrateAB(self.D0, self.C0, self.calc_area_CD)
                    print self.box
                    self.box[:, :] = 0
                    print "BC"
                    self.integrateAB(self.C0, self.B0, self.calc_area_BC)
                    print self.box
                    print self
                    raise RuntimeError()
                
    def integrateAB(self, double start, double stop, calc_area):
        cdef int i, h = 0
        cdef double P,dP, A, AA, dA, sign
#        print start, stop, calc_area(start, stop)
        if start < stop:  # positive contribution
            P = ceil(start)
            dP = P - start
#            print "Integrate", start, P, stop, calc_area(start, stop)
            if P > stop:  # start and stop are in the same unit
                A = calc_area(start, stop)
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
                    A = calc_area(start, P)
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
                    A = calc_area(i, i + 1)
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
                    A = calc_area(P, stop)
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
                A = calc_area(start, stop)
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
                    A = calc_area(start, P)
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
                    A = calc_area(i, i - 1)
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
                    A = calc_area(P, stop)
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