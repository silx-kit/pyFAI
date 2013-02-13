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
from pyFAI.detectors import Detector

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/01/2013"
__status__ = "development"

import logging
import types, os
import numpy
logger = logging.getLogger("pyFAI.distortion")
from math import ceil, floor
from pyFAI import detectors

class Distortion(object):
    """
    This class applies a distortion correction on an image.

    """
    def __init__(self, detector="detector", shape=None):
        """
        @param detector: detector instance or detector name
        """
        if type(detector) in types.StringTypes:
            self.detector = detectors.detector_factory(detector)
        else: #we assume it is a Detector instance
            self.detector = detector
        if "max_shape" in dir(self.detector):
            self.shape =  self.detector.max_shape
        else:
            self.shape = shape
        self.shape = tuple([int(i) for i in self.shape])

    def __repr__(self):
        return os.linesep.join(["Distortion correction for detector:",
                                self.detector.__repr__()])

    def correct(self, img):
        """
        Correct the image
        """
        shape = img.shape
        corr = numpy.zeros(shape, dtype=numpy.float32)

#    def calc_lut(self,shape):
    def split_pixel(self,):
        pass
    def calc_LUT_size(self):
        """
        TODO: here we have a problem:
        Considering the "half-CCD" spline from ID11 which describes a (1025,2048) detector,
        the physical location of pixels should go from:
        [-17.48634 : 1027.0543, -22.768829 : 2028.3689]
        we have 2 options:
         - discard pixels falling outside the [0:1025,0:2048] range with a lose of intensity
         - grow the output image to [-18:1028,-23:2029] with many empty pixels and an offset ) but conservation of the total intensity.
        """
        pos_corners = numpy.empty((self.shape[0] + 1, self.shape[1] + 1, 2), dtype=numpy.float32)
        d1 = numpy.outer(numpy.arange(self.shape[0] + 1, dtype=numpy.float32), numpy.ones(self.shape[1] + 1, dtype=numpy.float32)) - 0.5
        d2 = numpy.outer(numpy.ones(self.shape[0] + 1, dtype=numpy.float32), numpy.arange(self.shape[1] + 1, dtype=numpy.float32)) - 0.5
        pos_corners[:, :, 0], pos_corners[:, :, 1] = self.detector.calc_cartesian_positions(d1, d2)
        pos_corners[:, :, 0] /= self.detector.pixel1
        pos_corners[:, :, 1] /= self.detector.pixel2
        pos = numpy.empty((self.shape[0], self.shape[1], 4, 2), dtype=numpy.float32)
        pos[:, :, 0, :] = pos_corners[:-1, :-1]
        pos[:, :, 1, :] = pos_corners[:-1, 1: ]
        pos[:, :, 2, :] = pos_corners[1: , 1: ]
        pos[:, :, 3, :] = pos_corners[1: , :-1]
        pos0min = numpy.maximum(numpy.floor(pos[:, :, :, 0].min(axis= -1)).astype(numpy.int32), 0)
        pos1min = numpy.maximum(numpy.floor(pos[:, :, :, 1].min(axis= -1)).astype(numpy.int32), 0)
        pos0max = numpy.minimum(numpy.ceil(pos[:, :, :, 0].max(axis= -1)).astype(numpy.int32) + 1, self.shape[0])
        pos1max = numpy.minimum(numpy.ceil(pos[:, :, :, 1].max(axis= -1)).astype(numpy.int32) + 1, self.shape[1])
        lut_size = numpy.zeros(self.shape, dtype=numpy.int32)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                lut_size[pos0min[i, j]:pos0max[i, j], pos1min[i, j]:pos1max[i, j]] += 1
        self.lut_size = lut_size.max()
        return lut_size

class Quad(object):
    """

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
    def __init__(self, A, B, C, D):
        self.Ax = A[0]
        self.Ay = A[1]
        self.Bx = B[0]
        self.By = B[1]
        self.Cx = C[0]
        self.Cy = C[1]
        self.Dx = D[0]
        self.Dy = D[1]
        self.offset_x = int(min(self.Ax, self.Bx, self.Cx, self.Dx))
        self.offset_y = int(min(self.Ay, self.By, self.Cy, self.Dy))
        self.box_size_x = int(ceil(max(self.Ax, self.Bx, self.Cx, self.Dx))) - self.offset_x
        self.box_size_y = int(ceil(max(self.Ay, self.By, self.Cy, self.Dy))) - self.offset_y
        self.Ax -= self.offset_x
        self.Ay -= self.offset_y
        self.Bx -= self.offset_x
        self.By -= self.offset_y
        self.Cx -= self.offset_x
        self.Cy -= self.offset_y
        self.Dx -= self.offset_x
        self.Dy -= self.offset_y

        self.pAB = self.pBC = self.pCD = self.pDA = None
        self.cAB = self.cBC = self.cCD = self.cDA = None

        self.area = self.box = None

    def init_slope(self):
        if self.pAB is None:
            if self.Bx == self.Ax:
                self.pAB = numpy.inf
            else:
                self.pAB = (self.By - self.Ay) / (self.Bx - self.Ax)
            if self.Cx == self.Bx:
                self.pBC = numpy.inf
            else:
                 self.pBC = (self.Cy - self.By) / (self.Cx - self.Bx)
            if self.Dx == self.Cx:
                self.pCD = numpy.inf
            else:
                self.pCD = (self.Dy - self.Cy) / (self.Dx - self.Cx)
            if self.Ax == self.Dx:
                self.pDA = numpy.inf
            else:
                self.pDA = (self.Ay - self.Dy) / (self.Ax - self.Dx)
            self.cAB = self.Ay - self.pAB * self.Ax
            self.cBC = self.By - self.pBC * self.Bx
            self.cCD = self.Cy - self.pCD * self.Cx
            self.cDA = self.Dy - self.pDA * self.Dx

            self.box = numpy.zeros((self.box_size_x, self.box_size_y), dtype=numpy.float32)


    def calc_area_AB(self, I1x, I2x):
        return 0.5 * (I2x - I1x) * (self.pAB * (I2x + I1x) + 2 * self.cAB)
    def calc_area_BC(self, J1x, J2x):
        return 0.5 * (J2x - J1x) * (self.pBC * (J1x + J2x) + 2 * self.cBC)
    def calc_area_CD(self, K1x, K2x):
        return 0.5 * (K2x - K1x) * (self.pCD * (K2x + K1x) + 2 * self.cCD)
    def calc_area_DA(self, L1x, L2x):
        return 0.5 * (L2x - L1x) * (self.pDA * (L1x + L2x) + 2 * self.cDA)
    def calc_area(self):
        if self.area is None:
            if self.pAB is None:
                self.init_slope()
            self.area = -self.calc_area_AB(self.Ax, self.Bx) - \
               self.calc_area_BC(self.Bx, self.Cx) - \
               self.calc_area_CD(self.Cx, self.Dx) - \
               self.calc_area_DA(self.Dx, self.Ax)
        return self.area
    def populate_box(self):
        self.integrateAB(self.Bx, self.Ax, self.calc_area_AB)
        self.integrateAB(self.Ax, self.Dx, self.calc_area_DA)
        self.integrateAB(self.Dx, self.Cx, self.calc_area_CD)
        self.integrateAB(self.Cx, self.Bx, self.calc_area_BC)
#        print self.box.T
        self.box /= self.calc_area()
    def integrateAB(self, Ax, Bx, calc_area):
        h = 0
        if Ax < Bx: #positive contribution
            P = ceil(Ax)
            dP = P - Ax
            if dP > 0:
                A = calc_area(Ax, P)
                h = 0
                dA = dP
                while A > 0:
                    if dA > A:
                        dA = A
                    self.box[int(P) - 1, h] += dA
                    A -= dA
                    h += 1
            #subsection P1->Pn
            for i in range(int(P), int(Bx)):
                A = calc_area(i, i + 1)
                h = 0
                dA = 1.0
                while A > 0:
                    if dA > A:
                        dA = A
                    self.box[i , h] += dA
                    A -= dA
                    h += 1
            #Section Pn->B
            P = floor(Bx)
            dP = Bx - P
            if dP > 0:
                A = calc_area(P, Bx)
                h = 0
                dA = dP
                while A > 0:
                    if dA > A:
                        dA = A
                    self.box[int(P), h] += dA
                    A -= dA
                    h += 1
        elif    Ax > Bx: #negative contribution. Nota is Ax=Bx: no contribution
            P = floor(Ax)
            dP = P - Ax
            if dP < 0:
                A = calc_area(Ax, P)
                h = 0
                dA = dP
                while A < 0:
                    if dA < A:
                        dA = A
                    self.box[int(P) , h] += dA
                    A -= dA
                    h += 1
            #subsection P1->Pn
            for i in range(int(Ax), int(ceil(Bx)), -1):
                A = calc_area(i, i - 1)
                h = 0
                dA = -1.0
                print A
                while A < 0:
                    if dA < A:
                        dA = A
                    self.box[i - 1 , h] += dA
                    A -= dA
                    h += 1
            #Section Pn->B
            P = ceil(Bx)
            dP = Bx - P
            if dP < 0:
                A = calc_area(P, Bx)
                h = 0
                dA = dP
                while A < 0:
                    if dA < A:
                        dA = A
                    self.box[int(Bx), h] += dA
                    A -= dA
                    h += 1
def test():
    Q = Quad((7.5, 6.5), (2.5, 5.5), (3.5, 1.5), (8.5, 1.5))
    Q.init_slope()
    print Q.calc_area_AB(Q.Ax, Q.Bx)
    print Q.calc_area_BC(Q.Bx, Q.Cx)
    print Q.calc_area_CD(Q.Cx, Q.Dx)
    print Q.calc_area_DA(Q.Dx, Q.Ax)
    print Q.calc_area()
    Q.populate_box()
    print Q.box.T
    print Q.box.sum()
    import fabio, numpy
    raw = numpy.arange(256 * 256)
    raw.shape = 256, 256
    det = detectors.FReLoN("frelon_8_8.spline")
    print det, det.max_shape
    dis = Distortion(det)
    print dis
#    print Q.calc_area_DA(7.5, 8.5)

if __name__ == "__main__":
    test()
