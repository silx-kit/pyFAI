# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


"""
Module providing inversion transformation from pixel coordinate to radial/azimuthal
coordinate.
"""

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "17/05/2019"
__copyright__ = "2018-2018, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

include "regrid_common.pxi"

import logging
logger = logging.getLogger("pyFAI.ext.invert_geometry")

cdef class InvertGeometry:
    """
    Class to inverse the geometry: takes the nearest pixel then use linear interpolation

    :param radius: 2D array with radial position
    :param angle: 2D array with azimuthal position

    Call it with (r,chi) to retrieve the pixel where it comes from.
    """
    cdef:
        position_t[:, ::1] radius, angle
        int dim0, dim1
        position_t rad_min, rad_max, rad_scale, ang_min, ang_max, ang_scale

    def __cinit__(self, radius, angle):
        """Constructor of the class

        :param radius: 2D array with the radius for every position
        :param angle:  2D array with the azimuth for every position
        """
        cdef:
            int id0, id1
            position_t rad_min, rad_max, ang_min, ang_max, ang, rad
        self.dim0 = radius.shape[0]
        self.dim1 = radius.shape[1]

        assert angle.shape[0] == self.dim0, "the two array have the same shape"
        assert angle.shape[1] == self.dim1, "the two array have the same shape"

        self.radius = numpy.ascontiguousarray(radius, position_d)
        self.angle = numpy.ascontiguousarray(angle, position_d)
        rad_min = rad_max = self.radius[0, 0]
        ang_min = ang_max = self.angle[0, 0]
        for id0 in range(self.dim0):
            for id1 in range(self.dim1):
                ang = self.angle[id0, id1]
                rad = self.radius[id0, id1]
                if ang > ang_max:
                    ang_max = ang
                elif ang < ang_min:
                    ang_min = ang
                if rad > rad_max:
                    rad_max = rad
                elif rad < rad_min:
                    rad_min = rad
        self.rad_min = rad_min
        self.rad_max = rad_max
        self.rad_scale = 1.0 / (rad_max - rad_min) ** 2
        self.ang_min = ang_min
        self.ang_max = ang_max
        self.ang_scale = 1.0 / (ang_max - ang_min) ** 2

    def __dealloc__(self):
        self.radius = None
        self.angle = None

    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def __call__(self, position_t rad, position_t ang, bint refined=True):
        """Calculate the pixel coordinate leading to the value (rad, angle)

        :param rad: radial value
        :param ang: angular value
        :param refined: if True: use linear interpolation, else provide nearest pixel
        :return p0, p1: pixel coordinates
        """
        cdef:
            int id0, id1, best0, best1
            position_t cost, min_cost, gr0, ga0, gr1, ga1, cor0, cor1, target_ang, target_rad, det
        with nogil:
            best0 = best1 = 0
            cor0 = cor1 = 0.0
            cost = self.ang_scale * (self.angle[0, 0] - ang) ** 2 \
                 + self.rad_scale * (self.radius[0, 0] - rad) ** 2
            min_cost = cost
            for id0 in range(self.dim0):
                for id1 in range(self.dim1):
                    cost = self.ang_scale * (self.angle[id0, id1] - ang) ** 2 \
                         + self.rad_scale * (self.radius[id0, id1] - rad) ** 2
                    if cost < min_cost:
                        min_cost = cost
                        best0 = id0
                        best1 = id1
            if refined and \
                    (best0 > 0) and (best0 < self.dim0 - 1) and\
                    (best1 > 0) and (best1 < self.dim1 - 1):

                # First order Taylor expansion
                gr0 = 0.5 * (self.radius[best0 + 1, best1] - self.radius[best0 - 1, best1])
                ga0 = 0.5 * (self.angle[best0 + 1, best1] - self.angle[best0 - 1, best1])

                gr1 = 0.5 * (self.radius[best0, best1 + 1] - self.radius[best0, best1 - 1])
                ga1 = 0.5 * (self.angle[best0, best1 + 1] - self.angle[best0, best1 - 1])
                target_ang = ang - self.angle[best0, best1]
                target_rad = rad - self.radius[best0, best1]

                # inversion of the matrix
                det = ga1 * gr0 - ga0 * gr1
                if det == 0.0:
                    with gil:
                        logger.info("Impossible to invert the matrix")
                else:
                    cor0 = (target_rad * ga1 - target_ang * gr1) / det
                    cor1 = (-target_rad * ga0 + target_ang * gr0) / det
        return (best0 + cor0, best1 + cor1)
