# -*- coding: utf-8 -*-
# Copyright (C) 2012, Almar Klein
# Copyright (C) 2014-2016, European Synchrotron Radiation Facility
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Cythonized version of the marching square function for "isocontour" plot
"""
__authors__ = ["Almar Klein", "Jerome Kieffer"]
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "31/05/2016"
__status__ = "stable"
__license__ = "BSD-3 clauses"

# Cython specific imports

import numpy
cimport numpy
import cython
from libc.math cimport M_PI, sin, floor, fabs
cdef double epsilon = numpy.finfo(numpy.float64).eps
from cython.view cimport array as cvarray
from ..decorators import timeit

cdef numpy.int8_t[:, :] EDGETORELATIVEPOSX = numpy.array([[0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], dtype=numpy.int8)
cdef numpy.int8_t[:, :] EDGETORELATIVEPOSY = numpy.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0], [0, 0], [1, 1], [1, 1]], dtype=numpy.int8)
cdef numpy.int8_t[:, :] EDGETORELATIVEPOSZ = numpy.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [0, 1], [0, 1], [0, 1], [0, 1]], dtype=numpy.int8)
cdef numpy.int8_t[:, :] CELLTOEDGE = numpy.array([
                                                    [0, 0, 0, 0, 0],  # Case 0: nothing
                                                    [1, 0, 3, 0, 0],  # Case 1
                                                    [1, 0, 1, 0, 0],  # Case 2
                                                    [1, 1, 3, 0, 0],  # Case 3

                                                    [1, 1, 2, 0, 0],  # Case 4
                                                    [2, 0, 1, 2, 3],  # Case 5 > ambiguous
                                                    [1, 0, 2, 0, 0],  # Case 6
                                                    [1, 2, 3, 0, 0],  # Case 7

                                                    [1, 2, 3, 0, 0],  # Case 8
                                                    [1, 0, 2, 0, 0],  # Case 9
                                                    [2, 0, 3, 1, 2],  # Case 10 > ambiguous
                                                    [1, 1, 2, 0, 0],  # Case 11

                                                    [1, 1, 3, 0, 0],  # Case 12
                                                    [1, 0, 1, 0, 0],  # Case 13
                                                    [1, 0, 3, 0, 0],  # Case 14
                                                    [0, 0, 0, 0, 0],  # Case 15
                                                ], dtype=numpy.int8)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def marching_squares(float[:, :] img, double isovalue,
                     numpy.int8_t[:, :] cellToEdge,
                     numpy.int8_t[:, :] edgeToRelativePosX,
                     numpy.int8_t[:, :] edgeToRelativePosY):
    cdef:
        int dim_y = img.shape[0]
        int dim_x = img.shape[1]
        numpy.ndarray[numpy.float32_t, ndim = 2] edges = numpy.zeros((dim_x * dim_y, 2), numpy.float32)
        int x, y, z, i, j, k, index, edgeCount = 0
        int dx1, dy1, dz1, dx2, dy2, dz2
        double fx, fy, fz, ff, tmpf, tmpf1, tmpf2
    with nogil:
        for y in range(dim_y - 1):
            for x in range(dim_x - 1):

                # Calculate index.
                index = 0
                if img[y, x] > isovalue:
                    index += 1
                if img[y, x + 1] > isovalue:
                    index += 2
                if img[y + 1, x + 1] > isovalue:
                    index += 4
                if img[y + 1, x] > isovalue:
                    index += 8

                # Resolve ambiguity
                if index == 5 or index == 10:
                    # Calculate value of cell center (i.e. average of corners)
                    tmpf = 0.25 * (img[y, x] +
                                   img[y, x + 1] +
                                   img[y + 1, x] +
                                   img[y + 1, x + 1])
                    # If below isovalue, swap
                    if tmpf <= isovalue:
                        if index == 5:
                            index = 10
                        else:
                            index = 5

                # For each edge ...
                for i in range(cellToEdge[index, 0]):
                    # For both ends of the edge ...
                    for j in range(2):
                        # Get edge index
                        k = cellToEdge[index, 1 + i * 2 + j]
                        # Use these to look up the relative positions of the pixels to interpolate
                        dx1, dy1 = edgeToRelativePosX[k, 0], edgeToRelativePosY[k, 0]
                        dx2, dy2 = edgeToRelativePosX[k, 1], edgeToRelativePosY[k, 1]
                        # Define "strength" of each corner of the cube that we need
                        tmpf1 = 1.0 / (epsilon + fabs(img[y + dy1, x + dx1] - isovalue))
                        tmpf2 = 1.0 / (epsilon + fabs(img[y + dy2, x + dx2] - isovalue))
                        # Apply a kind of center-of-mass method
                        fx, fy, ff = 0.0, 0.0, 0.0
                        fx += <double> dx1 * tmpf1;
                        fy += <double>dy1 * tmpf1;
                        ff += tmpf1
                        fx += <double> dx2 * tmpf2;
                        fy += <double> dy2 * tmpf2;
                        ff += tmpf2
                        #
                        fx /= ff
                        fy /= ff
                        # Append point
                        edges[edgeCount, 0] = <float> (x + fx)
                        edges[edgeCount, 1] = <float> (y + fy)
                        edgeCount += 1
    return edges[:edgeCount, :]


@cython.boundscheck(False)
def sort_edges(edges):
    """
    Reorder edges in such a way they become contiguous
    """
    cdef:
        int size = edges.shape[0]
        int[:] pos = cvarray(shape=(size,), itemsize=sizeof(int), format="i")
        int[:] remaining = cvarray(shape=(size,), itemsize=sizeof(int), format="i")
        float[:, :] dist2 = cvarray(shape=(size, size), itemsize=sizeof(float), format="f")
        float d
        int i, j, index = 0, current = 0
        float[:, :]edges_ = numpy.ascontiguousarray(edges, numpy.float32)
    dist2[:, :] = 0
    pos[:] = 0

    with nogil:
        # initialize the distance (squared) array:
        for i in range(size):
            remaining[i] = i
            for j in range(i + 1, size):
                d = (edges_[i, 0] - edges_[j, 0]) ** 2 + (edges_[i, 1] - edges_[j, 1]) ** 2
                dist2[i, j] = d
                dist2[j, i] = d
        # set element in remaining to -1 when already transfered
        # O(n^2) implementation, not bright, any better idea is welcome
        remaining[0] = -1
        pos[0] = 0
        for i in range(1, size):
            current = pos[i - 1]
            index = -1
            for j in range(1, size):
                if remaining[j] == -1:
                    continue
                elif index == -1:  # not yet found a candidate
                    index = remaining[j]
                    d = dist2[index, current]
                    continue
                elif dist2[j, current] < d:
                    index = j
                    d = dist2[current, index]
            pos[i] = index
            remaining[index] = -1
    return edges[pos, :]


def isocontour(img, isovalue=None, sorted=False):
    """ isocontour(img, isovalue=None)

    Calculate the iso contours for the given 2D image. If isovalue
    is not given or None, a value between the min and max of the image
    is used.

    @param img: 2D array representing the image
    @param isovalue: the value for which the iso_contour shall be calculated
    @param sorted: perform a sorting of the points to have them contiguous ?

    Returns a pointset in which each two subsequent points form a line
    piece. This van be best visualized using "vv.plot(result, ls='+')".

    """

    # Check image
    if not isinstance(img, numpy.ndarray) or (img.ndim != 2):
        raise ValueError('img should be a 2D numpy array.')

    # Get isovalue
    if isovalue is None:
        isovalue = 0.5 * (img.min() + img.max())
    else:
        # Will raise error if not float-like value given
        isovalue = float(isovalue)
    res = marching_squares(numpy.ascontiguousarray(img, numpy.float32), isovalue,
                           CELLTOEDGE, EDGETORELATIVEPOSX, EDGETORELATIVEPOSY)
    if sorted:
        return sort_edges(res)
    else:
        return res
