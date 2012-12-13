#!/usr/bin/env python
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

import cython
cimport numpy
import numpy

ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t

def boundingBox(data):
    """
    Calculate bounding box around   

    @param img: 2D array like
    @return: 4-typle (d0_min, d1_min, d0_max, d1_max)
    
    
    NOTA: Does not work :( 
     
    """
    cdef numpy.ndarray[long, ndim = 1] shape = numpy.array(data.shape)
    cdef long ndims = data.ndim

#    cdef numpy.ndarray[float, ndim = ndims] fdata = data.astype(float)
    cdef numpy.ndarray[long, ndim = 1] mins = numpy.array(shape)
    cdef numpy.ndarray[long, ndim = 1] maxs = numpy.zeros(ndims, dtype=int)


    cdef long  i = 0
    cdef long  j = 0
    cdef long  k = 0
    cdef long  l = 0
#    cdef DTYPE_float64_t x = 0.0
#    cdef DTYPE_float64_t zero64 = 0.0
    if ndims == 1:
#        with nogil:
            for i in range(shape[0]):
                if data[i] > 0.0:
                    if i < mins[0]:
                        mins[0] = i
                    if i > maxs[0]:
                        maxs[0] = i
    elif ndims == 2:
#        with nogil:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if data[i, j] > 0.0 :
                        if i < mins[0]:
                            mins[0] = i
                        if i > maxs[0]:
                            maxs[0] = i
                        if j < mins[1]:
                            mins[1] = i
                        if j > maxs[1]:
                            maxs[1] = i
    elif ndims == 3:
#        with nogil:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        if  data[i, j, k] > 0.0:
                            if i < mins[0]:
                                mins[0] = i
                            if i > maxs[0]:
                                maxs[0] = i
                            if j < mins[1]:
                                mins[1] = i
                            if j > maxs[1]:
                                maxs[1] = i
                            if k < mins[2]:
                                mins[2] = i
                            if k > maxs[2]:
                                maxs[2] = i
    elif ndims == 4:
#        with nogil:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            if  data[i, j, k, l] > 0.0:
                                if i < mins[0]:
                                    mins[0] = i
                                if i > maxs[0]:
                                    maxs[0] = i
                                if j < mins[1]:
                                    mins[1] = i
                                if j > maxs[1]:
                                    maxs[1] = i
                                if k < mins[2]:
                                    mins[2] = i
                                if k > maxs[2]:
                                    maxs[2] = i
                                if l < mins[3]:
                                    mins[3] = i
                                if l > maxs[3]:
                                    maxs[3] = i
    else:
        raise RuntimeError("Dimensions > 4 not implemented")
    return tuple(mins) + tuple(maxs)

