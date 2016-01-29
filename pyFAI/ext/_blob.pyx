# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal Integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:   Aurore Deschildre <auroredeschildre@gmail.com>    
#                        Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Some Cythonized function for blob detection function
"""
__authors__ = ["Aurore Deschildre", "Jerome Kieffer"]
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20/10/2014"
__status__ = "stable"
__license__ = "GPLv3+"
import cython
import numpy
cimport numpy
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def local_max(float[:,:,:] dogs, mask=None, bint n_5=False):
    """
    Calculate if a point is a maximum in a 3D space: (scale, y, x)
    
    @param dogs: 3D array of difference of gaussian
    @param mask: mask with invalid pixels
    @param N-5: take a neighborhood of 5x5 pixel in plane
    @return: 3d_array with 1 where is_max 
    """
    cdef bint do_mask = mask is not None
    cdef int ns, ny, nx, s, x, y
    cdef numpy.int8_t m 
    cdef float c 
    cdef numpy.int8_t[:,:] cmask
    ns = dogs.shape[0]
    ny = dogs.shape[1]
    nx = dogs.shape[2]
    if do_mask:
        assert mask.shape[0] == ny
        assert mask.shape[1] == nx
        cmask = numpy.ascontiguousarray(mask, dtype=numpy.int8)

    cdef numpy.ndarray[numpy.int8_t, ndim=3] is_max = numpy.zeros((ns,ny,nx), dtype=numpy.int8)
    if ns<3 or ny<3 or nx<3:
        return is_max
    for s in range(1,ns-1):
        for y in range(1,ny-1):
            for x in range(1,nx-1):
                c =  dogs[s,y,x]
                if do_mask and cmask[y,x]:
                    m = 0
                else:
                    m = (c>dogs[s,y,x-1]) and (c>dogs[s,y,x+1]) and\
                        (c>dogs[s,y-1,x]) and (c>dogs[s,y+1,x]) and\
                        (c>dogs[s,y-1,x-1]) and (c>dogs[s,y-1,x+1]) and\
                        (c>dogs[s,y+1,x-1]) and (c>dogs[s,y+1,x+1]) and\
                        (c>dogs[s-1,y,x]) and (c>dogs[s-1,y,x]) and\
                        (c>dogs[s-1,y,x-1]) and (c>dogs[s-1,y,x+1]) and\
                        (c>dogs[s-1,y-1,x]) and (c>dogs[s-1,y+1,x]) and\
                        (c>dogs[s-1,y-1,x-1]) and (c>dogs[s-1,y-1,x+1]) and\
                        (c>dogs[s-1,y+1,x-1]) and (c>dogs[s-1,y+1,x+1]) and\
                        (c>dogs[s+1,y,x-1]) and (c>dogs[s+1,y,x+1]) and\
                        (c>dogs[s+1,y-1,x]) and (c>dogs[s+1,y+1,x]) and\
                        (c>dogs[s+1,y-1,x-1]) and (c>dogs[s+1,y-1,x+1]) and\
                        (c>dogs[s+1,y+1,x-1]) and (c>dogs[s+1,y+1,x+1])
                    if not m:
                        continue
                    if n_5:
                        if x>1:
                            m = m and (c>dogs[s  ,y,x-2]) and (c>dogs[s  ,y-1,x-2]) and (c>dogs[s  ,y+1,x-2])\
                                  and (c>dogs[s-1,y,x-2]) and (c>dogs[s-1,y-1,x-2]) and (c>dogs[s-1,y+1,x-2])\
                                  and (c>dogs[s+1,y,x-2]) and (c>dogs[s+1,y-1,x-2]) and (c>dogs[s+1,y+1,x-2])
                            if y>1:
                                m = m and (c>dogs[s,y-2,x-2])and (c>dogs[s-1,y-2,x-2]) and (c>dogs[s,y-2,x-2])
                            if y<ny-2:
                                m = m and (c>dogs[s,y+2,x-2])and (c>dogs[s-1,y+2,x-2]) and (c>dogs[s,y+2,x-2])
                        if x<nx-2:
                            m = m and (c>dogs[s  ,y,x+2]) and (c>dogs[s  ,y-1,x+2]) and (c>dogs[s  ,y+1,x+2])\
                                  and (c>dogs[s-1,y,x+2]) and (c>dogs[s-1,y-1,x+2]) and (c>dogs[s-1,y+1,x+2])\
                                  and (c>dogs[s+1,y,x+2]) and (c>dogs[s+1,y-1,x+2]) and (c>dogs[s+1,y+1,x+2])
                            if y>1:
                                m = m and (c>dogs[s,y-2,x+2])and (c>dogs[s-1,y-2,x+2]) and (c>dogs[s,y-2,x+2])
                            if y<ny-2:
                                m = m and (c>dogs[s,y+2,x+2])and (c>dogs[s-1,y+2,x+2]) and (c>dogs[s,y+2,x+2])

                        if y>1:
                            m = m and (c>dogs[s  ,y-2,x]) and (c>dogs[s  ,y-2,x-1]) and (c>dogs[s  ,y-2,x+1])\
                                  and (c>dogs[s-1,y-2,x]) and (c>dogs[s-1,y-2,x-1]) and (c>dogs[s-1,y-2,x+1])\
                                  and (c>dogs[s+1,y-2,x]) and (c>dogs[s+1,y-2,x-1]) and (c>dogs[s+1,y+2,x+1])
                            
                        if y<ny-2:
                            m = m and (c>dogs[s  ,y+2,x]) and (c>dogs[s  ,y+2,x-1]) and (c>dogs[s  ,y+2,x+1])\
                                  and (c>dogs[s-1,y+2,x]) and (c>dogs[s-1,y+2,x-1]) and (c>dogs[s-1,y+2,x+1])\
                                  and (c>dogs[s+1,y+2,x]) and (c>dogs[s+1,y+2,x-1]) and (c>dogs[s+1,y+2,x+1])
                        
                is_max[s,y,x] = m
    return is_max 
