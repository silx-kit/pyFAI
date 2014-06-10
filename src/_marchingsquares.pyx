# -*- coding: utf-8 -*-
# Copyright (C) 2012, Almar Klein
# Copyright (C) 2014, Jerome Kieffer
# Cython specific imports
import numpy 
cimport numpy
import cython
from libc.math cimport M_PI, import sin, floor, fabs



@cython.boundscheck(False)
@cython.wraparound(False)
def marching_squares(float[:,:] img, double isovalue,
                     numpy.int8_t[:,:] cellToEdge, 
                     numpy.int8_t[:,:] edgeToRelativePosX, 
                     numpy.int8_t[:,:] edgeToRelativePosY):
    cdef int dim_y = img.shape[0]
    cdef int dim_x = img.shape[1]
    #output arrays
    cdef numpy.ndarray[numpy.float32_t, ndim=2] edges_ = numpy.zeros((dim_x*dim_y,2), numpy.float32)

    cdef int x, y, z, i, j, k, index
    cdef int dx1, dy1, dz1, dx2, dy2, dz2
    cdef double fx, fy, fz, ff, tmpf, tmpf1, tmpf2, edgeCount = 0.0
    
    for y in range(dim_y-1):
        for x in range(dim_x-1):

            # Calculate index.
            index = 0
            if img[y, x] > isovalue:
                index += 1
            if img[y, x+1] > isovalue:
                index += 2
            if img[y+1, x+1] > isovalue:
                index += 4
            if img[y+1, x] > isovalue:
                index += 8

            # Resolve ambiguity
            if index == 5 or index == 10:
                # Calculate value of cell center (i.e. average of corners)
                tmpf = 0.0
                for dy1 in [0, 1]:
                    for dx1 in [0,1]:
                        tmpf += img[y+dy1,x+dx1]
                tmpf /= 4.0
                # If below isovalue, swap
                if tmpf <= isovalue:
                    if index == 5:
                        index = 10
                    else:
                        index = 5

            # For each edge ...
            for i in range(cellToEdge_[index,0]):
                # For both ends of the edge ...
                for j in range(2):
                    # Get edge index
                    k = cellToEdge_[index, 1+i*2+j]
                    # Use these to look up the relative positions of the pixels to interpolate
                    dx1, dy1 = edgeToRelativePosX_[k,0], edgeToRelativePosY_[k,0]
                    dx2, dy2 = edgeToRelativePosX_[k,1], edgeToRelativePosY_[k,1]
                    # Define "strength" of each corner of the cube that we need
                    tmpf1 = 1.0 / (0.0001 + fabs( img[y+dy1,x+dx1] - isovalue))
                    tmpf2 = 1.0 / (0.0001 + fabs( img[y+dy2,x+dx2] - isovalue))
                    # Apply a kind of center-of-mass method
                    fx, fy, ff = 0.0, 0.0, 0.0
                    fx += <double>dx1 * tmpf1;  fy += <double>dy1 * tmpf1;  ff += tmpf1
                    fx += <double>dx2 * tmpf2;  fy += <double>dy2 * tmpf2;  ff += tmpf2
                    #
                    fx /= ff
                    fy /= ff
                    # Append point
                    edges_[edgeCount,0] = <double>x + fx
                    edges_[edgeCount,1] = <double>y + fy
                    edgeCount += 1

    # Done
    return edges[:edgeCount,:]
