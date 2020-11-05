# -*- coding: utf-8 -*-
#cython: embedsignature=True, language_level=3
#cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False,
## This is for developping:
##cython: profile=True, warn.undeclared=True, warn.unused=True, warn.unused_result=False, warn.unused_arg=True
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2020-2020 European Synchrotron Radiation Facility, Grenoble, France
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
Export the mask as a set of rectangles.

This feature is needed for single crystal analysis programs (XDS, Crysalis, ...) 
"""
__author__ = "Jérôme Kieffer"
__date__ = "05/11/2020"
__contact__ = "Jerome.kieffer@esrf.fr"
__license__ = "MIT"

import numpy
cimport numpy

cdef struct Pair:
    long start, height

cdef class Rectangle:
    cdef:
        public long height, width, row, col
        
    def __cinit__(self, long height, long width, long row=0, long col=0):
        self.height = height
        self.width = width
        self.row = row
        self.col = col
        
    def __repr__(self):
        return f"Rectangle row:{self.row} col:{self.col} heigth:{self.height} width:{self.width} area:{self.area}"

    cdef long _area(self):
        return self.width*self.height

    @property
    def area(self):
        return self._area()    
    
cdef class Stack:
    cdef:
        long last, size
        long[:, ::1] stack
        
    def __cinit__(self, long size):
        self.stack = numpy.zeros((size, 2), dtype=numpy.int64)
        self.last = 0
        self.size = size
        
    def __dealloc__(self):
        self.stack = None
    
    cpdef push(self, long start, long height):
        if self.last<self.size:
            self.stack[self.last, 0] = start
            self.stack[self.last, 1] = height
            self.last += 1
        else:
            print("Overfull stack")
            
    cpdef Pair top(self):
        cdef Pair res
        if self.last:
            res=Pair(self.stack[self.last-1, 0], self.stack[self.last-1, 1])
            return res
        else:
            print("Emtpy stack")
    
    cpdef Pair pop(self):
        cdef Pair res
        if self.last:
            res = self.top()
            self.last -= 1
            return res
        else:
            print("Emtpy stack")
    
    cpdef long empty(self):
        return self.last == 0
    
    
cpdef Rectangle get_max_area(long[::1] histo, long row=-1):
    cdef:
        long size, height, start, pos
        Stack stack
        Pair top
        Rectangle best
    best = Rectangle(0, 0, 0, 0)
    size = histo.size
    stack = Stack(size)

    for pos in range(size):
        height = histo[pos]
        start = pos # position where rectangle starts
        while True:
            if (stack.empty()) or (height > stack.top().height): 
                stack.push(start, height)
            elif (not stack.empty()) and (height < stack.top().height):
                top = stack.top()
                if top.height * (pos - top.start) > best._area():
                    best = Rectangle(top.height, (pos - top.start), row-top.height+1, top.start)
                start = stack.pop().start
                continue
            break
    pos += 1
    while not stack.empty():
        top = stack.pop()
        if top.height * (pos - top.start) > best._area():
            best = Rectangle(top.height, (pos - top.start), row-top.height+1, top.start)

    return best


cpdef Rectangle get_largest_rectangle(numpy.int8_t[:, ::1] ary):
    """Find the largest rectangular region
    
    :param mask: 2D array with 1 for invalid pixels (0 elsewhere)
    :return: Largest rectangle of masked data
    """
    cdef:
        long ncols, nrows, i, j
        Rectangle rect, best
        long[::1] histogram
    nrows = ary.shape[0]
    ncols = ary.shape[1]
    histogram = numpy.zeros(ncols, dtype=numpy.int64)
    best = Rectangle(0, 0, -1, -1)
    for i in range(nrows):
        for j in range(ncols):
            if ary[i, j]:
                histogram[j] += 1
            else:
                histogram[j] = 0
        rect = get_max_area(histogram, i)
        if rect.area > best.area:
            best = rect 
    return best


cpdef bint any_non_zero(numpy.int8_t[::1] linear):
    cdef:
        int index
    for index in range(linear.size):
        if linear[index]:
            return True
    return False


def decompose_mask(mask):
    """Decompose a mask into a list of hiding rectangles
    
    :param mask: 2D array with 1 for invalid pixels (0 elsewhere)
    :return: list of Rectangles
    """    
    cdef:
        long idx, rlower, rupper, clower, cupper, width
        list res = []
        numpy.int8_t[:, ::1] remaining
        numpy.int8_t[::1] linear 
        Rectangle r
        
    width = mask.shape[1]
    remaining = numpy.array(mask, dtype=numpy.int8)
    linear = numpy.asarray(remaining).ravel()
    
    while any_non_zero(linear):
        r = get_largest_rectangle(remaining)
        res.append(r)
        rlower = r.row
        rupper = rlower + r.height
        clower = r.col 
        cupper = clower + r.width
        
        #Memset with tweaking in the case of non contiguous access
        if clower>0 or cupper<width:
            for idx in range(rlower, rupper):
                remaining[idx, clower:cupper] = 0
        else: #We are luck and the memory is in one block
            remaining[rlower:rupper, clower:cupper] = 0
    return res

        