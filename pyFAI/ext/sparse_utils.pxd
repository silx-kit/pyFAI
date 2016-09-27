# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2016 European Synchrotron Radiation Facility, Grenoble, France
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

import numpy

cdef struct lut_point:
    int idx
    float coef


cdef class Vector2:
    cdef:
        float[:] coef
        int[:] idx
        int size, allocated
        
    cdef inline void append(self, int idx, float coef):
        cdef:
            int new_allocated 
            int[:] newidx
            float[:] newcoef
        if self.size >= self.allocated - 1:
                new_allocated = self.allocated * 2
                newcoef = numpy.empty(new_allocated, dtype=numpy.float32)
                newcoef[:self.size] = self.coef[:self.size]
                self.coef = newcoef
                newidx = numpy.empty(new_allocated, dtype=numpy.int32)
                newidx[:self.size] = self.idx[:self.size]
                self.idx = newidx
                self.allocated = new_allocated
        self.coef[self.size] = coef
        self.idx[self.size] = idx
        self.size = self.size + 1


cdef class ArrayBuilder:
    """Container object representing a sparse matrix for look-up table creation"""
    cdef:
        int size 
        readonly list lines

    cdef inline void append(self, int line, int col, float value):
        """append an element to a line of the array:
        
        :param line: line number 
        :param col: theoretical column value (will be stored in a more eficient way)
        :param value: the value to be stored at the given place 
        """
        cdef: 
            Vector2 vector
        vector = self.lines[line]
        vector.append(col, value)
        
