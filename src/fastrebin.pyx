#!/usr/bin/env python
# -*- coding: utf8 -*-
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

__author__ = "Jerome Kieffer"
__date__ = "27-12-2011"
__contact__ = "Jerome.kieffer@esrf.fr"
__license__ = "GPL v3+"
__doc__ = """
Basically the same algorithm than "splitBBox" but uses a look-up table allowing parallel rebinning.
"""

import cython
cimport numpy
import numpy

cdef extern from "math.h":
    double floor(float)nogil
    double  fabs(double)nogil


cdef extern from "stdlib.h":
    void free(void * ptr)nogil
    void * calloc(size_t nmemb, size_t size)nogil
    void * malloc(size_t size)nogil

cdef extern from "slist.h":
    ctypedef struct SListEntry:
        pass
    ctypedef struct SListIterator:
        pass
    ctypedef void * SListValue

    void slist_free(SListEntry * slist)
    SListEntry * slist_prepend(SListEntry ** slist, SListValue data)
    SListEntry * slist_append(SListEntry ** slist, SListValue data)
    SListEntry * slist_next(SListEntry * listentry)
    SListValue slist_data(SListEntry * listentry)
    SListEntry * slist_nth_entry(SListEntry * slist, int n)
    SListValue slist_nth_data(SListEntry * slist, int n)
    int slist_length(SListEntry * slist)
    SListValue * slist_to_array(SListEntry * slist)
    int slist_remove_entry(SListEntry ** slist, SListEntry * entry)
#    int slist_remove_data(SListEntry ** slist,
#                          SListEqualFunc callback,
#                          SListValue data)
#    void slist_sort(SListEntry ** slist, SListCompareFunc compare_func)
#    SListEntry * slist_find_data(SListEntry * slist,
#                                SListEqualFunc callback,
#                                SListValue data)
    void slist_iterate(SListEntry ** slist, SListIterator * iter)
    int slist_iter_has_more(SListIterator * iterator)
    SListValue slist_iter_next(SListIterator * iterator)

struct MyType:
    int   position
    float fraction
ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t
ctypedef numpy.float32_t DTYPE_float32_t

cdef class SList:
    """A single chained paires of int/float

    >>> q = Queue()
    >>> q.append(5)
    >>> q.peek()
    5
    >>> q.pop()
    5
    """
    cdef cqueue.Queue * _c_queue
    def __cinit__(self):
        self._c_queue = cqueue.queue_new()
        if self._c_queue is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_queue is not NULL:
            cqueue.queue_free(self._c_queue)

    cpdef append(self, int value):
        if not cqueue.queue_push_tail(self._c_queue,
                                      < void *> value):
            raise MemoryError()



cdef int foo(int n):
    print "starting"
    cdef SListEntry mylist
    cdef MyType mydata
    print mydata
    mydata.position = 5
    mydata.fraction = 0.5
    print mydata
#    SListEntry * slist_append(mylist, SListValue data);
#    print SLIST_INIT(SLIST_HEAD * head)
    print "done"

def call_foo():
    foo(1)
#@cython.cdivision(True)
#cdef float  getBinNr(float x0, float pos0_min, float dpos) nogil:
#    """
#    calculate the bin number for any point 
#    param x0: current position
#    param pos0_min: position minimum
#    param dpos: bin width
#    """
#    return (x0 - pos0_min) / dpos
#
#class rebin1d():
#    """A class that does the re-binning in 1D """
#
#    cdef list * data
#    cdef float max, min
#    cdef long d0_max, d1_max, r
#
#    def __dealloc__(self):
#        free(self.data)
#
#    def __init__(self,
#                 numpy.ndarray pos0 not None,
#                 numpy.ndarray delta_pos0 not None,
#
#                ):
#        assert data.ndim == 2
#        self.d0_max = data.shape[0] - 1
#        self.d1_max = data.shape[1] - 1
#        self.r = data.shape[1]
#        self.max = data.max()
#        self.min = data.min()
#        self.data = < float *> malloc(data.size * sizeof(float))
#        cdef numpy.ndarray[DTYPE_float32_t, ndim = 2] data2 = numpy.ascontiguousarray(data.astype("float32"))
#        memcpy(self.data, data2.data, data.size * sizeof(float))
#
#
#
#@cython.cdivision(True)
#@cython.boundscheck(False)
#@cython.wraparound(False)
#def histoBBox1d(numpy.ndarray weights not None,
#                numpy.ndarray pos0 not None,
#                numpy.ndarray delta_pos0 not None,
#                pos1=None,
#                delta_pos1=None,
#                long bins=100,
#                pos0Range=None,
#                pos1Range=None,
#                float dummy=0.0
#              ):
#    """
#    Calculates histogram of pos0 (tth) weighted by weights
#    
#    Splitting is done on the pixel's bounding box like fit2D
#    
#    @param weights: array with intensities
#    @param pos0: 1D array with pos0: tth or q_vect
#    @param delta_pos0: 1D array with delta pos0: max center-corner distance
#    @param pos1: 1D array with pos1: chi
#    @param delta_pos1: 1D array with max pos1: max center-corner distance, unused ! 
#    @param bins: number of output bins
#    @param pos0Range: minimum and maximum  of the 2th range
#    @param pos1Range: minimum and maximum  of the chi range
#    @param dummy: value for bins without pixels 
#    @return 2theta, I, weighted histogram, unweighted histogram
#    """
#    cdef long  size = weights.size
#    assert pos0.size == size
#    assert delta_pos0.size == size
#    assert  bins > 1
#
#    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = weights.ravel().astype("float64")
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_inf = (pos0.ravel() - delta_pos0.ravel()).astype("float32")
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos0_sup = (pos0.ravel() + delta_pos0.ravel()).astype("float32")
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_inf
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] cpos1_sup
#
#    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outData = numpy.zeros(bins, dtype="float64")
#    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] outCount = numpy.zeros(bins, dtype="float64")
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] outMerge = numpy.zeros(bins, dtype="float32")
#    cdef numpy.ndarray[DTYPE_float32_t, ndim = 1] outPos = numpy.zeros(bins, dtype="float32")
#    cdef float  deltaR, deltaL, deltaA
#    cdef float pos0_min, pos0_max, pos0_maxin, pos1_min, pos1_max, pos1_maxin, min0, max0, fbin0_min, fbin0_max
#    cdef int checkpos1 = 0
#
#    if pos0Range is not None and len(pos0Range) > 1:
#        pos0_min = min(pos0Range)
#        if pos0_min < 0.0:
#            pos0_min = 0.0
#        pos0_maxin = max(pos0Range)
#    else:
#        pos0_min = cpos0_inf.min()
#        pos0_maxin = cpos0_sup.max()
#    pos0_max = pos0_maxin * (1 + numpy.finfo(numpy.float32).eps)
#
#    if pos1Range is not None and len(pos1Range) > 1:
#        assert pos1.size == size
#        assert delta_pos1.size == size
#        checkpos1 = 1
#        cpos1_inf = (pos1.ravel() - delta_pos1.ravel()).astype("float32")
#        cpos1_sup = (pos1.ravel() + delta_pos1.ravel()).astype("float32")
#        pos1_min = min(pos1Range)
#        pos1_maxin = max(pos1Range)
#        pos1_max = pos1_maxin * (1 + numpy.finfo(numpy.float32).eps)
#
#    cdef float dpos = (pos0_max - pos0_min) / (< float > (bins))
#    cdef long   bin = 0
#    cdef long   i, idx
#    cdef long   bin0_max, bin0_min
#    cdef double epsilon = 1e-10
#
#    with nogil:
#        for i in range(bins):
#                outPos[i] = pos0_min + (0.5 +< float > i) * dpos
#
#        for idx in range(size):
#            data = < double > cdata[idx]
#            min0 = cpos0_inf[idx]
#            max0 = cpos0_sup[idx]
#            if checkpos1:
#                if (cpos1_inf[idx] < pos1_min) or (cpos1_sup[idx] > pos1_max):
#                    continue
#
#            fbin0_min = getBinNr(min0, pos0_min, dpos)
#            fbin0_max = getBinNr(max0, pos0_min, dpos)
#            bin0_min = < long > floor(fbin0_min)
#            bin0_max = < long > floor(fbin0_max)
#
#            if bin0_min == bin0_max:
#                #All pixel is within a single bin
#                outCount[bin0_min] += < double > 1.0
#                outData[bin0_min] += < double > data
#
#            else: #we have pixel spliting.
#                deltaA = 1.0 / (fbin0_max - fbin0_min)
#
#                deltaL = < float > (bin0_min + 1) - fbin0_min
#                deltaR = fbin0_max - (< float > bin0_max)
#
#                outCount[bin0_min] += < double > deltaA * deltaL
#                outData[bin0_min] += < double > data * deltaA * deltaL
#
#                outCount[bin0_max] += < double > deltaA * deltaR
#                outData[bin0_max] += < double > data * deltaA * deltaR
#
#                if bin0_min + 1 < bin0_max:
#                    for i in range(bin0_min + 1, bin0_max):
#                        outCount[i] += < double > deltaA
#                        outData[i] += data * < double > deltaA
#
#        for i in range(bins):
#                if outCount[i] > epsilon:
#                    outMerge[i] = < float > (outData[i] / outCount[i])
#                else:
#                    outMerge[i] = dummy
#
#    return  outPos, outMerge, outData, outCount
#
#
#

