
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20110923"
__status__ = "stable"
__license__ = "GPLv3+"
import cython
cimport numpy
import numpy

ctypedef numpy.int64_t DTYPE_int64_t
ctypedef numpy.float64_t DTYPE_float64_t
@cython.boundscheck(False)
@cython.wraparound(False)
def countThem(numpy.ndarray label not None, \
              numpy.ndarray data not None, \
              numpy.ndarray blured not None):
    """
    @param label: 2D array containing labeled zones
    @param data: 2D array containing the raw data
    @param blured: 2D array containing the blured data
    @return: 2D arrays containing: 
        * count pixels in labelled zone: label == index).sum() 
        * max of data in that zone:      data[label == index].max()
        * max of blured in that zone:    blured[label == index].max()
        * data-blured where data is max.   
    """
    cdef int maxLabel = label.max()
    cdef numpy.ndarray[DTYPE_int64_t, ndim = 1] clabel = label.astype("int64").flatten()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cdata = data.astype("float64").flatten()
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] cblured = blured.astype("float64").flatten()
    cdef numpy.ndarray[DTYPE_int64_t, ndim = 1] count = numpy.zeros(maxLabel + 1, dtype=numpy.int64)
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] maxData = numpy.zeros(maxLabel + 1, dtype=numpy.float64)
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] maxBlured = numpy.zeros(maxLabel + 1, dtype=numpy.float64)
    cdef numpy.ndarray[DTYPE_float64_t, ndim = 1] maxDelta = numpy.zeros(maxLabel + 1, dtype=numpy.float64)
    cdef long s , i, idx
    cdef double d, b
    s = < long > label.size
    assert s == cdata.size
    assert s == cblured.size
    for i in range(s):
        idx = < long > clabel[i]
        d = < double > cdata[i]
        b = < double > cblured[i]
        count[idx] += 1
        if d > maxData[idx]:
            maxData[idx] = d
            maxDelta[idx] = d - b
        if b > maxBlured[idx]:
            maxBlured[idx] = b
    return count, maxData, maxBlured, maxDelta



