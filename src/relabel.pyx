
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20120916"
__status__ = "stable"
__license__ = "GPLv3+"
import cython
import numpy
cimport numpy

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
    cdef numpy.uint32_t[:] clabel = numpy.ascontiguousarray(label.ravel(), dtype=numpy.uint32)
    cdef float[:] cdata = numpy.ascontiguousarray(data.ravel(), dtype=numpy.float32)
    cdef float[:] cblured = numpy.ascontiguousarray(blured.ravel(), dtype=numpy.float32)
    cdef size_t maxLabel = label.max()
    cdef numpy.ndarray[numpy.uint_t, ndim = 1] count = numpy.zeros(maxLabel + 1, dtype=numpy.uint)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] maxData = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] maxBlured = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
    cdef numpy.ndarray[numpy.float32_t, ndim = 1] maxDelta = numpy.zeros(maxLabel + 1, dtype=numpy.float32)
    cdef int s , i, idx
    cdef float d, b
    s = label.size
    assert s == cdata.size
    assert s == cblured.size
    with nogil:
        for i in range(s):
            idx =  clabel[i]
            d =  cdata[i]
            b =  cblured[i]
            count[idx] += 1
            if d > maxData[idx]:
                maxData[idx] = d
                maxDelta[idx] = d - b
            if b > maxBlured[idx]:
                maxBlured[idx] = b
    return count, maxData, maxBlured, maxDelta



