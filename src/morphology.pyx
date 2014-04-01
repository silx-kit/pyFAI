__doc__ = """
A few binary morphology operation
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20140401"
__status__ = "stable"
__license__ = "GPL"
import cython
import numpy
cimport numpy

cdef inline numpy.int8_t MIN(numpy.int8_t a, numpy.int8_t b):
    return a if (a<b) else b
cdef inline numpy.int8_t MAX(numpy.int8_t a, numpy.int8_t b):
    return a if (a>b) else b

@cython.boundscheck(False)
@cython.wraparound(False)
def binary_dilation(numpy.int8_t[:,:] image, float radius=1.0):
    """
    Return fast binary morphological dilation of an image.
    
    Morphological dilation sets a pixel at (i,j) to the maximum over all pixels in the neighborhood centered at (i,j). 
    Dilation enlarges bright regions and shrinks dark regions.
    
    @param image : ndarray
    @param radius: float
    @return: ndiamge
    """
    cdef int x, y, i, j, size_x, size_y, px, py, r_int = int(radius), r2_int = int(radius*radius)  
    size_y = image.shape[0]
    size_x = image.shape[1]
    cdef numpy.int8_t val, curr
    cdef numpy.ndarray[numpy.int8_t, ndim=2] result = numpy.empty(dtype=numpy.int8, shape=(size_y, size_x))
    
    for y in range(size_y):
        for x in range(size_x):
            val = image[y,x]
            for j in range(-r_int, r_int+1):
                py = y + j
                if py<0 or py>=size_y: continue
                for i in range(-r_int, r_int+1):
                    px = x + i
                    if px<0 or px>=size_x: continue
                    if i*i + j*j <= r2_int:
                        curr = image[py,px]
                        val = MAX(val, curr)
            result[y,x] = val
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def binary_erosion(numpy.int8_t[:,:] image, float radius=1.0):
    """
    Return fast binary morphological erosion of an image.
    
    Morphological erosion sets a pixel at (i,j) to the minimum over all pixels in the neighborhood centered at (i,j). 
    Erosion shrinks bright regions and enlarges dark regions.
    
    @param image : ndarray
    @param radius: float
    @return: ndiamge
    """
    cdef int x, y, i, j, size_x, size_y, px, py, r_int = int(radius), r2_int = int(radius*radius)
    size_y = image.shape[0]
    size_x = image.shape[1]
    cdef numpy.int8_t val, curr
    cdef numpy.ndarray[numpy.int8_t, ndim=2] result = numpy.empty(dtype=numpy.int8, shape=(size_y, size_x))
    
    for y in range(size_y):
        for x in range(size_x):
            val = image[y,x]
            for j in range(-r_int, r_int+1):
                py = y + j
                if py<0 or py>=size_y: continue
                for i in range(-r_int, r_int+1):
                    px = x + i
                    if px<0 or px>=size_x: continue
                    if i*i + j*j <= r2_int:
                        curr = image[py,px]
                        val = MIN(val, curr)
            result[y,x] = val
    return result

