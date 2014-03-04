"""Implementation of a separable 2D convolution"""
import cython
import numpy
cimport numpy
from cython.parallel import prange

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def horizontal_convolution(float[:,:] img, float[:] filter):
    """
    Implements a 1D horizontal convolution with a filter

    @param img: input image
    @param filter: 1D array with the coeficients of the array
    @return: array of the same shape as image with
    """
    cdef int FILTER_SIZE, HALF_FILTER_SIZE
    cdef int IMAGE_H,IMAGE_W,
    cdef int x, y, pos, fIndex, newpos, c
    cdef float sum


    FILTER_SIZE =  filter.shape[0]
    if FILTER_SIZE % 2 == 1:
        HALF_FILTER_SIZE = (FILTER_SIZE)/2
    else:
        HALF_FILTER_SIZE = (FILTER_SIZE+1)/2

    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    cdef numpy.ndarray[numpy.float32_t, ndim=2] output = numpy.zeros((IMAGE_H,IMAGE_W), dtype=numpy.float32)
    for y in prange(IMAGE_H, nogil=True):
        for x in range(IMAGE_W):
            fIndex = 0
            sum = 0.0
            for c in range(-HALF_FILTER_SIZE, FILTER_SIZE-HALF_FILTER_SIZE):
                newpos = x + c
                if newpos < 0:
                    newpos = - newpos - 1
                elif newpos >= IMAGE_W:
                    newpos = 2*IMAGE_W - newpos - 1
                sum += img[y,newpos] * filter[fIndex]
                fIndex += 1;
            output[y,x]+=sum
    return output



"""


__kernel void vertical_convolution(
    const __global float * input,
    __global float * output,
    __constant float * filter __attribute__((max_constant_size(MAX_CONST_SIZE))),
    int FILTER_SIZE,
    int IMAGE_W,
    int IMAGE_H
)
{

    int gid1 = (int) get_global_id(1);
    int gid0 = (int) get_global_id(0);


    if (gid1 < IMAGE_H && gid0 < IMAGE_W) {

        int HALF_FILTER_SIZE = (FILTER_SIZE % 2 == 1 ? (FILTER_SIZE)/2 : (FILTER_SIZE+1)/2);

//        int pos = gid0 * IMAGE_W + gid1;
        int pos = gid1 * IMAGE_W + gid0;
        int fIndex = 0;
        float sum = 0.0f;
        int r = 0,newpos=0;
        int debug=0;

        for (r = -HALF_FILTER_SIZE ; r < FILTER_SIZE-HALF_FILTER_SIZE ; r++) {
            newpos = pos + r * (IMAGE_W);

            if (gid1+r < 0) {
                newpos = gid0 -(r+1)*IMAGE_W - gid1*IMAGE_W;
                //debug=1;
            }
            else if (gid1+r > IMAGE_H -1) {
                newpos= (IMAGE_H-1)*IMAGE_W + gid0 + (IMAGE_H - r)*IMAGE_W - gid1*IMAGE_W;
            }
            sum += input[ newpos ] * filter[ fIndex   ];
            fIndex += 1;

        }
        output[pos]=sum;
        if (debug == 1) output[pos]=0;
    }
}
"""