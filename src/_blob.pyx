"""
Some Cythonized function for blob detection function
"""
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
