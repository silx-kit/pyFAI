#Cython module to reconstruct the masked values of an image
import numpy
cimport numpy

def reconstruct(numpy ndarray data not None, numpy ndarray mask=None,int dummy=None, int delta_dummy=None):
    assert self.data.ndim==2
    cdef size_t d0=data.shape[0]
    cdef size_t d1=data.shape[1]
    cdef cdata[:,:] = numpy.ascontiguousarray(data, dtype=numpy.float32)
    if mask is not None:
        mask = numpy.ascontiguousarray(mask, dtype=numpy.int8)
    else:
        mask = numpy.zeros(self.data.shape,dtype=numpy.int8)
    if dummy is not None:
        if delta_dummy is None:
            mask+=(data==dummy)
        else:
            self.mask+=(abs(self.data-dummy)<delta_dummy)
        mask = mask.astype(numpy.int8)
    assert d0==mask.shape[0]
    assert d1==mask.shape[1]

    out = numpy.zeros((d0,d1),dtype=numpy.float32)
    out+=data
    out[mask]=0

    cdef size_t[:] masked = numpy.where(mask)[0]
    cdef numpy.int8t cmask = mask

    cdef size_t p0,p1,i,l
    cdef float sum,count
    for i in  masked:
        p0 = i//d1
        p1=i%d1
        out[p0,p1] += processPoint(data,mask,p0,p1,d0,d1)
    return out

