'''
Created on Nov 4, 2013

@author: zubair, Jerome Kieffer
'''

import numpy
cimport numpy
import cython
cimport cython
from cython cimport view
from cython.parallel import prange

#copied bisplev function from fitpack.bisplev
def bisplev(x,y,tck,dx=0,dy=0):
    """
    Evaluate a bivariate B-spline and its derivatives.

    Return a rank-2 array of spline function values (or spline derivative
    values) at points given by the cross-product of the rank-1 arrays x and
    y.  In special cases, return an array or just a float if either x or y or
    both are floats.  Based on BISPEV from FITPACK.

    Parameters
    ----------
    x, y : ndarray
        Rank-1 arrays specifying the domain over which to evaluate the
        spline or its derivative.
    tck : tuple
        A sequence of length 5 returned by `bisplrep` containing the knot
        locations, the coefficients, and the degree of the spline:
        [tx, ty, c, kx, ky].
    dx, dy : int, optional
        The orders of the partial derivatives in `x` and `y` respectively.
        This version does bot implement derivatives.

    Returns
    -------
    vals : ndarray
        The B-spline or its derivative evaluated over the set formed by
        the cross-product of `x` and `y`.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
        See `bisplrep` to generate the `tck` representation.

    References
    ----------
    .. [1] Dierckx P. : An algorithm for surface fitting
       with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P. : An algorithm for surface fitting
       with spline functions
       report tw50, Dept. Computer Science,K.U.Leuven, 1980.
    .. [3] Dierckx P. : Curve and surface fitting with splines,
       Monographs on Numerical Analysis, Oxford University Press, 1993.

    """
    cdef  int kx,ky
    cdef float[:] tx, ty, c, cy_x, cy_y
    tx = numpy.ascontiguousarray(tck[0], dtype=numpy.float32)
    ty = numpy.ascontiguousarray(tck[1], dtype=numpy.float32)
    c  = numpy.ascontiguousarray(tck[2], dtype=numpy.float32)
    kx = tck[3]
    ky = tck[4]

    if not (0<=dx<kx):
        raise ValueError("0 <= dx = %d < kx = %d must hold" % (dx,kx))
    if not (0<=dy<ky):
        raise ValueError("0 <= dy = %d < ky = %d must hold" % (dy,ky))
    if (len(x.shape) != 1) or (len(y.shape) != 1):
        raise ValueError("First two entries should be rank-1 arrays.")

    cy_x = numpy.ascontiguousarray(x, dtype=numpy.float32)
    cy_y = numpy.ascontiguousarray(y, dtype=numpy.float32)

    z = cy_bispev(tx, ty, c, kx, ky, cy_x, cy_y)
    z.shape = len(y),len(x)
    return z.T #this is a trick as we transpose again afterwards to retrieve a memory-contiguous object
#    if len(z)>1:
#        return z
#    elif len(z[0])>1:
#        return z[0]
#    else:
#        return z[0][0]
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void fpbspl(float[:]t,
                 int n,
                 int k,
                 float x,
                 int l,
                 float[:] h,
                 float[:] hh) nogil:
    """
    subroutine fpbspl evaluates the (k+1) non-zero b-splines of
    degree k at t(l) <= x < t(l+1) using the stable recurrence
    relation of de boor and cox.
    """
    cdef int i, j, li, lj#, n=t.shape 
    cdef float f

    h[0] = 1.00
    for j in range(1,k+1):  #adding +1 in index
        for i in range(j):  
            hh[i] = h[i]    
        h[0] = 0.00
        for i in range(j): 
            f = hh[i]/(t[l+i]-t[l+i-j]) 
            h[i] = h[i] + f*(t[l+i]-x)
            h[i+1] = f*(x-t[l+i-j])     

cdef void init_w(float[:]t, int k, float[:]x, numpy.int32_t[:]lx, float[:,:]w):
    cdef int i, l, l1, n=t.size, m=x.size
    cdef float arg  
    cdef float tb = t[k]
    cdef float te = t[n-k-1]
    cdef float[:] h = view.array(shape=(6,), itemsize=sizeof(float), format="f")
    cdef float[:] hh = view.array(shape=(5,), itemsize=sizeof(float), format="f")

    l = k+1
    l1 = l+1
    for i in range(m): 
        arg = x[i]      
        if arg < tb: 
            arg = tb
        if arg > te: 
            arg = te
        while not( arg < t[l] or l == (n-k-1)):    
            l = l1
            l1 = l + 1
        fpbspl(t, n, k, arg, l, h, hh)

        lx[i] = l - k - 1
        for j in range(k + 1):
            w[i,j] = h[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef cy_bispev(float[:] tx,
                        float[:] ty,
                        float[:] c,
                        int kx,
                        int ky,
                        float[:] x,
                        float[:] y):
    cdef int nx = tx.size
    cdef int ny = ty.size
    cdef int mx = x.size
    cdef int my = y.size

    cdef int kx1 = kx+1
    cdef int ky1 = ky+1

    cdef int nkx1 = nx-kx1
    cdef int nky1 = ny-ky1 

    #initializing scratch space
    cdef float[:,:] wx = view.array(shape=(mx,kx1), itemsize=sizeof(float), format="f")
    cdef float[:,:] wy = view.array(shape=(my,ky1), itemsize=sizeof(float), format="f")

    cdef numpy.int32_t[:] lx = view.array(shape=(mx,), itemsize=sizeof(numpy.int32_t), format="i")
    cdef numpy.int32_t[:] ly = view.array(shape=(my,), itemsize=sizeof(numpy.int32_t), format="i")

    cdef int i, j, m, i1, l2, j1, size_z = mx*my

    # initializing z and h
    cdef numpy.ndarray[numpy.float32_t, ndim=1] z = numpy.zeros(size_z, numpy.float32)
    cdef float arg, sp, err, tmp, a

    init_w(tx, kx, x, lx, wx)
    init_w(ty, ky, y, ly, wy)

    with nogil:
        for j in prange(my):
            for i in range(mx):
                sp = 0.0
                err = 0.0
                for i1 in range(kx1):
                    for j1 in range(ky1):
                        # Implements Kahan summation
                        l2 = lx[i] * nky1 + ly[j] + i1 * nky1 + j1
                        a = c[l2] * wx[i,i1] * wy[j,j1] - err
                        tmp = sp + a
                        err = (tmp - sp) - a
                        sp = tmp
                z[j*mx + i] += sp
    return z

