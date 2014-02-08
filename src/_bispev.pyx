'''
Created on Nov 4, 2013

@author: zubair, Jerome Kieffer
'''

import numpy
cimport numpy
import cython
cimport cython
#from cython cimport view


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
    z.shape = len(x),len(y)
    return z
#    if len(z)>1:
#        return z
#    elif len(z[0])>1:
#        return z[0]
#    else:
#        return z[0][0]

def fpbspl(float[:]t,
            int n,
            int k,
            float x,
            int l,
            float[:] h):
    """
    subroutine fpbspl evaluates the (k+1) non-zero b-splines of
    degree k at t(l) <= x < t(l+1) using the stable recurrence
    relation of de boor and cox.
    """
    h[0] = 1.00
    cdef float[:] hh = numpy.zeros(5,dtype=numpy.float32)
    cdef int i, j, li, lj
    cdef float f
    #hh =
    for j in range(,k):
        for i in range(j):
            hh[i] = h[i+1]

        h[0] = 0.00
        for i in range(j):
            li = l+j
            lj = li-j+1
            f = hh[i]/(t[li]-t[lj])
            h[i] = h[i-1] + f*(t[li]-x)
            h[i+1] = f*(x-t[lj])

#    one = 1.f;
#    h__[1] = one;
#    i__1 = *k;
#    for (j = 1; j <= i__1; ++j) {
#        i__2 = j;
#        for (i__ = 1; i__ <= i__2; ++i__) {
#            hh[i__ - 1] = h__[i__];
#/* L10: */
#        }
#        h__[1] = 0.f;
#        i__2 = j;
#        for (i__ = 1; i__ <= i__2; ++i__) {
#            li = *l + i__;
#            lj = li - j;
#            f = hh[i__ - 1] / (t[li] - t[lj]);
#            h__[i__] += f * (t[li] - *x);
#            h__[i__ + 1] = f * (*x - t[lj]);
#/* L20: */
#        }
#    }
#    return 0;


def cy_bispev(float[:] tx,
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
    cdef float tb = tx[kx]
    cdef float te = tx[nkx1]

    cdef int l = kx1
    cdef int l1 = l+1


    #initializing scratch space
    cdef float[:,:] wx = numpy.zeros((mx,kx1), numpy.float32)
    cdef float[:,:] wy = numpy.zeros((my,ky1), numpy.float32)

    cdef numpy.int32_t[:] lx = numpy.zeros(mx, numpy.int32)
    cdef numpy.int32_t[:] ly = numpy.zeros(my, numpy.int32)

    cdef int i, j, m, i1, nky1, l2, size_z = mx*my

    # initializing z and h
    cdef float[:] z = numpy.zeros(size_z, numpy.float32)
    cdef float[:] h = numpy.zeros(6, numpy.float32)
    cdef float arg

    for i in range(mx):
        arg = x[i]
        if arg < tb: arg = tb
        if arg > te: arg = te
        while not( arg < tx[l1-1] or l == nkx1):
            l = l1
            l1 = l + 1

        h = fpbspl(tx,nx,kx,arg,l,h)

        lx[i] = l - kx1
        for j in range(kx1):
            wx[i,j] = h[j]

    ky1 = ky + 1
    nky1 = ny - ky1
    tb = ty[ky1]
    te = ty[nky1]
    l = ky1
    l1 = l+1

    for i in range(my):
        arg = y[i]
        if arg < tb: arg = tb
        if arg > te: arg = te
        while not( arg < ty[l] or l == nky1):
            l += 1

        h = fpbspl(ty,ny,ky,arg,l,h)

        ly[i] = l - ky1
        for j in range(ky1):
            wy[i,j] = h[j]



    m = -1
    for i in range(mx):
        l = lx[i] * nky1
        for i1 in range(kx1):
            h[i1] = wx[i,i1]
        for j in range(my):
            l1 = l + ly[j]
            sp = 0
            for i1 in range(kx1):
                l2 = l1
                for j1 in range(ky1):
                    l2 += 1
                    sp = sp + c[l2-1] * h[i1] * wy[j,j1]
                l1 = l1 + nky1
            m += 1
            z[m] = sp

    return z


