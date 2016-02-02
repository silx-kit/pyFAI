# coding: utf-8
__doc__ = """Common cdef constants and functions for preprocessing"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "02/02/2016"
__status__ = "stable"
__license__ = "GPLv3+"

include "numpy_common.pxi"

import cython
cimport numpy
import numpy
from cython cimport floating
from libc.math cimport fabs, M_PI
cdef:
    float pi = <float> M_PI
    float piover2 = <float> (pi * 0.5)
    float onef = <float> 1.0
    float zerof = <float> 1.0
    double EPS32 = (1.0 + numpy.finfo(numpy.float32).eps)


@cython.cdivision(True)
cdef inline floating  get_bin_number(floating x0, floating pos0_min, floating delta) nogil:
    """
    calculate the bin number for any point (as floating)

    @param x0: current position
    @param pos0_min: position minimum
    @param delta: bin width
    @return: bin number as floating point.
    """
    return (x0 - pos0_min) / delta
