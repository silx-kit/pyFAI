#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

__doc__ = """
Convertion between sparse matrix representations
"""
__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "20141020"
__status__ = "stable"
__license__ = "GPLv3+"


import cython
cimport numpy
import numpy


def LUT_to_CSR(lut):
    """
    Convertion between sparse matrix representations
    """
    cdef numpy.uint32_t nrow, ncol, nelt = 0
    ncol = lut.shape[-1]
    nrow = lut.shape[0]
    cdef:
        int[:, :] idx = lut.idx
        float[:, :] coef = lut.coef
        numpy.ndarray[numpy.float32_t, ndim = 1] data = numpy.zeros(nrow * ncol, numpy.float32)
        numpy.ndarray[numpy.uint32_t, ndim = 1]  indices = numpy.zeros(nrow * ncol, numpy.uint32)
        numpy.ndarray[numpy.uint32_t, ndim = 1] indptr = numpy.zeros(nrow + 1, numpy.uint32)
        numpy.uint32_t i, j
    for i in range(nrow):
        indptr[i] = nelt
        for j in range(ncol):
            if coef[i, j] <= 0.0:
                break
            else:
                data[nelt] = coef[i, j]
                indices[nelt] = idx[i, j]
                nelt += 1
    indptr[nrow] = nelt
    return data[:nelt], indices[:nelt], indptr