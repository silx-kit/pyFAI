# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Common Look-Up table common object creation tools"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "26/06/2020"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"

import cython
from cython.parallel import prange
import numpy
cimport numpy as cnumpy

from .preproc import preproc
from ..containers import Integrate1dtpl



cdef class LutIntegrator(object):
    """Abstract class which implements only the integrator...

    Now uses LUT format with main attributes:
    * width: width of the LUT
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]+1 = len(indices) = len(data)
    """
    cdef:
        readonly index_t input_size, output_size, nnz
        readonly data_t empty
        readonly lut_t[:, ::1] _lut

    def __init__(self,
                 lut_t[:, ::1] lut,
                 int image_size,
                 data_t empty=0.0):

        """Constructor for a CSR generic integrator

        :param lut: Sparse matrix in CSR format, tuple of 3 arrays with (data, indices, indptr)
        :param size: input image size
        :param empty: value for empty pixels
        """
        self.empty = empty
        self.input_size = image_size
        self.output_size = len(lut)
        self._lut = lut
    
    def __dealloc__(self):
        self._lut = None
        self.empty = 0
        self.input_size = 0
        self.output_size = 0 
        self.nnz = 0
