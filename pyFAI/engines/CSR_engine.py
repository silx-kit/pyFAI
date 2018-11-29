#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.


"""CSR rebinning engine implemented in pure python (with bits of scipy !) 
"""

from __future__ import absolute_import, print_function, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/11/2018"
__status__ = "development"

import logging
logger = logging.getLogger(__name__)
import numpy
from scipy.sparse import csr_matrix
from .preproc import preproc as preproc_np
try:
    from ..ext.preproc import preproc as preproc_cy
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    preproc = preproc_np
else:
    preproc = preproc_cy

from collections import namedtuple

Integrate1dResult = namedtuple("Integrate1dResult", ["bins", "signal", "propagated"])
Integrate2dResult = namedtuple("Integrate2dResult", ["signal", "bins0", "bins1", "propagated"])
Integrate1dWithErrorResult = namedtuple("Integrate1dWithErrorResult", ["bins", "signal", "error", "propagated"])
Integrate2dWithErrorResult = namedtuple("Integrate2dWithErrorResult", ["signal", "error", "bins0", "bins1", "propagated"])


class CsrIntegrator2d(object):
    def __init__(self,
                 size,
                 data=None,
                 indices=None,
                 indptr=None,
                 empty=0.0,
                 bin_centers0=None,
                 bin_centers1=None):
        """Constructor of the abstract class
        
        :param bins: number of output bins 
        :param size: input image size
        :param data: data of the CSR matrix
        :param indices: indices of the CSR matrix
        :param indptr: indices of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param bin_center: position of the bin center
        """
        self.empty = empty
        self.bin_centers0 = bin_centers0
        self.bin_centers1 = bin_centers1
        self.bins = None
        self.size = size
        self._csr = None
        self.lut_size = 0  # actually nnz
        self.data = None
        self.indices = None
        self.indptr = None
        if (data is not None) and (indices is not None) and (indptr is not None):
            self.set_matrix(data, indices, indptr)

    def set_matrix(self, data, indices, indptr):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.lut_size = len(indices)
        self._csr = csr_matrix((data, indices, indptr))
        if (self.bin_centers0 is not None)  and (self.bin_centers1 is not None):
            assert len(self.bin_centers0) * len(self.bin_centers1) == len(indptr) - 1
            self.bins = (len(self.bin_centers0), len(self.bin_centers1))

    def integrate(self,
                  signal,
                  variance=None,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidangle=None,
                  polarization=None,
                  absorption=None,
                  normalization_factor=1.0,
                  ):
        shape = signal.shape
        if variance is None:
            do_variance = False
        else:
            do_variance = True
        prep = preproc(signal,
                       dark=dark,
                       flat=flat,
                       solidangle=solidangle,
                       polarization=polarization,
                       absorption=absorption,
                       mask=None,
                       dummy=dummy,
                       delta_dummy=delta_dummy,
                       normalization_factor=normalization_factor,
                       empty=self.empty,
                       split_result=4,
                       variance=variance,
                       dtype=numpy.float32)
        prep.shape = numpy.prod(shape), -1
        trans = self._csr.dot(prep)
        trans.shape = self.bins + (-1,)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            norm = trans[..., -1]
            intensity = trans[..., 0] / norm
            mask = norm == 0
            intensity[mask] = self.empty
            if do_variance:
                error = numpy.sqrt(trans[..., 1]) / norm
                error[mask] = self.empty

        if do_variance:
            result = Integrate2dWithErrorResult(intensity,
                                                error,
                                                self.bin_centers0,
                                                self.bin_centers1,
                                                trans)
        else:
            result = Integrate2dResult(intensity,
                                       self.bin_centers0,
                                       self.bin_centers1,
                                       trans)
        return result
