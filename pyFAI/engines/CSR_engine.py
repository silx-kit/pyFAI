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
__date__ = "24/04/2019"
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

from ..containers import Integrate1dtpl, Integrate2dtpl


class CSRIntegrator(object):
    def __init__(self,
                 size,
                 data=None,
                 indices=None,
                 indptr=None,
                 empty=0.0):
        """Constructor of the abstract class
        
        :param size: input image size        
        :param data: data of the CSR matrix
        :param indices: indices of the CSR matrix
        :param indptr: indices of the start of line in the CSR matrix
        :param empty: value for empty pixels
        """
        self.size = size
        self.empty = empty
        self.bins = None
        self._csr = None
        self._csr2 = None
        self.lut_size = 0  # actually nnz
        self.data = None
        self.indices = None
        self.indptr = None
        if (data is not None) and (indices is not None) and (indptr is not None):
            self.set_matrix(data, indices, indptr)

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content
        
        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.lut_size = len(indices)
        self._csr = csr_matrix((data, indices, indptr))
        self._csr2 = csr_matrix((data * data, indices, indptr))  # contains the coef squared, used for variance propagation
        self.bins = len(indptr) - 1

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
        """Actually perform the CSR matrix multiplication after preprocessing.
        
        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :return: the preprocessed data integrated as array nbins x 4 which contains:
                    regrouped signal, variance, normalization and pixel count 

        Nota: all normalizations are grouped in the preprocessing step.
        """
        shape = signal.shape
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
        res = self._csr.dot(prep)
        if variance is not None:
            res[:, 1] = self._csr2.dot(prep[:, 1])
        return res


class CsrIntegrator1d(CSRIntegrator):
    def __init__(self,
                 size,
                 data=None,
                 indices=None,
                 indptr=None,
                 empty=0.0,
                 bin_centers=None,
                 ):
        """Constructor of the abstract class for 1D integration
        
        :param data: data of the CSR matrix
        :param indices: indices of the CSR matrix
        :param indptr: indices of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param bin_center: position of the bin center
        
        Nota: bins are deduced from bin_centers 

        """
        self.bin_centers = bin_centers
        CSRIntegrator.__init__(self, size, data, indices, indptr, empty)

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content
        
        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""

        CSRIntegrator.set_matrix(self, data, indices, indptr)
        assert len(self.bin_centers) == self.bins

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
        """Actually perform the 1D integration 
        
        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :return: Integrate1dResult or Integrate1dWithErrorResult object depending on variance 
        
        """
        if variance is None:
            do_variance = False
        else:
            do_variance = True
        trans = CSRIntegrator.integrate(self, signal, variance, dummy, delta_dummy,
                                        dark, flat, solidangle, polarization,
                                        absorption, normalization_factor)
        signal = trans[:, 0]
        variance = trans[:, 1]
        normalization = trans[:, 2]
        count = trans[..., -1]  # should be 3
        mask = (normalization == 0)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            intensity = signal / normalization
            intensity[mask] = self.empty
            if do_variance:
                error = numpy.sqrt(variance) / normalization
                error[mask] = self.empty
            else:
                variance = error = None
        return Integrate1dtpl(self.bin_centers,
                              intensity, error,
                              signal, variance, normalization, count)


class CsrIntegrator2d(CSRIntegrator):
    def __init__(self,
                 size,
                 data=None,
                 indices=None,
                 indptr=None,
                 empty=0.0,
                 bin_centers0=None,
                 bin_centers1=None):
        """Constructor of the abstract class for 2D integration
        
        :param size: input image size
        :param data: data of the CSR matrix
        :param indices: indices of the CSR matrix
        :param indptr: indices of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param bin_center: position of the bin center

        Nota: bins are deduced from bin_centers0, bin_centers1 
    
        """
        self.bin_centers0 = bin_centers0
        self.bin_centers1 = bin_centers1
        CSRIntegrator.__init__(self, size, data, indices, indptr, empty)

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content
        
        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""

        CSRIntegrator.set_matrix(self, data, indices, indptr)
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
                  normalization_factor=1.0):
        """Actually perform the 2D integration 
        
        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :return: Integrate2dResult or Integrate2dWithErrorResult object depending is variance is provided 
        
        """
        if variance is None:
            do_variance = False
        else:
            do_variance = True
        trans = CSRIntegrator.integrate(self, signal, variance, dummy, delta_dummy,
                                        dark, flat, solidangle, polarization,
                                        absorption, normalization_factor)
        trans.shape = self.bins + (-1,)

        signal = trans[..., 0]
        variance = trans[..., 1]
        normalization = trans[..., 2]
        count = trans[..., -1]  # should be 3
        mask = (normalization == 0)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            intensity = signal / normalization
            intensity[mask] = self.empty
            if do_variance:
                error = numpy.sqrt(variance) / normalization
                error[mask] = self.empty
            else:
                variance = error = None
        return Integrate2dtpl(self.bin_centers0, self.bin_centers1,
                              intensity, error,
                              signal, variance, normalization, count)

