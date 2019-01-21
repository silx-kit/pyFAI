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

"""Common Look-Up table/CSR object creation tools"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "21/01/2019"
__status__ = "stable"
__license__ = "MIT"

include "regrid_common.pxi"

import cython
from cython.parallel import prange
import numpy
cimport numpy as cnumpy

cdef struct lut_t:
    cnumpy.int32_t idx
    cnumpy.float32_t coef

LUT_ITEMSIZE = int(sizeof(lut_t))

# Work around for issue similar to : https://github.com/pandas-dev/pandas/issues/16358
if _numpy_1_12_py2_bug:
    lut_d = numpy.dtype([(b"idx", numpy.int32), (b"coef", numpy.float32)])
else:
    lut_d = numpy.dtype([("idx", numpy.int32), ("coef", numpy.float32)])


class CsrIntegrator2d(object):
    """Abstract class which implements only the integrator...

    Now uses CSR (Compressed Sparse raw) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]
    """
    def __init__(self,
                 int size,
                 cnumpy.float32_t[::1] data=None,
                 cnumpy.int32_t[::1] indices=None,
                 cnumpy.int32_t[::1] indptr=None,
                 data_t empty=0.0,
                 bin_centers0=None,
                 bin_centers1=None):

        """Constructor of the abstract class

        :param bins: number of output bins
        :param size: input image size
        :param data: data of the CSR matrix
        :param indices: indices of the CSR matrix
        :param indptr: indices of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param bin_centers0: position of the bin center along dim0
        :param bin_centers0: position of the bin center along dim1
        """
        self.empty = empty
        self.bin_centers0 = bin_centers0
        self.bin_centers1 = bin_centers1
        self.size = size
        self._csr = None
        self.lut_size = 0  # actually nnz
        self.data = None
        self.indices = None
        self.indptr = None
        if (data is not None) and (indices is not None) and (indptr is not None):
            self.set_matrix(data, indices, indptr)

    def set_matrix(self,
                   cnumpy.float32_t[::1] data not None,
                   cnumpy.int_t[::1] indices not None,
                   cnumpy.int_t[::1] indptr not None):
        """Actually create the CSR_matrix"""
        from scipy.sparse import csr_matrix

        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.lut_size = len(indices)

        self._csr = csr_matrix((data, indices, indptr))
        if (self.bin_centers0 is not None) and (self.bin_centers1 is not None):
            assert len(self.bin_centers0) * len(self.bin_centers1) == len(indptr) - 1
            self.bins = (len(self.bin_centers0), len(self.bin_centers1))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def integrate_legacy(self,
                         weights,
                         dummy=None,
                         delta_dummy=None,
                         dark=None,
                         flat=None,
                         solidAngle=None,
                         polarization=None,
                         double normalization_factor=1.0,
                         int coef_power=1):
        """
        Actually perform the integration which in this case looks more like a matrix-vector product

        :param weights: input image
        :type weights: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param normalization_factor: divide the valid result by this value
        :param coef_power: set to 2 for variance propagation, leave to 1 for mean calculation

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: 4-tuple of ndarrays

        """
        cdef:
            cnumpy.int32_t i = 0, j = 0, idx = 0, bins = self.bins, size = self.size
            acc_t acc_data = 0.0, acc_count = 0.0, epsilon = 1e-10, coef = 0.0
            data_t data = 0.0, cdummy = 0.0, cddummy = 0.0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(self.bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.bins, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(self.bins, dtype=data_d)
            data_t[::1] ccoef = self.data
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
            cnumpy.int32_t[::1] indices = self.indices, indptr = self.indptr
        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)

            if delta_dummy is None:
                cddummy = <data_t> 0.0
            else:
                cddummy = <data_t> float(delta_dummy)
        else:
            do_dummy = False
            cdummy = <data_t> self.empty

        if flat is not None:
            do_flat = True
            assert flat.size == size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
        if dark is not None:
            do_dark = True
            assert dark.size == size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == size, "Solid angle array size"
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=data_d)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == size, "polarization array size"
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)

        if (do_dark + do_flat + do_polarization + do_solidAngle):
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
            cdata = numpy.zeros(size, dtype=data_d)
            if do_dummy:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        # Nota: -= and /= operatore are seen as reduction in cython parallel.
                        if do_dark:
                            data = data - cdark[i]
                        if do_flat:
                            data = data / cflat[i]
                        if do_polarization:
                            data = data / cpolarization[i]
                        if do_solidAngle:
                            data = data / csolidAngle[i]
                        cdata[i] += data
                    else:  # set all dummy_like values to cdummy. simplifies further processing
                        cdata[i] += cdummy
            else:
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if do_dark:
                        data = data - cdark[i]
                    if do_flat:
                        data = data / cflat[i]
                    if do_polarization:
                        data = data / cpolarization[i]
                    if do_solidAngle:
                        data = data / csolidAngle[i]
                    cdata[i] += data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
                cdata = numpy.zeros(size, dtype=data_d)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] += data
                    else:
                        cdata[i] += cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i in prange(bins, nogil=True, schedule="guided"):
            acc_data = 0.0
            acc_count = 0.0
            for j in range(indptr[i], indptr[i + 1]):
                idx = indices[j]
                coef = ccoef[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef

            sum_data[i] += acc_data
            sum_count[i] += acc_count
            if acc_count > epsilon:
                merged[i] += acc_data / acc_count / normalization_factor
            else:
                merged[i] += cdummy
        return (self.bin_centers,
                numpy.asarray(merged),
                numpy.asarray(sum_data),
                numpy.asarray(sum_count))

