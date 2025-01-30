# coding: utf-8
#
#    Project: Azimuthal integration using single threaded CSC integrators
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2022-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""Common CSR integrator"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "24/04/2024"
__status__ = "stable"
__license__ = "MIT"

import numpy

from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel


cdef class CscIntegrator(object):
    """Abstract class which implements only the integrator...

    Now uses CSC (Compressed Sparse Column) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Line index position for the data (same size as data)
    * indptr: Column pointer indicates the start of a given Column. len ncol+1, or size of the image+1.

    Nota: nnz = indptr[-1]+1 = len(indices) = len(data)
    """
    cdef:
        readonly index_t input_size, output_size, nnz
        public data_t empty
        readonly data_t[::1] _data
        readonly index_t[::1] _indices, _indptr

    def __init__(self,
                  tuple lut,
                  int input_size,
                  int output_size,
                  data_t empty=0.0):

        """Constructor for a CSC generic integrator

        :param lut: Sparse matrix in CSC format, tuple of 3 arrays with (data, line_indices, col_indptr)
        :param size: input image size
        :param empty: value for empty pixels
        """
        self.empty = empty
        assert input_size >= len(lut[2])-1, "image size is OK"
        assert len(lut) == 3, "Sparse matrix is expected as 3-tuple CSC with (data, line_indices, col_indptr)"
        assert len(lut[1]) == len(lut[0]),  "Sparse matrix in CSC format is expected to have len(data) == len(indices) is expected as 3-tuple CSR with (data, indices, indptr)"
        self._data = numpy.ascontiguousarray(lut[0], dtype=data_d)
        self._indices = numpy.ascontiguousarray(lut[1], dtype=numpy.int32)
        self._indptr = numpy.ascontiguousarray(lut[2], dtype=numpy.int32)
        self.nnz = len(lut[1])
        self.output_size = output_size
        self.input_size = input_size

    def __dealloc__(self):
        self._data = None
        self._indices = None
        self._indptr = None
        self.empty = 0
        self.input_size = 0
        self.output_size = 0
        self.nnz = 0

    @property
    def data(self):
        return numpy.asarray(self._data)
    @property
    def indices(self):
        return numpy.asarray(self._indices)
    @property
    def indptr(self):
        return numpy.asarray(self._indptr)


    def integrate_ng(self,
                     weights,
                     variance=None,
                     error_model=ErrorModel.NO,
                     dummy=None,
                     delta_dummy=None,
                     dark=None,
                     flat=None,
                     solidangle=None,
                     polarization=None,
                     absorption=None,
                     data_t normalization_factor=1.0,
                     bint weighted_average=True,
                     ):
        """
        Actually perform the integration which in this case consists of:
         * Calculate the signal, variance and the normalization parts
         * Perform the integration which is here a matrix-vector product

        :param weights: input image
        :type weights: ndarray
        :param variance: the variance associate to the image
        :type variance: ndarray
        :param error_model: enum ErrorModel
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
        :param absorption: Apparent efficiency of a pixel due to parallax effect
        :type absorption: ndarray
        :param normalization_factor: divide the valid result by this value
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average WIP
        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: Integrate1dtpl 4-named-tuple of ndarrays
        """
        cdef:
            index_t idx, start, stop, j, bin_idx
            acc_t coef, w
            data_t empty, cdummy, cddummy
            data_t[::1] cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
            acc_t[::1] sum_sig = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_var = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm_sq = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.output_size, dtype=acc_d)
            data_t[::1] merged = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] std = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] sem = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] cvariance, cdark, cflat, cpolarization, csolidangle, cabsorption
            bint do_azimuthal_variance = error_model is ErrorModel.AZIMUTHAL
            int c_error_model
            bint is_valid, do_dark, do_flat, do_polarization, check_dummy, do_solidangle, do_absorption
            preproc_t value
            acc_t delta1, delta2, b, omega_A, omega_B, omega2_A, omega2_B, omega_AB

        assert weights.size == self.input_size, "weights size"
        empty = dummy if dummy is not None else self.empty


        if (dummy is not None) and (delta_dummy is not None):
            check_dummy = True
            cdummy = <data_t> float(dummy)
            cddummy = <data_t> float(delta_dummy)
        elif (dummy is not None):
            check_dummy = True
            cdummy = <data_t> float(dummy)
            cddummy = 0.0
        else:
            check_dummy = False
            cdummy = <data_t> float(empty)
            cddummy = 0.0

        if variance is not None:
            assert variance.size == self.input_size, "variance size"
            cvariance = numpy.ascontiguousarray(variance.ravel(), dtype=data_d)
            error_model = max(error_model, 1)

        if dark is not None:
            do_dark = True
            assert dark.size == self.input_size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
        else:
            do_dark = False
        if flat is not None:
            do_flat = True
            assert flat.size == self.input_size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
        else:
            do_flat = False
        if polarization is not None:
            do_polarization = True
            assert polarization.size == self.input_size, "polarization array size"
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)
        else:
            do_polarization = False
        if solidangle is not None:
            do_solidangle = True
            assert solidangle.size == self.input_size, "Solid angle array size"
            csolidangle = numpy.ascontiguousarray(solidangle.ravel(), dtype=data_d)
        else:
            do_solidangle = False
        if absorption is not None:
            do_absorption = True
            assert absorption.size == self.input_size, "absorption array size"
            cabsorption = numpy.ascontiguousarray(absorption.ravel(), dtype=data_d)
        else:
            do_absorption = False
        c_error_model = <int> error_model
        with nogil:
            #first loop for pixel in input image
            for idx in range(self._indptr.shape[0]-1):
                # skip pixel if masked:
                start = self._indptr[idx]
                stop = self._indptr[idx+1]
                if stop == start: # pixel contributes to no bin
                    continue
                is_valid = preproc_value_inplace(&value,
                                                 cdata[idx],
                                                 variance=cvariance[idx] if c_error_model==1 else 0.0,
                                                 dark=cdark[idx] if do_dark else 0.0,
                                                 flat=cflat[idx] if do_flat else 1.0,
                                                 solidangle=csolidangle[idx] if do_solidangle else 1.0,
                                                 polarization=cpolarization[idx] if do_polarization else 1.0,
                                                 absorption=cabsorption[idx] if do_absorption else 1.0,
                                                 mask=0, #masked pixel have already been treated
                                                 dummy=cdummy,
                                                 delta_dummy=cddummy,
                                                 check_dummy=check_dummy,
                                                 normalization_factor=normalization_factor,
                                                 dark_variance=0.0,
                                                 error_model=c_error_model,
                                                 apply_normalization=not weighted_average
                                                 )
                if not is_valid:
                    continue
                for j in range(start, stop):
                    bin_idx = self._indices[j]
                    coef = self._data[j]
                    w = coef * value.norm
                    sum_count[bin_idx] += coef * value.count
                    if do_azimuthal_variance:
                        if sum_norm_sq[bin_idx] > 0.0:
                            # Inspired from https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                            # Not correct, Inspired by VV_{A+b} = VV_A + ω²·(b-V_A/Ω_A)·(b-V_{A+b}/Ω_{A+b})
                            # Emprically validated against 2-pass implementation in Python/scipy-sparse

                            omega_A = sum_norm[bin_idx]
                            omega_B = w
                            omega2_A = sum_norm_sq[bin_idx]
                            omega2_B = omega_B*omega_B
                            sum_norm[bin_idx] = omega_AB = omega_A + omega_B
                            sum_norm_sq[bin_idx] = omega2_A + omega2_B

                            # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                            b = value.signal / value.norm
                            delta1 = sum_sig[bin_idx]/omega_A - b
                            sum_sig[bin_idx] += coef * value.signal
                            delta2 = sum_sig[bin_idx] / omega_AB - b
                            sum_var[bin_idx] += omega2_B * delta1 * delta2
                        else:
                            sum_sig[bin_idx] = coef * value.signal
                            sum_norm[bin_idx] = w
                            sum_norm_sq[bin_idx] = w * w
                    else:
                        sum_sig[bin_idx] += coef * value.signal
                        sum_norm[bin_idx] += w
                        if c_error_model:
                            sum_norm_sq[bin_idx] += w * w
                            sum_var[bin_idx] += coef * coef * value.variance

            #calulate means from accumulators:
            for bin_idx in range(self.output_size):
                if sum_count[bin_idx]:
                    merged[bin_idx] = sum_sig[bin_idx] / sum_norm[bin_idx]
                    if c_error_model:
                        sem[bin_idx] = sqrt(sum_var[bin_idx]) / sum_norm[bin_idx]
                        std[bin_idx] = sqrt(sum_var[bin_idx] / sum_norm_sq[bin_idx])
                    else:
                        std[bin_idx] = sem[bin_idx] = empty
                else:
                    merged[bin_idx] = empty
                    if c_error_model:
                        std[bin_idx] = sem[bin_idx] = empty

        if self.bin_centers is None:
            # 2D integration case
            return Integrate2dtpl(self.bin_centers0, self.bin_centers1,
                              numpy.asarray(merged).reshape(self.bins).T,
                              numpy.asarray(sem).reshape(self.bins).T if error_model else None,
                              numpy.asarray(sum_sig).reshape(self.bins).T,
                              numpy.asarray(sum_var).reshape(self.bins).T if error_model else None,
                              numpy.asarray(sum_norm).reshape(self.bins).T,
                              numpy.asarray(sum_count).reshape(self.bins).T,
                              numpy.asarray(std).reshape(self.bins).T if error_model else None,
                              numpy.asarray(sem).reshape(self.bins).T if error_model else None,
                              numpy.asarray(sum_norm_sq).reshape(self.bins).T if error_model else None)
        else:
            # 1D integration case: "position intensity error signal variance normalization count std sem norm_sq"
            return Integrate1dtpl(self.bin_centers,
                                  numpy.asarray(merged),numpy.asarray(sem)  if error_model else None,
                                  numpy.asarray(sum_sig),numpy.asarray(sum_var) if error_model else None,
                                  numpy.asarray(sum_norm), numpy.asarray(sum_count),
                                  numpy.asarray(std) if error_model else None,
                                  numpy.asarray(sem) if error_model else None,
                                  numpy.asarray(sum_norm_sq) if error_model else None)
