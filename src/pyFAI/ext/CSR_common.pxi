# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2024 European Synchrotron Radiation Facility, Grenoble, France
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
__date__ = "05/12/2024"
__status__ = "stable"
__license__ = "MIT"


from libcpp cimport bool
from libcpp.algorithm cimport sort
from cython cimport floating

import os
import cython
from cython.parallel import prange
import numpy

from .preproc import preproc
from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel

# cdef Py_ssize_t MAX_THREADS = 8
# try:
#     MAX_THREADS = min(MAX_THREADS, len(os.sched_getaffinity(os.getpid()))) # Limit to the actual number of threads
# except Exception:
#     MAX_THREADS = min(MAX_THREADS, os.cpu_count() or 1)


cdef struct float4_t:
    float s0
    float s1
    float s2
    float s3
float4_d = numpy.dtype([('s0','f4'),('s1','f4'),('s2','f4'),('s3','f4')])

cdef inline bool cmp(float4_t a, float4_t b) noexcept nogil:
    return True if a.s0<b.s0 else False

cdef inline void sort_float4(float4_t[::1] ary) noexcept nogil:
    "Sort in place of an array of float4 along first element (s0)"
    cdef:
        int size
    size = ary.shape[0]
    sort(&ary[0], &ary[size-1]+1, cmp)


cdef class CsrIntegrator(object):
    """Abstract class which implements only the integrator...

    Now uses CSR (Compressed Sparse raw) with main attributes:
    * nnz: number of non zero elements
    * data: coefficient of the matrix in a 1D vector of float32
    * indices: Column index position for the data (same size as
    * indptr: row pointer indicates the start of a given row. len nrow+1

    Nota: nnz = indptr[-1]+1 = len(indices) = len(data)
    """
    cdef:
        readonly index_t input_size, output_size, nnz
        public data_t empty
        readonly data_t[::1] _data
        readonly index_t[::1] _indices, _indptr
        readonly data_t[:, ::1] preprocessed

    def __init__(self,
                  tuple lut,
                  int image_size,
                  data_t empty=0.0):

        """Constructor for a CSR generic integrator

        :param lut: Sparse matrix in CSR format, tuple of 3 arrays with (data, indices, indptr)
        :param size: input image size
        :param empty: value for empty pixels
        """
        self.empty = empty
        self.input_size = image_size
        self.preprocessed = numpy.empty((image_size, 4), dtype=data_d)
        assert len(lut) == 3, "Sparse matrix is expected as 3-tuple CSR with (data, indices, indptr)"
        assert len(lut[1]) == len(lut[0]),  "Sparse matrix in CSR format is expected to have len(data) == len(indices) is expected as 3-tuple CSR with (data, indices, indptr)"
        self._data = numpy.ascontiguousarray(lut[0], dtype=data_d)
        self._indices = numpy.ascontiguousarray(lut[1], dtype=numpy.int32)
        self._indptr = numpy.ascontiguousarray(lut[2], dtype=numpy.int32)
        self.nnz = len(lut[1])
        self.output_size = len(lut[2])-1

    def __dealloc__(self):
        self._data = None
        self._indices = None
        self._indpts = None
        self.preprocessed = None
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

        Deprecated version !

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
            index_t i = 0, j = 0, idx = 0
            acc_t acc_data = 0.0, acc_count = 0.0, epsilon = 1e-10, coef = 0.0
            data_t data = 0.0, cdummy = 0.0, cddummy = 0.0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.output_size, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(self.output_size, dtype=data_d)
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
        assert weights.size == self.input_size, "weights size"

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
            assert flat.size == self.input_size, "flat-field array size"
            cflat = numpy.ascontiguousarray(flat.ravel(), dtype=data_d)
        if dark is not None:
            do_dark = True
            assert dark.size == self.input_size, "dark current array size"
            cdark = numpy.ascontiguousarray(dark.ravel(), dtype=data_d)
        if solidAngle is not None:
            do_solidAngle = True
            assert solidAngle.size == self.input_size, "Solid angle array size"
            csolidAngle = numpy.ascontiguousarray(solidAngle.ravel(), dtype=data_d)
        if polarization is not None:
            do_polarization = True
            assert polarization.size == self.input_size, "polarization array size"
            cpolarization = numpy.ascontiguousarray(polarization.ravel(), dtype=data_d)

        if (do_dark + do_flat + do_polarization + do_solidAngle):
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
            cdata = numpy.empty(self.input_size, dtype=data_d)
            if do_dummy:
                for i in prange(self.input_size, nogil=True, schedule="static"):
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
                        cdata[i] = data
                    else:  # set all dummy_like values to cdummy. simplifies further processing
                        cdata[i] = cdummy
            else:
                for i in prange(self.input_size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if do_dark:
                        data = data - cdark[i]
                    if do_flat:
                        data = data / cflat[i]
                    if do_polarization:
                        data = data / cpolarization[i]
                    if do_solidAngle:
                        data = data / csolidAngle[i]
                    cdata[i] = data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
                cdata = numpy.zeros(self.input_size, dtype=data_d)
                for i in prange(self.input_size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] = data
                    else:
                        cdata[i] = cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i in prange(self.output_size, nogil=True, schedule="guided"):
            acc_data = 0.0
            acc_count = 0.0
            for j in range(self._indptr[i], self._indptr[i + 1]):
                idx = self._indices[j]
                coef = self._data[j]
                if coef == 0.0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue
                acc_data = acc_data + pown(coef, coef_power) * data
                acc_count = acc_count + coef

            sum_data[i] = acc_data
            sum_count[i] = acc_count
            if acc_count > epsilon:
                merged[i] = acc_data / acc_count / normalization_factor
            else:
                merged[i] = cdummy

        if self.bin_centers is None:
            # 2D integration case
            return (numpy.asarray(merged).reshape(self.bins).T,
                    self.bin_centers0,
                    self.bin_centers1,
                    numpy.asarray(sum_data).reshape(self.bins).T,
                    numpy.asarray(sum_count).reshape(self.bins).T)
        else:
            # 1D integration case
            return (self.bin_centers,
                    numpy.asarray(merged),
                    numpy.asarray(sum_data),
                    numpy.asarray(sum_count))


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
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average. WIP
        :return: namedtuple with "position intensity sigma signal variance normalization count std sem norm_sq"
        :rtype: Integrate1dtpl named-tuple of ndarrays
        """
        cdef:
            index_t i, j, idx = 0
            acc_t acc_sig = 0.0, acc_var = 0.0, acc_norm = 0.0, acc_count = 0.0, coef = 0.0, acc_norm_sq=0.0
            acc_t delta1, delta2, b, omega_A, omega_B, omega2_A, omega2_B, w, norm, sig, var, count
            data_t empty
            acc_t[::1] sum_sig = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_var = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm_sq = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_count = numpy.empty(self.output_size, dtype=acc_d)
            data_t[::1] merged = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] std = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] sem = numpy.empty(self.output_size, dtype=data_d)
            data_t[:, ::1] preproc4
            bint do_azimuthal_variance = error_model is ErrorModel.AZIMUTHAL
            bint do_variance = error_model is not ErrorModel.NO
        assert weights.size == self.input_size, "weights size"
        empty = dummy if dummy is not None else self.empty
        #Call the preprocessor ...
        preproc4 = preproc(weights.ravel(),
                           dark=dark,
                           flat=flat,
                           solidangle=solidangle,
                           polarization=polarization,
                           absorption=absorption,
                           mask=self.cmask if self.check_mask else None,
                           dummy=dummy,
                           delta_dummy=delta_dummy,
                           normalization_factor=normalization_factor,
                           empty=self.empty,
                           split_result=4,
                           variance=variance,
                           dtype=data_d,
                           error_model=error_model,
                           apply_normalization=not weighted_average,
                           out=self.preprocessed)

        for i in prange(self.output_size, nogil=True, schedule="guided"):
            acc_sig = 0.0
            acc_var = 0.0
            acc_norm = 0.0
            acc_norm_sq = 0.0
            acc_count = 0.0
            for j in range(self._indptr[i], self._indptr[i + 1]):
                coef = self._data[j]
                if coef == 0.0:
                    continue
                idx = self._indices[j]

                sig = preproc4[idx, 0]
                var = preproc4[idx, 1]
                norm = preproc4[idx, 2]
                count = preproc4[idx, 3]

                acc_count = acc_count + coef * count
                if do_azimuthal_variance:
                    if acc_norm_sq <= 0.0:
                        acc_sig = coef * sig
                        #Variance remains at 0
                        acc_norm = coef * norm
                        acc_norm_sq = acc_norm * acc_norm
                    else:
                        # see https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                        # Not correct, Inspired by VV_{A+b} = VV_A + ω²·(b-V_A/Ω_A)·(b-V_{A+b}/Ω_{A+b})
                        # Emprically validated against 2-pass implementation in Python/scipy-sparse

                        omega_A = acc_norm
                        omega_B = coef * norm # ω_i = c_i * norm_i
                        omega2_A = acc_norm_sq
                        omega2_B = omega_B*omega_B
                        acc_norm = omega_A + omega_B
                        acc_norm_sq = omega2_A + omega2_B
                        # omega3 = acc_norm * omega_A * omega2_B
                        # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                        b = sig / norm
                        delta1 = acc_sig/omega_A - b
                        acc_sig = acc_sig + coef * sig
                        delta2 = acc_sig / acc_norm - b
                        acc_var = acc_var +  omega2_B * delta1 * delta2
                else:
                    acc_sig = acc_sig + coef * sig
                    if do_variance:
                        acc_var = acc_var + coef * coef * var
                    w = coef * norm
                    acc_norm = acc_norm + w
                    acc_norm_sq = acc_norm_sq + w*w

            sum_sig[i] = acc_sig
            sum_var[i] = acc_var
            sum_norm[i] = acc_norm
            sum_norm_sq[i] = acc_norm_sq
            sum_count[i] = acc_count
            if acc_norm_sq > 0.0:
                merged[i] = acc_sig / acc_norm
                if do_variance:
                    std[i] = sqrt(acc_var / acc_norm_sq)
                    sem[i] = sqrt(acc_var) / acc_norm
                else:
                    std[i] = sem[i] = empty
            else:
                merged[i] = std[i] = sem[i] = empty
        if self.bin_centers is None:
            # 2D integration case
            return Integrate2dtpl(self.bin_centers0, self.bin_centers1,
                              numpy.asarray(merged).reshape(self.bins).T,
                              numpy.asarray(sem).reshape(self.bins).T,
                              numpy.asarray(sum_sig).reshape(self.bins).T,
                              numpy.asarray(sum_var).reshape(self.bins).T,
                              numpy.asarray(sum_norm).reshape(self.bins).T,
                              numpy.asarray(sum_count).reshape(self.bins).T,
                              numpy.asarray(std).reshape(self.bins).T,
                              numpy.asarray(sem).reshape(self.bins).T,
                              numpy.asarray(sum_norm_sq).reshape(self.bins).T)
        else:
            # 1D integration case: "position intensity error signal variance normalization count std sem norm_sq"
            return Integrate1dtpl(self.bin_centers,
                                  numpy.asarray(merged),numpy.asarray(sem) ,
                                  numpy.asarray(sum_sig),numpy.asarray(sum_var),
                                  numpy.asarray(sum_norm), numpy.asarray(sum_count),
                                  numpy.asarray(std), numpy.asarray(sem), numpy.asarray(sum_norm_sq))

    integrate = integrate_ng

    def sigma_clip(self,
                   weights,
                   dark=None,
                   dummy=None,
                   delta_dummy=None,
                   variance=None,
                   dark_variance=None,
                   flat=None,
                   solidangle=None,
                   polarization=None,
                   absorption=None,
                   bint safe=True,
                   error_model=ErrorModel.NO,
                   data_t normalization_factor=1.0,
                   double cutoff=0.0,
                   int cycle=5,
                   ):
        """Perform a sigma-clipping iterative filter within each along each row.
        see the doc of scipy.stats.sigmaclip for more descriptions.

        If the error model is "azimuthal": the variance is the variance within a bin,
        which is refined at each iteration, can be costly !

        Else, the error is propagated according to:

        .. math::

            signal = (raw - dark)
            variance = variance + dark_variance
            normalization  = normalization_factor*(flat * solidangle * polarization * absortoption)
            count = number of pixel contributing

        Integration is performed using the CSR representation of the look-up table on all
        arrays: signal, variance, normalization and count

        The threshold can automaticlly be calculated from Chauvenet's: sqrt(2*log(nbpix/sqrt(2.0f*pi)))

        :param weights: input image
        :type weights: ndarray
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param variance: the variance associate to the image
        :type variance: ndarray
        :param dark_variance: the variance associate to the dark
        :type dark_variance: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param absorption: Apparent efficiency of a pixel due to parallax effect
        :type absorption: ndarray
        :param safe: set to True to save some tests
        :param error_model: set to "poissonian" to use signal as variance (minimum 1), "azimuthal" to use the variance in a ring.
        :param normalization_factor: divide the valid result by this value

        :return: namedtuple with "position intensity sigma signal variance normalization count std sem norm_sq"
        :rtype: Integrate1dtpl named-tuple of ndarrays

        """
        error_model = ErrorModel.parse(error_model)
        cdef:
            index_t i, j, c, bad_pix, idx = 0
            acc_t acc_sig = 0.0, acc_var = 0.0, acc_norm = 0.0, acc_count = 0.0, coef = 0.0, acc_norm_sq=0.0
            acc_t sig, norm, count, var
            acc_t delta1, delta2, b, x, omega_A, omega_B, aver, std, chauvenet_cutoff, omega2_A, omega2_B, w
            data_t empty
            acc_t[::1] sum_sig = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_var = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm_sq = numpy.empty(self.output_size, dtype=acc_d)
            acc_t[::1] sum_count = numpy.empty(self.output_size, dtype=acc_d)
            data_t[::1] merged = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] stda = numpy.empty(self.output_size, dtype=data_d)
            data_t[::1] sema = numpy.empty(self.output_size, dtype=data_d)
            data_t[:, ::1] preproc4
            bint do_azimuthal_variance = error_model == ErrorModel.AZIMUTHAL
            bint do_hybrid_variance = error_model == ErrorModel.HYBRID
        assert weights.size == self.input_size, "weights size"
        empty = dummy if dummy is not None else self.empty
        #Call the preprocessor ...
        preproc4 = preproc(weights.ravel(),
                           dark=dark,
                           flat=flat,
                           solidangle=solidangle,
                           polarization=polarization,
                           absorption=absorption,
                           mask=self.cmask if self.check_mask else None,
                           dummy=dummy,
                           delta_dummy=delta_dummy,
                           normalization_factor=normalization_factor,
                           empty=self.empty,
                           split_result=4,
                           variance=variance,
                           dtype=data_d,
                           error_model=error_model,
                           out=self.preprocessed)
        with nogil:
            # Integrate once
            for i in prange(self.output_size, schedule="guided"):
                acc_sig = 0.0
                acc_var = 0.0
                acc_norm = 0.0
                acc_norm_sq = 0.0
                acc_count = 0.0
                for j in range(self._indptr[i], self._indptr[i + 1]):
                    coef = self._data[j]
                    if coef == 0.0:
                        continue
                    idx = self._indices[j]
                    sig = preproc4[idx, 0]
                    var = preproc4[idx, 1]
                    norm = preproc4[idx, 2]
                    count = preproc4[idx, 3]
                    if isnan(sig) or isnan(var) or isnan(norm) or isnan(count) or (norm == 0.0) or (count == 0.0):
                        continue
                    w = coef * norm
                    if do_azimuthal_variance or (do_hybrid_variance and cycle):
                        if acc_norm == 0.0:
                            # Initialize the accumulator with data from the pixel
                            acc_sig = coef * sig
                            #Variance remains at 0
                            acc_norm = w
                            acc_norm_sq = w*w
                            acc_count = coef * count
                        else:
                            omega_A = acc_norm
                            omega_B = coef * norm # ω_i = c_i * norm_i
                            omega2_A = acc_norm_sq
                            omega2_B = omega_B*omega_B
                            acc_norm = omega_A + omega_B
                            acc_norm_sq = omega2_A + omega2_B
                            # omega3 = acc_norm * omega_A * omega2_B
                            # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                            b = sig / norm
                            delta1 = acc_sig/omega_A - b
                            acc_sig = acc_sig + coef * sig
                            delta2 = acc_sig / acc_norm - b
                            acc_var = acc_var +  omega2_B * delta1 * delta2
                            acc_count = acc_count + coef * count
                    else:
                        acc_sig = acc_sig + coef * sig
                        acc_var = acc_var + coef * coef * var
                        acc_norm = acc_norm + w
                        acc_norm_sq = acc_norm_sq + w*w
                        acc_count = acc_count + coef * count

                if (acc_norm_sq > 0.0):
                    aver = acc_sig / acc_norm
                    std = sqrt(acc_var / acc_norm_sq)
                    # sem = sqrt(acc_var) / acc_norm
                else:
                    aver = NAN;
                    std = NAN;
                    # sem = NAN;

                # cycle for sigma-clipping
                for c in range(cycle):
                    # Sigma-clip
                    if (acc_norm == 0.0) or (acc_count == 0.0):
                        break

                    #This is for Chauvenet's criterion
                    acc_count = max(3.0, acc_count)
                    chauvenet_cutoff = max(cutoff, sqrt(2.0*log(acc_count/sqrt(2.0*M_PI))))
                    # Discard  outliers
                    bad_pix = 0
                    for j in range(self._indptr[i], self._indptr[i + 1]):
                        coef = self._data[j]
                        if coef == 0.0:
                            continue
                        idx = self._indices[j]
                        sig = preproc4[idx, 0]
                        var = preproc4[idx, 1]
                        norm = preproc4[idx, 2]
                        count = preproc4[idx, 3]
                        if isnan(sig) or isnan(var) or isnan(norm) or isnan(count) or (norm == 0.0) or (count == 0.0):
                            continue
                        # Check validity (on cnt, i.e. s3) and normalisation (in s2) value to avoid zero division
                        x = sig / norm
                        if fabs(x - aver) > chauvenet_cutoff * std:
                            preproc4[idx, 3] = NAN;
                            bad_pix = bad_pix + 1
                    if bad_pix == 0:
                        if do_hybrid_variance:
                            c = cycle #enforce the leave of the loop
                        else:
                            break

                    #integrate again
                    acc_sig = 0.0
                    acc_var = 0.0
                    acc_norm = 0.0
                    acc_norm_sq = 0.0
                    acc_count = 0.0

                    for j in range(self._indptr[i], self._indptr[i + 1]):
                        coef = self._data[j]
                        if coef == 0.0:
                            continue
                        idx = self._indices[j]
                        sig = preproc4[idx, 0]
                        var = preproc4[idx, 1]
                        norm = preproc4[idx, 2]
                        count = preproc4[idx, 3]
                        if isnan(sig) or isnan(var) or isnan(norm) or isnan(count) or (norm == 0.0) or (count == 0.0):
                            continue
                        w = coef * norm
                        if do_azimuthal_variance or (do_hybrid_variance and c+1<cycle):
                            if acc_norm_sq == 0.0:
                                # Initialize the accumulator with data from the pixel
                                acc_sig = coef * sig
                                #Variance remains at 0
                                acc_norm = w
                                acc_norm_sq = w*w
                                acc_count = coef * count
                            else:
                                omega_A = acc_norm
                                omega_B = coef * norm # ω_i = c_i * norm_i
                                omega2_A = acc_norm_sq
                                omega2_B = omega_B*omega_B
                                acc_norm = omega_A + omega_B
                                acc_norm_sq = omega2_A + omega2_B
                                # omega3 = acc_norm * omega_A * omega2_B
                                # VV_{AUb} = VV_A + ω_b^2 * (b-<A>) * (b-<AUb>)
                                b = sig / norm
                                delta1 = acc_sig/omega_A - b
                                acc_sig = acc_sig + coef * sig
                                delta2 = acc_sig / acc_norm - b
                                acc_var = acc_var +  omega2_B * delta1 * delta2
                                acc_count = acc_count + coef * count
                        else:
                            acc_sig = acc_sig + coef * sig
                            acc_var = acc_var + coef * coef * var
                            acc_norm = acc_norm + w
                            acc_norm_sq = acc_norm_sq + w*w
                            acc_count = acc_count + coef * count
                    if (acc_norm_sq > 0.0):
                        aver = acc_sig / acc_norm
                        std = sqrt(acc_var / acc_norm_sq)
                        # sem = sqrt(acc_var) / acc_norm # Not needed yet
                    else:
                        aver = NAN;
                        std = NAN;
                        # sem = NAN;
                    if bad_pix == 0:
                        break

                #collect things ...
                sum_sig[i] = acc_sig
                sum_var[i] = acc_var
                sum_norm[i] = acc_norm
                sum_norm_sq[i] = acc_norm_sq
                sum_count[i] = acc_count
                if (acc_norm_sq > 0.0):
                    merged[i] = aver  # calculated previously
                    stda[i] = std  # sqrt(acc_var / acc_norm_sq) # already calculated previously
                    sema[i] = sqrt(acc_var) / acc_norm  # not calculated previously since it was not needed
                else:
                    merged[i] = empty
                    stda[i] = empty
                    sema[i] = empty

        #"position intensity sigma signal variance normalization count std sem norm_sq"
        return Integrate1dtpl(self.bin_centers,
                              numpy.asarray(merged),numpy.asarray(sema) ,
                              numpy.asarray(sum_sig),numpy.asarray(sum_var),
                              numpy.asarray(sum_norm), numpy.asarray(sum_count),
                              numpy.asarray(stda), numpy.asarray(sema), numpy.asarray(sum_norm_sq))


    def medfilt(   self,
                   weights,
                   dark=None,
                   dummy=None,
                   delta_dummy=None,
                   variance=None,
                   dark_variance=None,
                   flat=None,
                   solidangle=None,
                   polarization=None,
                   absorption=None,
                   bint safe=True,
                   error_model=ErrorModel.NO,
                   data_t normalization_factor=1.0,
                   double quant_min=0.5,
                   double quant_max=0.5,
                   ):
        """Perform a median filter/quantile averaging in azimuthal space

        Else, the error is propagated like Poisson or pre-defined variance, no azimuthal variance for now.

        Integration is performed using the CSR representation of the look-up table on all
        arrays: signal, variance, normalization and count

        All data are duplicated, sorted and the relevant values (i.e. within [quant_min..quant_max])
        are averaged like in `integrate_ng`

        :param weights: input image
        :type weights: ndarray
        :param dark: array with the dark-current value to be subtracted (if any)
        :type dark: ndarray
        :param dummy: value for dead pixels (optional)
        :type dummy: float
        :param delta_dummy: precision for dead-pixel value in dynamic masking
        :type delta_dummy: float
        :param variance: the variance associate to the image
        :type variance: ndarray
        :param dark_variance: the variance associate to the dark
        :type dark_variance: ndarray
        :param flat: array with the dark-current value to be divided by (if any)
        :type flat: ndarray
        :param solidAngle: array with the solid angle of each pixel to be divided by (if any)
        :type solidAngle: ndarray
        :param polarization: array with the polarization correction values to be divided by (if any)
        :type polarization: ndarray
        :param absorption: Apparent efficiency of a pixel due to parallax effect
        :type absorption: ndarray
        :param safe: set to True to save some tests
        :param error_model: set to "poissonian" to use signal as variance (minimum 1), "azimuthal" to use the variance in a ring.
        :param normalization_factor: divide the valid result by this value
        :param quant_min: start percentile/100 to use. Use 0.5 for the median (default). 0<=quant_min<=1
        :param quant_max: stop percentile/100 to use. Use 0.5 for the median (default). 0<=quant_max<=1

        :return: namedtuple with "position intensity sigma signal variance normalization count std sem norm_sq"
        :rtype: Integrate1dtpl named-tuple of ndarrays
        """
        error_model = ErrorModel.parse(error_model)
        cdef:
            index_t i, j, c, bad_pix, npix = self._indices.shape[0], idx = 0, start, stop, cnt=0
            acc_t acc_sig = 0.0, acc_var = 0.0, acc_norm = 0.0, acc_count = 0.0,  coef = 0.0, acc_norm_sq=0.0
            acc_t cumsum = 0.0
            data_t qmin, qmax
            data_t empty, sig, var, nrm, weight, nrm2
            acc_t[::1] sum_sig = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_var = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.zeros(self.output_size, dtype=acc_d)
            acc_t[::1] sum_norm_sq = numpy.zeros(self.output_size, dtype=acc_d)
            index_t[::1] sum_count = numpy.zeros(self.output_size, dtype=index_d)
            data_t[::1] merged = numpy.zeros(self.output_size, dtype=data_d)
            data_t[::1] stda = numpy.zeros(self.output_size, dtype=data_d)
            data_t[::1] sema = numpy.zeros(self.output_size, dtype=data_d)
            data_t[:, ::1] preproc4
            bint do_azimuthal_variance = error_model == ErrorModel.AZIMUTHAL
            bint do_hybrid_variance = error_model == ErrorModel.HYBRID
            float4_t element, former_element
            float4_t[::1] work = numpy.zeros(npix, dtype=float4_d)

        assert weights.size == self.input_size, "weights size"
        empty = dummy if dummy is not None else self.empty
        #Call the preprocessor ...
        preproc4 = preproc(weights.ravel(),
                           dark=dark,
                           flat=flat,
                           solidangle=solidangle,
                           polarization=polarization,
                           absorption=absorption,
                           mask=self.cmask if self.check_mask else None,
                           dummy=dummy,
                           delta_dummy=delta_dummy,
                           normalization_factor=normalization_factor,
                           empty=self.empty,
                           split_result=4,
                           variance=variance,
                           dtype=data_d,
                           error_model=error_model,
                           out=self.preprocessed)
        # print("start nogil", npix)
        with nogil:
            # Duplicate the input data and populate the large work-array
            for i in range(npix):  # NOT faster in parallel !
                weight = self._data[i]
                j = self._indices[i]
                sig = preproc4[j,0]
                var = preproc4[j,1]
                nrm = preproc4[j,2]
                element.s0 = sig/nrm                 # average signal
                element.s1 = sig * weight            # weighted raw signal
                element.s2 = var * weight * weight   # weighted raw variance
                element.s3 = nrm * weight            # weighted raw normalization
                work[i] = element
            for idx in prange(self.output_size, schedule="guided"):
                start = self._indptr[idx]
                stop = self._indptr[idx+1]
                acc_sig = acc_var = acc_norm = acc_norm_sq = 0.0
                cnt = 0
                cumsum = 0.0

                sort_float4(work[start:stop])

                for i in range(start, stop):
                    cumsum = cumsum + work[i].s3
                    work[i].s0 = cumsum

                qmin = quant_min * cumsum
                qmax = quant_max * cumsum

                element.s0 = 0.0
                for i in range(start, stop):
                    former_element = element
                    element = work[i]
                    if ((element.s3!=0) and
                        (((qmin<=former_element.s0) and (element.s0 <= qmax)) or
                        ((qmin>=former_element.s0)  and (element.s0 >= qmax)))):   #specific case where qmin==qmax
                        acc_sig = acc_sig + element.s1
                        acc_var = acc_var + element.s2
                        acc_norm = acc_norm + element.s3
                        acc_norm_sq = acc_norm_sq + element.s3*element.s3
                        cnt = cnt + 1

                #collect things ...
                sum_sig[idx] = acc_sig
                sum_var[idx] = acc_var
                sum_norm[idx] = acc_norm
                sum_norm_sq[idx] = acc_norm_sq
                sum_count[idx] = cnt
                if (acc_norm_sq):
                    merged[idx] = acc_sig/acc_norm
                    stda[idx] = sqrt(acc_var / acc_norm_sq)
                    sema[idx] = sqrt(acc_var) / acc_norm
                else:
                    merged[idx] = empty
                    stda[idx] = empty
                    sema[idx] = empty

        #"position intensity sigma signal variance normalization count std sem norm_sq"
        return Integrate1dtpl(self.bin_centers,
                              numpy.asarray(merged),numpy.asarray(sema) ,
                              numpy.asarray(sum_sig),numpy.asarray(sum_var),
                              numpy.asarray(sum_norm), numpy.asarray(sum_count),
                              numpy.asarray(stda), numpy.asarray(sema), numpy.asarray(sum_norm_sq))
