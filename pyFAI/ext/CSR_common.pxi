# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2021 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.kieffer@esrf.fr"
__date__ = "25/06/2021"
__status__ = "stable"
__license__ = "MIT"


import cython
from cython.parallel import prange
import numpy

from .preproc import preproc
from ..containers import Integrate1dtpl

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
        readonly data_t empty
        readonly data_t[::1] _data
        readonly index_t[::1] _indices, _indptr

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
            index_t i = 0, j = 0, idx = 0, bins = self.output_size, size = self.input_size 
            acc_t acc_data = 0.0, acc_count = 0.0, epsilon = 1e-10, coef = 0.0
            data_t data = 0.0, cdummy = 0.0, cddummy = 0.0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.zeros(self.bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.zeros(self.bins, dtype=acc_d)
            data_t[::1] merged = numpy.zeros(self.bins, dtype=data_d)
            data_t[::1] cdata, tdata, cflat, cdark, csolidAngle, cpolarization
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
            cdata = numpy.empty(size, dtype=data_d)
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
                        cdata[i] = data
                    else:  # set all dummy_like values to cdummy. simplifies further processing
                        cdata[i] = cdummy
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
                    cdata[i] = data
        else:
            if do_dummy:
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)
                cdata = numpy.zeros(size, dtype=data_d)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] = data
                    else:
                        cdata[i] = cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=data_d)

        for i in prange(bins, nogil=True, schedule="guided"):
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
                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef

            sum_data[i] = acc_data
            sum_count[i] = acc_count
            if acc_count > epsilon:
                merged[i] = acc_data / acc_count / normalization_factor
            else:
                merged[i] = cdummy
        return (self.bin_centers, 
                numpy.asarray(merged), 
                numpy.asarray(sum_data), 
                numpy.asarray(sum_count))

    integrate = integrate_legacy

    def integrate_ng(self,
                     weights,
                     variance=None,
                     poissonian=None,
                     dummy=None,
                     delta_dummy=None,
                     dark=None,
                     flat=None,
                     solidangle=None,
                     polarization=None,
                     absorption=None,
                     data_t normalization_factor=1.0,
                     ):
        """
        Actually perform the integration which in this case consists of:
         * Calculate the signal, variance and the normalization parts
         * Perform the integration which is here a matrix-vector product

        :param weights: input image
        :type weights: ndarray
        :param variance: the variance associate to the image
        :type variance: ndarray
        :param poissonian: set to use signal as variance (minimum 1), set to False to use azimuthal model. 
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

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: Integrate1dtpl 4-named-tuple of ndarrays
        """
        cdef:
            int32_t i, j, idx = 0, bins = self.bins, size = self.size
            acc_t acc_sig = 0.0, acc_var = 0.0, acc_norm = 0.0, acc_count = 0.0, coef = 0.0
            acc_t delta, x, omega_A, omega_B, omega3
            data_t empty
            acc_t[::1] sum_sig = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_var = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.empty(bins, dtype=acc_d)
            data_t[::1] merged = numpy.empty(bins, dtype=data_d)
            data_t[::1] error = numpy.empty(bins, dtype=data_d)
            data_t[:, ::1] preproc4
            bint do_azimuthal_variance = poissonian is False
        assert weights.size == size, "weights size"
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
                           poissonian= poissonian is True)

        for i in prange(bins, nogil=True, schedule="guided"):
            acc_sig = 0.0
            acc_var = 0.0
            acc_norm = 0.0
            acc_count = 0.0
            for j in range(self._indptr[i], self._indptr[i + 1]):
                coef = self._data[j]
                if coef == 0.0:
                    continue
                idx = self._indices[j]
                
                if do_azimuthal_variance:
                    if acc_count == 0.0:
                        acc_sig = coef * preproc4[idx, 0]
                        #Variance remains at 0
                        acc_norm = coef * preproc4[idx, 2] 
                        acc_count = coef * preproc4[idx, 3]
                    else:
                        # see https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                        omega_A = acc_norm
                        omega_B = coef * preproc4[idx, 2]
                        acc_norm = omega_A + omega_B
                        omega3 = acc_norm * omega_A * omega_B
                        x = coef * preproc4[idx, 0]
                        delta = omega_B*acc_sig - omega_A*x
                        acc_var = acc_var +  delta*delta/omega3
                        acc_sig = acc_sig + x
                        acc_count = acc_count + coef * preproc4[idx, 3]
                else:
                    acc_sig = acc_sig + coef * preproc4[idx, 0]
                    acc_var = acc_var + coef * coef * preproc4[idx, 1]
                    acc_norm = acc_norm + coef * preproc4[idx, 2] 
                    acc_count = acc_count + coef * preproc4[idx, 3]

            sum_sig[i] = acc_sig
            sum_var[i] = acc_var
            sum_norm[i] = acc_norm
            sum_count[i] = acc_count
            if acc_count > 0.0:
                merged[i] = acc_sig / acc_norm
                error[i] = sqrt(acc_var) / acc_norm
            else:
                merged[i] = empty
                error[i] = empty
        #"position intensity error signal variance normalization count"
        return Integrate1dtpl(self.bin_centers, 
                              numpy.asarray(merged),numpy.asarray(error) ,
                              numpy.asarray(sum_sig),numpy.asarray(sum_var), 
                              numpy.asarray(sum_norm), numpy.asarray(sum_count))
    
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
                   error_model=None,
                   data_t normalization_factor=1.0,
                   double cutoff=0.0, 
                   int cycle=5,
                   ):
        """Perform the sigma-clipping
        
        The threshold can automaticlly be calculated from Chauvenet's: sqrt(2*log(nbpix/sqrt(2.0f*pi)))
        
        TODO: doc
        :return: Integrate1dtpl containing the result 
        """
        error_model = error_model.lower() if error_model else ""
        cdef:
            int32_t i, j, c, idx = 0, bins = self.bins, size = self.size
            acc_t acc_sig = 0.0, acc_var = 0.0, acc_norm = 0.0, acc_count = 0.0, coef = 0.0
            acc_t delta, x, omega_A, omega_B, omega3, aver, std, chauvenet_cutoff, signal
            data_t empty
            acc_t[::1] sum_sig = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_var = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_norm = numpy.empty(bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.empty(bins, dtype=acc_d)
            data_t[::1] merged = numpy.empty(bins, dtype=data_d)
            data_t[::1] error = numpy.empty(bins, dtype=data_d)
            data_t[:, ::1] preproc4
            bint do_azimuthal_variance = error_model.startswith("azim")
        assert weights.size == size, "weights size"
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
                           poissonian= error_model.startswith("poisson"))
        with nogil:
            # Integrate once
            for i in prange(bins, schedule="guided"):
                acc_sig = 0.0
                acc_var = 0.0
                acc_norm = 0.0
                acc_count = 0.0
                for j in range(self._indptr[i], self._indptr[i + 1]):
                    coef = self._data[j]
                    if coef == 0.0:
                        continue
                    idx = self._indices[j]
                    if isnan(preproc4[idx, 3]):
                        continue
                    
                    if do_azimuthal_variance:
                        if acc_count == 0.0:
                            acc_sig = coef * preproc4[idx, 0]
                            #Variance remains at 0
                            acc_norm = coef * preproc4[idx, 2] 
                            acc_count = coef * preproc4[idx, 3]
                        else:
                            # see https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                            omega_A = acc_norm
                            omega_B = coef * preproc4[idx, 2]
                            acc_norm = omega_A + omega_B
                            omega3 = acc_norm * omega_A * omega_B
                            x = coef * preproc4[idx, 0]
                            delta = omega_B*acc_sig - omega_A*x
                            acc_var = acc_var +  delta*delta/omega3
                            acc_sig = acc_sig + x
                            acc_count = acc_count + coef * preproc4[idx, 3]
                    else:
                        acc_sig = acc_sig + coef * preproc4[idx, 0]
                        acc_var = acc_var + coef * coef * preproc4[idx, 1]
                        acc_norm = acc_norm + coef * preproc4[idx, 2] 
                        acc_count = acc_count + coef * preproc4[idx, 3]
                        
                if (acc_norm > 0.0):
                    aver = acc_sig / acc_norm
                    std = sqrt(acc_var / acc_norm)
                    # sem = sqrt(acc_sig) / acc_norm
                else:
                    aver = NAN;
                    std = NAN;
                    # sem = NAN;
                
                # cycle for sigma-clipping
                for c in range(cycle):
                    # Sigma-clip
                    if (acc_norm == 0.0) or (acc_count == 0.0):
                        break
                    acc_count = max(3.0, acc_count) #This is for Chauvenet's criterion
                    chauvenet_cutoff = max(cutoff, sqrt(2.0*log(acc_count/sqrt(2.0*M_PI)))) * std
                    
                    for j in range(self._indptr[i], self._indptr[i + 1]):
                        coef = self._data[j]
                        if coef == 0.0:
                            continue
                        idx = self._indices[j]
                        
                        acc_sig = preproc4[idx, 0]
                        acc_var = preproc4[idx, 1]
                        acc_norm = preproc4[idx, 2]
                        acc_count = preproc4[idx, 3]
                        # Check validity (on cnt, i.e. s3) and normalisation (in s2) value to avoid zero division 
                        if (not isnan(acc_count) and (acc_norm > 0.0)):
                            if fabs(acc_sig / acc_norm - aver) > chauvenet_cutoff:
                                preproc4[idx, 3] = NAN;
                    
                    #integrate again
                    acc_sig = 0.0
                    acc_var = 0.0
                    acc_norm = 0.0
                    acc_count = 0.0
                    for j in range(self._indptr[i], self._indptr[i + 1]):
                        coef = self._data[j]
                        if coef == 0.0:
                            continue
                        idx = self._indices[j]
                        if isnan(preproc4[idx, 3]):
                            continue
                                 
                        if do_azimuthal_variance:
                            if acc_count == 0.0:
                                acc_sig = coef * preproc4[idx, 0]
                                #Variance remains at 0
                                acc_norm = coef * preproc4[idx, 2] 
                                acc_count = coef * preproc4[idx, 3]
                            else:
                                # see https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                                omega_A = acc_norm
                                omega_B = coef * preproc4[idx, 2]
                                acc_norm = omega_A + omega_B
                                omega3 = acc_norm * omega_A * omega_B
                                x = coef * preproc4[idx, 0]
                                delta = omega_B*acc_sig - omega_A*x
                                acc_var = acc_var +  delta*delta/omega3
                                acc_sig = acc_sig + x
                                acc_count = acc_count + coef * preproc4[idx, 3]
                        else:
                            acc_sig = acc_sig + coef * preproc4[idx, 0]
                            acc_var = acc_var + coef * coef * preproc4[idx, 1]
                            acc_norm = acc_norm + coef * preproc4[idx, 2] 
                            acc_count = acc_count + coef * preproc4[idx, 3]
                    if (acc_norm > 0.0):
                        aver = acc_sig / acc_norm
                        std = sqrt(acc_var / acc_norm)
                        # sem = sqrt(acc_sig) / acc_norm
                    else:
                        aver = NAN;
                        std = NAN;
                        # sem = NAN;
                    
                #collect things ...                     
                sum_sig[i] = acc_sig
                sum_var[i] = acc_var
                sum_norm[i] = acc_norm
                sum_count[i] = acc_count
                if acc_count > 0.0:
                    merged[i] = acc_sig / acc_norm
                    error[i] = sqrt(acc_var) / acc_norm
                else:
                    merged[i] = empty
                    error[i] = empty

        
        #"position intensity error signal variance normalization count"
        return Integrate1dtpl(self.bin_centers, 
                              numpy.asarray(merged),numpy.asarray(error) ,
                              numpy.asarray(sum_sig),numpy.asarray(sum_var), 
                              numpy.asarray(sum_norm), numpy.asarray(sum_count))
