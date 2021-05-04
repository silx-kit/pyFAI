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
__date__ = "31/03/2021"
__status__ = "stable"
__license__ = "MIT"


from cython.parallel import prange
import numpy

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
        readonly index_t input_size, output_size, lut_size
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
        self.output_size = lut.shape[0]
        self.lut_size = lut.shape[1]
        self._lut = lut
    
    def __dealloc__(self):
        self._lut = None
        self.empty = 0
        self.input_size = 0
        self.output_size = 0 
        self.nnz = 0
        
    def integrate_legacy(self, weights,
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
        :param coef_power: put coef to a given power, 2 for variance, 1 for mean

        :return: positions, pattern, weighted_histogram and unweighted_histogram
        :rtype: 4-tuple of ndarrays

        """
        cdef:
            int i = 0, j = 0, idx = 0, bins = self.bins, lut_size = self.lut_size, size = self.size
            acc_t acc_data = 0, acc_count = 0, epsilon = 1e-10
            data_t data = 0, coef = 0, cdummy = 0, cddummy = 0
            bint do_dummy = False, do_dark = False, do_flat = False, do_polarization = False, do_solidAngle = False
            acc_t[::1] sum_data = numpy.empty(self.bins, dtype=acc_d)
            acc_t[::1] sum_count = numpy.empty(self.bins, dtype=acc_d)
            data_t[::1] merged = numpy.empty(self.bins, dtype=data_d)
            float[:] cdata, tdata, cflat, cdark, csolidAngle, cpolarization

        assert weights.size == size, "weights size"

        if dummy is not None:
            do_dummy = True
            cdummy = <data_t> float(dummy)
            if delta_dummy is None:
                cddummy = zerof
            else:
                cddummy = <data_t> float(delta_dummy)
        else:
            cdummy = self.empty

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
            tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
            cdata = numpy.zeros(size, dtype=numpy.float32)
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
                    else:
                        # set all dummy_like values to cdummy. simplifies further processing
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
                tdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)
                cdata = numpy.zeros(size, dtype=numpy.float32)
                for i in prange(size, nogil=True, schedule="static"):
                    data = tdata[i]
                    if ((cddummy != 0) and (fabs(data - cdummy) > cddummy)) or ((cddummy == 0) and (data != cdummy)):
                        cdata[i] = data
                    else:
                        cdata[i] = cdummy
            else:
                cdata = numpy.ascontiguousarray(weights.ravel(), dtype=numpy.float32)

        for i in prange(bins, nogil=True):
            acc_data = 0.0
            acc_count = 0.0
            for j in range(lut_size):
                idx = self._lut[i, j].idx
                coef = self._lut[i, j].coef
                if coef == 0.0 or idx < 0:
                    continue
                data = cdata[idx]
                if do_dummy and (data == cdummy):
                    continue

                acc_data = acc_data + (coef ** coef_power) * data
                acc_count = acc_count + coef
            sum_data[i] = acc_data
            sum_count[i] = acc_count
            if acc_count > epsilon:
                merged[i] = <data_t>(acc_data / acc_count / normalization_factor)
            else:
                merged[i] = cdummy

        return (self.bin_centers, 
                numpy.asarray(merged), 
                numpy.asarray(sum_data), 
                numpy.asarray(sum_count))

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
            int32_t i, j, idx = 0, 
            index_t bins = self.bins, size = self.size, lut_size = self.lut_size
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

        for i in prange(bins, nogil=True):
            acc_sig = 0.0
            acc_var = 0.0
            acc_norm = 0.0
            acc_count = 0.0
            for j in range(lut_size):
                idx = self._lut[i, j].idx
                coef = self._lut[i, j].coef
                if idx<0 or coef == 0.0:
                    continue
                
                if do_azimuthal_variance:
                    if acc_count == 0.0:
                        acc_sig = acc_sig + coef * preproc4[idx, 0]
                        #Variance remains at 0
                        acc_norm = acc_norm + coef * preproc4[idx, 2] 
                        acc_count = acc_count + coef * preproc4[idx, 3]
                    else:
                        # see https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf
                        omega_A = acc_norm
                        omega_B = coef * preproc4[idx, 2]
                        acc_norm = omega_A + omega_B
                        omega3 = acc_norm * omega_A * omega_B
                        x = <acc_t> preproc4[idx, 0] / preproc4[idx, 2]
                        delta = omega_B*acc_sig - omega_A*x
                        acc_var = acc_var +  delta*delta/omega3
                        acc_sig = acc_sig + coef * preproc4[idx, 0]
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

    integrate = integrate_legacy
