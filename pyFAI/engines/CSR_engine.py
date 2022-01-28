#
#    Copyright (C) 2017-2021 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "28/01/2022"
__status__ = "development"

import logging
import warnings
logger = logging.getLogger(__name__)
import numpy
from scipy.sparse import csr_matrix
from .preproc import preproc as preproc_np
from ..utils.mathutil import interp_filter
try:
    from ..ext.preproc import preproc as preproc_cy
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    preproc = preproc_np
else:
    preproc = preproc_cy
from ..utils import calc_checksum
from ..containers import Integrate1dtpl, Integrate2dtpl


class CSRIntegrator(object):

    def __init__(self,
                 image_size,
                 lut=None,
                 empty=0.0):
        """Constructor of the abstract class
        
        :param size: input image size
        :param lut: tuple of 3 arrays with data, indices and indptr,
                     index of the start of line in the CSR matrix
        :param empty: value for empty pixels
        """
        self.size = image_size
        self.empty = empty
        self.bins = None
        self._csr = None
        self._csr2 = None  # Used for propagating variance
        self.lut_size = 0  # actually nnz
        self.data = None
        self.indices = None
        self.indptr = None
        if lut is not None:
            assert len(lut) == 3
            self.set_matrix(*lut)

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content
        
        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.lut_size = len(indices)
        self.bins = len(indptr) - 1
        self._csr = csr_matrix((data, indices, indptr), shape=(self.bins, self.size))
        self._csr2 = csr_matrix((data * data, indices, indptr), shape=(self.bins, self.size))  # contains the coef squared, used for variance propagation

    def integrate(self,
                  signal,
                  variance=None,
                  poissonian=None,
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
        :param poissonian: set to use signal as variance (minimum 1), set to False to use azimuthal model. 
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
                       dtype=numpy.float32,
                       poissonian=bool(poissonian))
        prep.shape = numpy.prod(shape), -1
        # logger.warning("prep.shape %s lut_size %s, image_size %s, bins %s", prep.shape, self.lut_size, self.size, self.bins)
        res = numpy.empty((numpy.prod(self.bins), 4), dtype=numpy.float32)
        # logger.warning(self._csr.shape)
        res[:, 0] = self._csr.dot(prep[:, 0])
        res[:, 2] = self._csr.dot(prep[:, 2])
        res[:, 3] = self._csr.dot(prep[:, 3])
        if variance is not None or poissonian:
            res[:, 1] = self._csr2.dot(prep[:, 1])
        elif poissonian is False:
            # ask for azimuthal error model
            avg = res[:, 0] / res[:, 2]
            avg_ext = self._csr.T.dot(avg)
            delta = (prep[:, 0] / prep[:, 2] - avg_ext) ** 2
            res[:, 1] = self._csr.dot(delta)
        return res


class CsrIntegrator1d(CSRIntegrator):

    def __init__(self,
                 image_size,
                 lut=None,
                 empty=0.0,
                 unit=None,
                 bin_centers=None,
                 mask_checksum=None
                 ):
        """Constructor of the abstract class for 1D integration
        
        :param image_size: size of the image 
        :param lut: (data, indices, indptr) of the CSR matrix
        :param empty: value for empty pixels
        :param unit: the kind of radial units
        :param bin_center: position of the bin center
        :param mask_checksum: just a place-holder to track which mask was used
        Nota: bins value is deduced from the dimentionality of bin_centers 
        """
        self.bin_centers = bin_centers
        CSRIntegrator.__init__(self, image_size, lut, empty)
        self.pos0_range = self.pos1_range = None
        self.unit = unit
        self.mask_checksum = mask_checksum

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
                  poissonian=None,
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
        :param poissonian: set to use signal as variance (minimum 1), set to False to use azimuthal model.
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
        if variance is None and poissonian is None:
            do_variance = False
        else:
            do_variance = True
        trans = CSRIntegrator.integrate(self, signal, variance, poissonian,
                                        dummy, delta_dummy,
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

    integrate_ng = integrate

    def sigma_clip(self, data, dark=None, dummy=None, delta_dummy=None,
                   variance=None, dark_variance=None,
                   flat=None, solidangle=None, polarization=None, absorption=None,
                   safe=True, error_model=None,
                   normalization_factor=1.0,
                   cutoff=4.0, cycle=5):
        """
        Perform a sigma-clipping iterative filter within each along each row. 
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

        Formula for azimuthal variance from:
        https://dbs.ifi.uni-heidelberg.de/files/Team/eschubert/publications/SSDBM18-covariance-authorcopy.pdf

        :param dark: array of same shape as data for pre-processing
        :param dummy: value for invalid data
        :param delta_dummy: precesion for dummy assessement
        :param variance: array of same shape as data for pre-processing
        :param dark_variance: array of same shape as data for pre-processing
        :param flat: array of same shape as data for pre-processing
        :param solidangle: array of same shape as data for pre-processing
        :param polarization: array of same shape as data for pre-processing
        :param safe: Unused in this implementation
        :param normalization_factor: divide raw signal by this value
        :param cutoff: discard all points with |value - avg| > cutoff * sigma. 3-4 is quite common 
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :return: namedtuple with "position intensity error signal variance normalization count"
        """
        shape = data.shape
        error_model = error_model.lower() if error_model else ""

        poissonian = error_model.startswith("pois")
        azimuthal = (not poissonian) and (variance is None)

        prep = preproc(data,
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
                       dark_variance=dark_variance,
                       dtype=numpy.float32,
                       poissonian=poissonian)

        prep_flat = prep.reshape((numpy.prod(shape), 4))
        res = numpy.empty((numpy.prod(self.bins), 4), dtype=numpy.float32)

        # First azimuthal integration:
        res[:, 0] = self._csr.dot(prep_flat[:, 0])
        res[:, 2] = self._csr.dot(prep_flat[:, 2])
        res[:, 3] = self._csr.dot(prep_flat[:, 3])
        if azimuthal:
            # ask for azimuthal error model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg = res[:, 0] / res[:, 2]
                avg_ext = self._csr.T.dot(interp_filter(avg, avg))  # backproject the average value to the image
                norm = prep_flat[:, 2]
                msk = (norm == 0)
                delta2 = (prep_flat[:, 0] / norm - avg_ext) ** 2
                delta2[msk] = 0
            res[:, 1] = self._csr.dot(delta2)  # * norm) / self._csr.dot(norm)
        else:
            res[:, 1] = self._csr2.dot(prep_flat[:, 1])

        for _ in range(cycle):
            nrm = res[:, 2]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                avg = res[:, 0] / nrm
                std = numpy.sqrt(res[:, 1] / nrm)
            # Replace NaNs with interpolated values.
            avg = interp_filter(avg, avg)
            std = interp_filter(std, std)
            cnt = numpy.maximum(res[:, 3], 3)

            # Interpolate in 2D:
            avg2d = self._csr.T.dot(avg)
            std2d = self._csr.T.dot(std)
            cnt2d = numpy.maximum(self._csr.T.dot(cnt), 3)  # Needed for Chauvenet criterion
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                delta = abs(prep_flat[..., 0] / prep_flat[..., 2] - avg2d)
            chauvenet = numpy.maximum(cutoff, numpy.sqrt(2.0 * numpy.log(cnt2d / numpy.sqrt(2.0 * numpy.pi)))) * std2d
            # chauvenet = cutoff * std2d
            msk2d = numpy.where(numpy.logical_not(delta <= chauvenet))
            # print(len(msk2d[0]))
            prep_flat[msk2d] = 0
            # subsequent integrations:
            res[:, 0] = self._csr.dot(prep_flat[:, 0])
            res[:, 2] = self._csr.dot(prep_flat[:, 2])
            res[:, 3] = self._csr.dot(prep_flat[:, 3])
            if azimuthal:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    avg = res[:, 0] / res[:, 2]
                    avg_ext = self._csr.T.dot(interp_filter(avg, avg))  # backproject the average value to the image
                    msk = prep_flat[:, 2] == 0
                    delta2 = (prep_flat[:, 0] / prep_flat[:, 2] - avg_ext) ** 2
                    delta2[msk] = 0
                res[:, 1] = self._csr.dot(delta2)
            else:
                res[:, 1] = self._csr2.dot(prep_flat[:, 1])

        nrm = res[:, 2]
        msk = nrm <= 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg = res[:, 0] / nrm
            # Here we return the standaard deviation and not the standard error of the mean !
            std = numpy.sqrt(res[:, 1]) / nrm
        avg[msk] = self.empty
        std[msk] = self.empty

        return Integrate1dtpl(self.bin_centers, avg, std, res[:, 0], res[:, 1], res[:, 2], res[:, 3])

    @property
    def check_mask(self):
        return self.mask_checksum is not None


class CsrIntegrator2d(CSRIntegrator):

    def __init__(self,
                 image_size,
                 lut=None,
                 empty=0.0,
                 bin_centers0=None,
                 bin_centers1=None,
                 checksum=None,
                 mask_checksum=None):
        """Constructor of the abstract class for 2D integration
        
        :param size: input image size
        :param lut: tuple of 3 arrays with data, indices and indptr,
                     index of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param bin_center: position of the bin center
        :param checksum: checksum for the LUT, if not provided, recalculated
        :param mask_checksum: just a place-holder to track which mask was used

        Nota: bins are deduced from bin_centers0, bin_centers1 
    
        """
        self.bin_centers0 = bin_centers0
        self.bin_centers1 = bin_centers1
        if not checksum:
            self.checksum = calc_checksum(lut[0])
        else:
            self.checksum = checksum
        CSRIntegrator.__init__(self, image_size, lut, empty)

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
                  poissonian=False,
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
        :param poissonian: set to True to variance=max(signal,1), False will implement azimuthal variance
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :return: Integrate2dtpl namedtuple: "radial azimuthal intensity error signal variance normalization count"
        
        """
        if variance is None and poissonian is None:
            do_variance = False
        else:
            do_variance = True
        trans = CSRIntegrator.integrate(self, signal, variance, poissonian, dummy, delta_dummy,
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

    integrate_ng = integrate

    @property
    def check_mask(self):
        return self.mask_checksum is not None
