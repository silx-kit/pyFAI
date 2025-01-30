#
#    Copyright (C) 2017-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""CSC rebinning engine implemented in pure python (with bits of scipy !)
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/09/2024"
__status__ = "development"

import logging
import warnings
logger = logging.getLogger(__name__)
import numpy
from scipy.sparse import csc_matrix
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
from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel


class CSCIntegrator(object):

    def __init__(self,
                 input_size,
                 output_size,
                 lut=None,
                 empty=0.0):
        """Constructor of the abstract class

        :param input_size: input image size
        :param output_size: output histogram size (int or 2-tuple)
        :param lut: tuple of 3 arrays with data, indices and indptr,
                     index of the start of line in the CSC matrix
        :param empty: value for empty pixels
        """
        self.size = input_size
        self.preprocessed = numpy.empty((input_size, 4), dtype=numpy.float32)
        self.empty = empty
        self.bins = output_size
        self._csc = None
        self._csc2 = None  # Used for propagating variance
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
        if self.size > len(indptr) - 1:
            new_indptr = numpy.empty(self.size + 1, indptr.dtype)
            new_indptr[:] = indptr[-1]
            new_indptr[:len(indptr)] = indptr
            indptr = new_indptr
        nbins = numpy.prod(self.bins)
        if len(indices):
            assert max(indices) <= nbins
        self._csc = csc_matrix((data, indices, indptr), shape=(nbins, self.size))
        self._csc2 = csc_matrix((data * data, indices, indptr), shape=(nbins, self.size))  # contains the coef squared, used for variance propagation

    def integrate(self,
                  signal,
                  variance=None,
                  error_model=None,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidangle=None,
                  polarization=None,
                  absorption=None,
                  normalization_factor=1.0,
                  weighted_average=True,
                  ):
        """Actually perform the CSR matrix multiplication after preprocessing.

        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param error_model: Enum or string, set to "poisson" to use signal as variance (minimum 1), set to "azimuthal" to use azimuthal model.
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average
        :return: the preprocessed data integrated as array nbins x 4 which contains:
                    regrouped signal, variance, normalization, pixel count, sum_norm²

        Nota: all normalizations are grouped in the preprocessing step.
        """
        error_model = ErrorModel.parse(error_model)
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
                       error_model=error_model,
                       apply_normalization=not weighted_average,
                       out=self.preprocessed)
        prep.shape = numpy.prod(shape), 4
        flat_sig, flat_var, flat_nrm, flat_cnt = prep.T  # should create views!
        res = numpy.empty((numpy.prod(self.bins), 5), dtype=numpy.float32)
        res[:, 0] = self._csc.dot(flat_sig)  # Σ c·x
        res[:, 2] = self._csc.dot(flat_nrm)  # Σ c·ω
        res[:, 3] = self._csc.dot(flat_cnt)  # Σ c·1
        if error_model is ErrorModel.AZIMUTHAL:
            avg = res[:, 0] / res[:, 2]
            avg2d = self._csc.T.dot(avg)  # tranform 1D average into 2D (works only if splitting is disabled)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                delta = (flat_sig / flat_nrm - avg2d)
            res[:, 1] = self._csc2.dot((delta * flat_nrm) ** 2)  # Σ c²·ω²·(x-x̄ )²
            res[:, 4] = self._csc2.dot(flat_nrm ** 2)  # Σ c²·ω²
        elif error_model.do_variance:
            res[:, 1] = self._csc2.dot(flat_var)  # Σ c²·σ²
            res[:, 4] = self._csc2.dot(flat_nrm ** 2)  # Σ c²·ω²
        return res


class CscIntegrator1d(CSCIntegrator):

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
        CSCIntegrator.__init__(self, image_size, len(bin_centers), lut, empty)
        self.pos0_range = self.pos1_range = None
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (list, tuple)) else  str(unit).split("_")[0]
        self.mask_checksum = mask_checksum

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content

        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""

        CSCIntegrator.set_matrix(self, data, indices, indptr)
        assert len(self.bin_centers) == self.bins

    def integrate(self,
                  signal,
                  variance=None,
                  error_model=None,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidangle=None,
                  polarization=None,
                  absorption=None,
                  normalization_factor=1.0,
                  weighted_average=True,
                  ):
        """Actually perform the 1D integration

        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param error_model: Enum or str, set to "poisson" to use signal as variance (minimum 1), set to "azimuthal" to use azimuthal model.
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average
        :return: Integrate1dResult or Integrate1dWithErrorResult object depending on variance

        """
        error_model = ErrorModel.parse(error_model)
        if variance is not None:
            error_model = ErrorModel.VARIANCE
        trans = CSCIntegrator.integrate(self, signal, variance, error_model,
                                        dummy, delta_dummy,
                                        dark, flat, solidangle, polarization,
                                        absorption, normalization_factor, weighted_average)
        signal = trans[:, 0]
        variance = trans[:, 1]
        normalization = trans[:, 2]
        count = trans[:, 3]
        mask = (normalization == 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity = signal / normalization
            intensity[mask] = self.empty
            if error_model.do_variance:
                sum_nrm2 = trans[:, 4]
                std = numpy.sqrt(variance / sum_nrm2)
                sem = numpy.sqrt(variance) / normalization
                std[mask] = self.empty
                sem[mask] = self.empty
            else:
                variance = std = sem = sum_nrm2 = None
        return Integrate1dtpl(self.bin_centers,
                              intensity, sem,
                              signal, variance, normalization, count, std, sem, sum_nrm2)

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
        :param error_model: Enum or str, "azimuthal" or "poisson"
        :param normalization_factor: divide raw signal by this value
        :param cutoff: discard all points with ``|value - avg| > cutoff * sigma``. 3-4 is quite common
        :param cycle: perform at maximum this number of cycles. 5 is common.
        :return: namedtuple with "position intensity error signal variance normalization count"

        """
        shape = data.shape
        error_model = ErrorModel.parse(error_model)
        if error_model is ErrorModel.NO:
            logger.error("No variance propagation is incompatible with sigma-clipping. Using `azimuthal` model !")
            error_model = ErrorModel.AZIMUTHAL

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
                       error_model=error_model,
                       out=self.preprocessed)

        prep_flat = prep.reshape((numpy.prod(shape), 4))

        # First azimuthal integration:
        flat_sig, flat_var, flat_nrm, flat_cnt = prep_flat.T  # should create views!
        sum_sig = self._csc.dot(flat_sig)
        sum_nrm = self._csc.dot(flat_nrm)
        sum_nrm2 = self._csc2.dot(flat_nrm ** 2)
        cnt = self._csc.dot(flat_cnt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg = sum_sig / sum_nrm
            interp_filter(avg, avg)
            if error_model == ErrorModel.AZIMUTHAL:
                avg2d = self._csc.T.dot(avg)  # backproject the average value to the image
                msk = (flat_nrm == 0)
                delta = (flat_sig / flat_nrm - avg2d)
                delta[msk] = 0
                sum_var = self._csc2.dot((delta * flat_nrm) ** 2)
            else:
                sum_var = self._csc2.dot(flat_var)
            std = numpy.sqrt(sum_var / sum_nrm2)
            interp_filter(std, std)
            for _ in range(cycle):
                # Interpolate in 2D: TODO: can be skipped in the case of azimuthal...
                avg2d = self._csc.T.dot(avg)
                std2d = self._csc.T.dot(std)
                cnt2d = numpy.maximum(self._csc.T.dot(cnt), 3)  # Needed for Chauvenet criterion
                delta = abs(flat_sig / flat_nrm - avg2d)
                chauvenet = numpy.maximum(cutoff, numpy.sqrt(2.0 * numpy.log(cnt2d / numpy.sqrt(2.0 * numpy.pi))))
                msk2d = numpy.where(numpy.logical_not(abs(delta) <= chauvenet * std2d))
                # discard outlier pixel here:
                prep_flat[msk2d] = 0
                # subsequent integrations:
                sum_sig = self._csc.dot(flat_sig)
                sum_nrm = self._csc.dot(flat_nrm)
                sum_nrm2 = self._csc2.dot(flat_nrm ** 2)
                cnt = self._csc.dot(flat_cnt)
                avg = sum_sig / sum_nrm
                interp_filter(avg, avg)
                if error_model == ErrorModel.AZIMUTHAL:
                    avg2d = self._csc.T.dot(avg)  # backproject the average value to the image
                    msk = (flat_nrm == 0)
                    delta = (flat_sig / flat_nrm - avg2d)
                    delta[msk] = 0
                    sum_var = self._csc2.dot((delta * flat_nrm) ** 2)
                else:
                    sum_var = self._csc2.dot(flat_var)
                std = numpy.sqrt(sum_var / sum_nrm2)
                interp_filter(std, std)
            # Finally calculate the sem in addition to the std

            sem = std * numpy.sqrt(sum_nrm2) / sum_nrm
        # mask out remaining NaNs
        msk = sum_nrm <= 0
        avg[msk] = self.empty
        std[msk] = self.empty
        sem[msk] = self.empty

        # Here we return the standard deviation and not the standard error of the mean !
        return Integrate1dtpl(self.bin_centers, avg, std, sum_sig, sum_var, sum_nrm, cnt, std, sem, sum_nrm2)

    @property
    def check_mask(self):
        return self.mask_checksum is not None


class CscIntegrator2d(CSCIntegrator):

    def __init__(self,
                 image_size,
                 lut=None,
                 empty=0.0,
                 unit=None,
                 bin_centers0=None,
                 bin_centers1=None,
                 checksum=None,
                 mask_checksum=None):
        """Constructor of the abstract class for 2D integration

        :param image_size: input image size
        :param lut: tuple of 3 arrays with data, indices and indptr,
                     index of the start of line in the CSR matrix
        :param empty: value for empty pixels
        :param unit: unit to be used
        :param bin_center0: position of the bin center along dim0
        :param bin_center1: position of the bin center along dim1
        :param checksum: checksum for the LUT, if not provided, recalculated
        :param mask_checksum: just a place-holder to track which mask was used

        Nota: bins are deduced from bin_centers0, bin_centers1

        """
        self.bin_centers0 = bin_centers0
        self.bin_centers1 = bin_centers1
        bins = (len(bin_centers0), len(bin_centers1))
        self.unit = unit
        self.space = tuple(str(u).split("_")[0] for u in unit) if isinstance(unit, (list, tuple)) else  str(unit).split("_")[0]
        self.mask_checksum = mask_checksum

        if not checksum:
            self.checksum = calc_checksum(lut[0])
        else:
            self.checksum = checksum
        CSCIntegrator.__init__(self, image_size, bins, lut, empty)

    def set_matrix(self, data, indices, indptr):
        """Actually set the CSR sparse matrix content

        :param data: the non zero values NZV
        :param indices: the column number of the NZV
        :param indptr: the index of the start of line"""

        CSCIntegrator.set_matrix(self, data, indices, indptr)

        assert self.size == len(indptr) - 1

    def integrate(self,
                  signal,
                  variance=None,
                  error_model=ErrorModel.NO,
                  dummy=None,
                  delta_dummy=None,
                  dark=None,
                  flat=None,
                  solidangle=None,
                  polarization=None,
                  absorption=None,
                  normalization_factor=1.0,
                  weighted_average=True,
                  **kwargs):
        """Actually perform the 2D integration

        :param signal: array of the right size with the signal in it.
        :param variance: Variance associated with the signal
        :param error_model: enum ErrorModel
        :param dummy: values which have to be discarded (dynamic mask)
        :param delta_dummy: precision for dummy values
        :param dark: noise to be subtracted from signal
        :param flat: flat-field normalization array
        :param flat: solidangle normalization array
        :param polarization: :solidangle normalization array
        :param absorption: :absorption normalization array
        :param normalization_factor: scale all normalization with this scalar
        :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average
        :return: Integrate2dtpl namedtuple: "radial azimuthal intensity error signal variance normalization count"

        """
        error_model = ErrorModel.parse(error_model)
        do_variance = variance is not None or  error_model.do_variance
        trans = CSCIntegrator.integrate(self, signal, variance, error_model, dummy, delta_dummy,
                                        dark, flat, solidangle, polarization,
                                        absorption, normalization_factor, weighted_average=weighted_average)
        trans.shape = self.bins + (-1,)

        signal = trans[..., 0]
        variance = trans[..., 1]
        normalization = trans[..., 2]
        count = trans[..., 3]
        sum_nrm2 = trans[..., 4]

        mask = (normalization == 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            intensity = signal / normalization
            intensity[mask] = self.empty
            if do_variance:
                sem = numpy.sqrt(variance) / normalization
                std = numpy.sqrt(variance / sum_nrm2)
                sem[mask] = self.empty
                std[mask] = self.empty
                variance = variance.T
                sem = sem.T
                std = std.T
                sum_nrm2 = sum_nrm2.T
            else:
                variance = std = sem = sum_nrm2 = None
        return Integrate2dtpl(self.bin_centers0, self.bin_centers1,
                              intensity.T, sem,
                              signal.T, variance, normalization.T, count.T, std, sem, sum_nrm2)

    integrate_ng = integrate

    @property
    def check_mask(self):
        return self.mask_checksum is not None
