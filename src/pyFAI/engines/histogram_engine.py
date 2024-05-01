#
#    Copyright (C) 2019-2024 European Synchrotron Radiation Facility, Grenoble, France
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

"""simple histogram rebinning engine implemented in pure python (with the help of numpy !)
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/04/2024"
__status__ = "development"

import logging
logger = logging.getLogger(__name__)
import numpy
from ..utils import EPS32
from .preproc import preproc as preproc_np
try:
    from ..ext.preproc import preproc as preproc_cy
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    preproc = preproc_np
else:
    preproc = preproc_cy

from ..containers import Integrate1dtpl, Integrate2dtpl, ErrorModel


def histogram1d_engine(radial, npt,
                       raw,
                       dark=None,
                       flat=None,
                       solidangle=None,
                       polarization=None,
                       absorption=None,
                       mask=None,
                       dummy=None,
                       delta_dummy=None,
                       normalization_factor=1.0,
                       empty=None,
                       split_result=False,
                       variance=None,
                       dark_variance=None,
                       error_model=ErrorModel.NO,
                       weighted_average=True,
                       radial_range=None
                       ):
    """Implementation of rebinning engine using pure numpy histograms

    :param radial: radial position 2D array (same shape as raw)
    :param npt: number of points to integrate over
    :param raw: 2D array with the raw signal
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: 2d array of int/bool: non-null where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: provide an estimation of the variance
    :param dark_variance: provide an estimation of the variance of the dark_current,
    :param error_model: Use the provided ErrorModel, only "poisson" and "variance" is valid
    :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average

    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are left at 0.

    Nota: "azimuthal_range" has to be integrated into the
           mask prior to the call of this function

    :return: Integrate1dtpl named tuple containing:
            position, average intensity, std on intensity,
            plus the various histograms on signal, variance, normalization and count.

    """
    error_model = ErrorModel.parse(error_model)
    prep = preproc(raw,
                   dark=dark,
                   flat=flat,
                   solidangle=solidangle,
                   polarization=polarization,
                   absorption=absorption,
                   mask=mask,
                   dummy=dummy,
                   delta_dummy=delta_dummy,
                   normalization_factor=normalization_factor,
                   split_result=4,
                   variance=variance,
                   dark_variance=dark_variance,
                   error_model=error_model,
                   empty=0,
                   apply_normalization=not weighted_average,
                   )
    radial = radial.ravel()
    prep.shape = -1, 4
    assert prep.shape[0] == radial.size
    if radial_range is None:
        radial_range = (radial.min(), radial.max() * EPS32)

    histo_signal, _ = numpy.histogram(radial, npt, weights=prep[:, 0], range=radial_range)
    if error_model == ErrorModel.AZIMUTHAL:
        raise NotImplementedError("Numpy histogram are not (YET) able to assess variance in azimuthal bins")
    elif error_model.do_variance:  # Variance, Poisson and Hybrid
        histo_variance, _ = numpy.histogram(radial, npt, weights=prep[:, 1], range=radial_range)
        histo_normalization2, _ = numpy.histogram(radial, npt, weights=prep[:, 2] ** 2, range=radial_range)
    else:  # No error propagated
        std = sem = histo_variance = histo_normalization2 = None

    histo_normalization, _ = numpy.histogram(radial, npt, weights=prep[:, 2], range=radial_range)
    histo_count, position = numpy.histogram(radial, npt, weights=numpy.round(prep[:, 3]).astype(int), range=radial_range)
    positions = (position[1:] + position[:-1]) / 2.0

    mask_empty = histo_count < 1e-6
    if dummy is not None:
        empty = dummy
    with numpy.errstate(divide='ignore', invalid='ignore'):
        intensity = histo_signal / histo_normalization
        intensity[mask_empty] = empty
        if error_model.do_variance:
            std = numpy.sqrt(histo_variance / histo_normalization2)
            sem = numpy.sqrt(histo_variance) / histo_normalization
            std[mask_empty] = empty
            sem[mask_empty] = empty
        else:
            std = sem = None
    return Integrate1dtpl(positions,
                          intensity,
                          sem,
                          histo_signal,
                          histo_variance,
                          histo_normalization,
                          histo_count,
                          std,
                          sem,
                          histo_normalization2)


def histogram2d_engine(radial, azimuthal, bins,
                       raw,
                       dark=None,
                       flat=None,
                       solidangle=None,
                       polarization=None,
                       absorption=None,
                       mask=None,
                       dummy=None,
                       delta_dummy=None,
                       normalization_factor=1.0,
                       empty=None,
                       variance=None,
                       dark_variance=None,
                       error_model=ErrorModel.NO,
                       weighted_average=True,
                       radial_range=None,
                       azimuth_range=None,
                       allow_radial_neg=False,
                       chiDiscAtPi=True,
                       clip_pos1=True
                       ):
    """Implementation of 2D rebinning engine using pure numpy histograms

    :param radial: radial position 2D array (same shape as raw)
    :param azimuthal: azimuthal position 2D array (same shape as raw)
    :param bins: number of points to integrate over in (radial, azimuthal) dimensions
    :param raw: 2D array with the raw signal
    :param dark: array containing the value of the dark noise, to be subtracted
    :param flat: Array containing the flatfield image. It is also checked for dummies if relevant.
    :param solidangle: the value of the solid_angle. This processing may be performed during the rebinning instead. left for compatibility
    :param polarization: Correction for polarization of the incident beam
    :param absorption: Correction for absorption in the sensor volume
    :param mask: 2d array of int/bool: non-null where data should be ignored
    :param dummy: value of invalid data
    :param delta_dummy: precision for invalid data
    :param normalization_factor: final value is divided by this
    :param empty: value to be given for empty bins
    :param variance: provide an estimation of the variance
    :param dark_variance: provide an estimation of the variance of the dark_current,
    :param error_model: set to "poisson" for assuming the detector is poissonian and variance = raw + dark
    :param bool weighted_average: set to False to use an unweighted mean (similar to legacy) instead of the weighted average
    :param radial_range: enforce boundaries in radial dimention, 2tuple with lower and upper bound
    :param azimuth_range: enforce boundaries in azimuthal dimention, 2tuple with lower and upper bound
    :param allow_radial_neg: clip negative radial position (can a dimention be negative ?)
    :param chiDiscAtPi: boolean; by default the chi_range is in the range ]-pi,pi[ set to 0 to have the range ]0,2pi[ TODO: unimplemented
    :param clip_pos1: clip the azimuthal range to [-pi pi] (or [0 2pi]), set to False to deactivate behavior TODO: unimplemented



    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are left at 0.

    Nota: "azimuthal_range" has to be integrated into the
           mask prior to the call of this function

    :return: Integrate1dtpl named tuple containing:
            position, average intensity, std on intensity,
            plus the various histograms on signal, variance, normalization and count.

    """
    error_model = ErrorModel.parse(error_model)
    prep = preproc(raw,
                   dark=dark,
                   flat=flat,
                   solidangle=solidangle,
                   polarization=polarization,
                   absorption=absorption,
                   mask=mask,
                   dummy=dummy,
                   delta_dummy=delta_dummy,
                   normalization_factor=normalization_factor,
                   split_result=4,
                   variance=variance,
                   dark_variance=dark_variance,
                   error_model=error_model,
                   empty=0,
                   apply_normalization=not weighted_average,
                   )
    radial = radial.ravel()
    azimuthal = azimuthal.ravel()
    prep.shape = -1, 4
    assert prep.shape[0] == radial.size
    assert prep.shape[0] == azimuthal.size
    npt = tuple(max(1, i) for i in bins)
    if radial_range is None:
        if allow_radial_neg:
            radial_range = [radial.min(), radial.max() * EPS32]
        else:
            radial_range = [max(0, radial.min()), radial.max() * EPS32]
    if azimuth_range is None:
            azimuth_range = [azimuthal.min(), azimuthal.max() * EPS32]

    rng = [radial_range, azimuth_range]
    histo_signal, _, _ = numpy.histogram2d(radial, azimuthal, npt, weights=prep[:, 0], range=rng)
    histo_normalization, _, _ = numpy.histogram2d(radial, azimuthal, npt, weights=prep[:, 2], range=rng)
    histo_count, position_rad, position_azim = numpy.histogram2d(radial, azimuthal, npt, weights=prep[:, 3], range=rng)

    histo_signal = histo_signal.T
    histo_normalization = histo_normalization.T
    histo_count = histo_count.T

    if error_model == ErrorModel.AZIMUTHAL:
        raise NotImplementedError("Numpy histogram are not (YET) able to assess variance in azimuthal bins")
    elif error_model.do_variance:  # Variance, Poisson and Hybrid
        histo_variance, _, _ = numpy.histogram2d(radial, azimuthal, npt, weights=prep[:, 1], range=rng)
        histo_normalization2, _, _ = numpy.histogram2d(radial, azimuthal, npt, weights=prep[:, 2] ** 2, range=rng)
        histo_variance = histo_variance.T
        histo_normalization2 = histo_normalization2.T
    else:  # No error propagated
        std = sem = histo_variance = histo_normalization2 = None

    bins_azim = 0.5 * (position_azim[1:] + position_azim[:-1])
    bins_rad = 0.5 * (position_rad[1:] + position_rad[:-1])

    mask_empty = (histo_count == 0)
    if dummy is not None:
        empty = dummy
    with numpy.errstate(divide='ignore', invalid='ignore'):
        intensity = histo_signal / histo_normalization
        intensity[mask_empty] = empty
        if error_model.do_variance:
            std = numpy.sqrt(histo_variance / histo_normalization2)
            sem = numpy.sqrt(histo_variance) / histo_normalization
            std[mask_empty] = empty
            sem[mask_empty] = empty
    return Integrate2dtpl(bins_rad, bins_azim, intensity, sem, histo_signal, histo_variance, histo_normalization, histo_count, std, sem, histo_normalization2)
