#
#    Copyright (C) 2019 European Synchrotron Radiation Facility, Grenoble, France
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

from __future__ import absolute_import, print_function, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/04/2019"
__status__ = "development"

import logging
logger = logging.getLogger(__name__)
import numpy
from .preproc import preproc as preproc_np
try:
    from ..ext.preproc import preproc as preproc_cy
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    preproc = preproc_np
else:
    preproc = preproc_cy

from ..containers import Integrate1dtpl, Integrate2dtpl


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
                       poissonian=False,
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
    :param poissonian: set to "True" for assuming the detector is poissonian and variance = raw + dark


    NaN are always considered as invalid values

    if neither empty nor dummy is provided, empty pixels are left at 0.
    
    Nota: "azimuthal_range" has to be integrated into the 
           mask prior to the call of this function 
    
    :return: Integrate1dtpl named tuple containing: 
            position, average intensity, std on intensity, 
            plus the various histograms on signal, variance, normalization and count.  
                                               
    """
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
                   split_result=True,
                   variance=variance,
                   dark_variance=dark_variance,
                   poissonian=poissonian,
                   empty=0
                   )
    print(prep.dtype, prep.shape)
    radial = radial.ravel()
    prep.shape = -1, 4
    histo_signal, position = numpy.histogram(radial, npt, weights=prep[0], range=radial_range)
    histo_variance, position = numpy.histogram(radial, npt, weights=prep[1], range=radial_range)
    histo_normalization, position = numpy.histogram(radial, npt, weights=prep[2], range=radial_range)
    histo_count, position = numpy.histogram(radial, npt, weights=prep[3], range=radial_range)
    positions = (position[1:] + position[:-1]) / 2.0
    with numpy.errstate(divide='ignore'):
        intensity = histo_variance / histo_normalization
        error = numpy.sqrt(histo_variance) / histo_normalization
    empty_bins = numpy.where(histo_count == 0)
    if dummy is not None:
        empty = dummy
    intensity[empty_bins] = empty
    error[empty_bins] = empty
    return Integrate1dtpl(positions, intensity, error, histo_signal, histo_variance, histo_normalization, histo_count)
