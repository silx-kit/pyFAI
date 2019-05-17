# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2016-2018 European Synchrotron Radiation Facility, Grenoble, France
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

"""Bayesian evaluation of background for 1D powder diffraction pattern.

Code according to Sivia and David, J. Appl. Cryst. (2001). 34, 318-324

* Version: 0.1 2012/03/28
* Version: 0.2 2016/10/07: OOP implementation
"""

from __future__ import absolute_import, print_function, division

__authors__ = ["Vincent Favre-Nicolin", "Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/05/2019"
__status__ = "development"
__docformat__ = 'restructuredtext'

import numpy
from scipy.interpolate import UnivariateSpline, RectBivariateSpline
from scipy import optimize


class BayesianBackground(object):
    """This class estimates the background of a powder diffraction pattern

    http://journals.iucr.org/j/issues/2001/03/00/he0278/he0278.pdf

    The log likelihood is described in correspond to eq7 of the paper:

    .. math:: z =  y / sigma^2

    * if z<0 a quadratic behaviour is expected
    * if z>>1 it is likely a bragg peak so the penalty should be small: log(z).
    * The spline is used to have a quadratic behaviour near 0 and the log one
      near the junction

    The threshold is taken at 8 as erf is 1 above 6:
    The points 6, 7 and 8 are used in the spline to ensure a continuous junction
    with the logarithmic continuation.
    """
    s1 = None
    PREFACTOR = 1

    @classmethod
    def classinit(cls):
        # Spline depends on integration constant...
        # from quadratic behaviour near 0 to logarithmic one between 6 and 8
        splinex = numpy.array([0.,
                               1e-6,
                               1e-5,
                               1e-4,
                               1e-3,
                               1e-2,
                               1e-1,
                               1.1,
                               2.1,
                               3.1,
                               4.1,
                               5.1,
                               6.1,
                               7.1,
                               8.1])

        spliney = numpy.array([0.0,
                               1e-12,
                               1e-10,
                               1e-8,
                               1e-6,
                               1e-4,
                               1.77123249e-03,
                               1.00997634,
                               2.89760310,
                               3.61881096,
                               3.93024374,
                               4.16063018,
                               4.34600620,
                               4.50155649,
                               4.63573160])
        cls.spline = UnivariateSpline(splinex, spliney, s=0)
        cls.s1 = cls.spline(8.0) - numpy.log(8.0)

    def __init__(self):
        if self.s1 is None:
            self.classinit()

    @classmethod
    def bayes_llk_negative(cls, z):
        "used to calculate the log-likelihood of negative values: quadratic"
        return cls.PREFACTOR * z * z

    @classmethod
    def bayes_llk_small(cls, z):
        "used to calculate the log-likelihood of small positive values: fitted with spline"
        return cls.spline(z)

    @classmethod
    def bayes_llk_large(cls, z):
        "used to calculate the log-likelihood of large positive values: logarithmic"
        return cls.s1 + numpy.log(abs(z))

    @classmethod
    def bayes_llk(cls, z):
        """Calculate actually the log-likelihood from a set of weighted error

        Re implementation of the following code even slightly faster:

        .. code-block:: python

            (y<=0)*5*y**2 + (y>0)*(y<8)*pyFAI.utils.bayes.background.spline(y) + (y>=8)*(s1+log(abs(y)+1*(y<8)))

        :param float[:] z: weighted error
        :return: log likelihood
        :rtype: float[:]
        """

        llk = numpy.zeros_like(z)
        neg = (z < 0)
        llk[neg] = cls.bayes_llk_negative(z[neg])
        small = numpy.logical_and(z > 0, z < 8)
        llk[small] = cls.bayes_llk_small(z[small])
        large = (z >= 8)
        llk[large] = cls.bayes_llk_large(z[large])
        return llk

    @classmethod
    def test_bayes_llk(cls):
        """Test plot of log(likelihood)
        Similar to as figure 3 of Sivia and David, J. Appl. Cryst. (2001). 34, 318-324
        """
        x = numpy.linspace(-5, 15, 2001)
        y = -cls.bayes_llk(x)
        return(x, y)

    @classmethod
    def func_min(cls, y0, x_obs, y_obs, w_obs, x0, k):
        """ Function to optimize

        :param y0: values of the background
        :param x_obs: experimental values
        :param y_obs: experimental values
        :param w_obs: weights of the experimental points
        :param x0: position of evaluation of the spline
        :param k: order of the spline, usually 3
        :return: sum of the log-likelihood to be minimized
        """
        s = UnivariateSpline(x0, y0, s=0, k=k)
        tmp = cls.bayes_llk(w_obs * (y_obs - s(x_obs))).sum()
        return tmp

    def __call__(self, x, y, sigma=None, npt=40, k=3):
        """Function like class instance...

        :param float[:] x: coordinates along the horizontal axis
        :param float[:] y: coordinates along the vertical axis
        :param float[:] sigma: error along the vertical axis
        :param int npt: number of points of the fitting spline
        :param int k: order of the fitted spline.
        :return: the background for y
        :rtype: float[:]

        Nota: Due to spline function, one needs: npt >= k + 1
        """
        if sigma is None:
            # assume sigma=sqrt(yobs) !
            w_obs = 1.0 / numpy.sqrt(y)
        else:
            w_obs = 1.0 / sigma
        # deal with 0-variance points
        mask = numpy.logical_not(numpy.isnan(w_obs))
        x_obs = x[mask]
        y_obs = y[mask]
        w_obs = w_obs[mask]
        x0 = numpy.linspace(x.min(), x.max(), npt)
        y0 = numpy.zeros(npt) + y_obs.mean()
        # Minimize
        y1 = optimize.fmin_powell(self.func_min, y0,
                                  args=(x_obs, y_obs, w_obs, x0, k),
                                  disp=False)
        # Result
        y_calc = UnivariateSpline(x0, y1, s=0, k=k)(x)
        return y_calc

    @classmethod
    def func2d_min(cls, values, d0_sparse, d1_sparse, d0_pos, d1_pos, y_obs, w_obs, valid, k):
        """ Function to optimize

        :param values: values of the background on spline knots
        :param d0_sparse: positions along slowest axis of the spline knots
        :param d1_pos: positions along fastest axis of the spline knots
        :param d0_pos: positions along slowest axis (all coordinates)
        :param d1_pos: positions along fastest axis (all coordinates)
        :param y_obs: intensities actually measured
        :param w_obs: weights of the experimental points
        :param valid: coordinated of valid pixels
        :param k: order of the spline, usually 3
        :return: sum of the log-likelihood to be minimized
        """
        values = values.reshape(d0_sparse.size, d1_sparse.size)
        spline = RectBivariateSpline(d0_sparse, d1_sparse, values, kx=k, ky=k)
        bg = spline(d0_pos, d1_pos)
        err = w_obs * (y_obs - bg)
        if valid is not True:
            err = err.take(valid)
        else:
            err = err.ravel()
        sum_err = cls.bayes_llk(err).sum()
        return sum_err

    def background_image(self, img, sigma=None, mask=None, npt=10, k=3):
        shape = img.shape
        if sigma is not None:
            assert sigma.shape == shape
        else:
            sigma = numpy.sqrt(img)

        w = 1 / sigma

        mask_nan = numpy.isnan(w)
        if mask is not None:
            assert mask.shape == shape
            mask = numpy.logical_or(mask_nan, mask)
        else:
            mask = mask_nan

        if mask.sum() == 0:
            valid = numpy.where(numpy.logical_not(mask))
        else:
            valid = True
        d0_pos = numpy.arange(0, shape[0])
        d1_pos = numpy.arange(0, shape[1])

        d0_sparse = numpy.linspace(0, shape[0], npt)
        d1_sparse = numpy.linspace(0, shape[1], npt)

        y0 = numpy.zeros((npt, npt)) + img.mean()
        y1 = optimize.fmin_powell(self.func2d_min, y0,
                                  args=(d0_sparse, d1_sparse, d0_pos, d1_pos, img, w, valid, k),
                                  disp=True, callback=lambda x: print(x))

        values = y1.reshape(d0_sparse.size, d1_sparse.size)
        spline = RectBivariateSpline(d0_sparse, d1_sparse, values, k, k)
        bg = spline(d0_pos, d1_pos)
        return bg


background = BayesianBackground()
