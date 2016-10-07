# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) 2016 European Synchrotron Radiation Facility, Grenoble, France
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

""" Bayesian evaluation of background for 1D powder diffraction pattern
 
Code according to Sivia and David, J. Appl. Cryst. (2001). 34, 318-324
# Version: 0.1 2012/03/28
# Version: 0.2 2016/10/07: OOP implementation
"""

from __future__ import absolute_import, print_function, division

__authors__ = ["Vincent Favre-Nicolin", "Jérôme Kieffer"]
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "07/10/2016"
__status__ = "development"
__docformat__ = 'restructuredtext'

import numpy
from scipy.interpolate import UnivariateSpline
from scipy import optimize


class BayesianBackground(object):
    """This class estimates the background of a powder diffraction pattern
    
    http://journals.iucr.org/j/issues/2001/03/00/he0278/he0278.pdf
    
    The log likelihood is described in correspond to eq7 of the paper:
    z = y/sigma^2
    * if z<0: a quadratic behaviour is expected
    * if z>>1 it is likely a bragg peak so the penalty should be small: log(z). 
    * The spline is used to have a quadratic behaviour near 0 and the log one 
      near the junction   
    
    The threshold is taken at 8 as erf is 1 above 6: 
    The points 6, 7 and 8 are used in the spline to ensure a continuous junction 
    with the logarithmic continuation 
     
    """
    s1 = None

    @classmethod
    def classinit(cls):
#         print("init")
        # Spline depends on integration constant...
        # from quadratic behaviour near 0 to logarithmic one between 6 and 8
        spliney = numpy.array([0.00000000e+00,
                               1e-4,
                               1.77123249e-03,
                               1.00997634e+00,
                               2.89760310e+00,
                               3.61881096e+00,
                               3.93024374e+00,
                               4.16063018e+00,
                               4.34600620e+00,
                               4.50155649e+00,
                               4.63573160e+00])
        splinex = numpy.array([0.,
                               0.01,
                               0.1,
                               1.1,
                               2.1,
                               3.1,
                               4.1,
                               5.1,
                               6.1,
                               7.1,
                               8.1])
        cls.spline = UnivariateSpline(splinex, spliney, s=0)
        cls.s1 = cls.spline(8.0) - numpy.log(8.0)

    def __init__(self):
        if self.s1 is None:
            self.classinit()

    @staticmethod
    def bayes_llk_negative(z):
        "used to calculate the log-likelihood of negative values: quadratic"
        return 5 * z ** 2

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
        """Calculate actually the log-likelyhood from a set of weighted error
        
        Re implementation of:
        (y<=0)*5*y**2 + (y>0)*(y<8)*pyFAI.utils.bayes.background.spline(y) + (y>=8)*(s1+log(abs(y)+1*(y<8)))
        even slightly faster
        
        :param float[:] z: weighted error
        :return float[:]: log likelihood 
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
        x = numpy.linspace(-5, 15, 501)
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

    def __call__(self, x, y, sigma=None, npt=40):
        """
        :param float[:] x: coordinates along the horizontal axis
        :param float[:] y: coordinates along the vertical axis
        :param float[:] sigma: error along the vertical axis
        :param int npt: number of points of the fitting spline
        :return float[:]: the background for y 
        """
        if sigma is None:
            # assume sigma=sqrt(yobs) !
            w_obs = 1.0 / y
        else:
            w_obs = 1.0 / sigma ** 2
        # deal with 0-variance points
        mask = numpy.logical_not(numpy.isnan(w_obs))
        x_obs = x[mask]
        y_obs = y[mask]
        w_obs = w_obs[mask]
        k = 3  # spline parameter
        x0 = numpy.linspace(x_obs.min(), x_obs.max(), npt)
        y0 = numpy.zeros(npt) + y_obs.mean()
        # Minimize
        y1 = optimize.fmin_powell(self.func_min, y0, args=(x_obs, y_obs, w_obs, x0, k))
        # Result
        y_calc = UnivariateSpline(x0, y1, s=0, k=k)(x)
        return y_calc

background = BayesianBackground()
