# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2022 European Synchrotron Radiation Facility, Grenoble, France
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

"""Module used to perform the geometric refinement of the model
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/10/2024"
__status__ = "development"

import os
import copy
import tempfile
import subprocess
import logging
import numpy
import math
from math import pi
from .integrator.azimuthal import AzimuthalIntegrator
from .calibrant import Calibrant, CALIBRANT_FACTORY
from .utils.ellipse import fit_ellipse
from .utils.decorators import deprecated
from scipy.optimize import fmin, leastsq, fmin_slsqp

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import basinhopping as anneal
except ImportError:
    from scipy.optimize import anneal
try:
    from scipy.optimize import curve_fit
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    curve_fit = None

if os.name != "nt":
    WindowsError = RuntimeError

ROCA = "/opt/saxs/roca"

####################
# GeometryRefinement
####################


class GeometryRefinement(AzimuthalIntegrator):
    PARAM_ORDER = ("dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength")

    def __init__(self, data=None, calibrant=None,
                 dist=1, poni1=None, poni2=None,
                 rot1=0, rot2=0, rot3=0,
                 pixel1=None, pixel2=None, splineFile=None, detector=None,
                 wavelength=None, **kwargs):
        """
        :param data: ndarray float64 shape = n, 3
            col0: pos in dim0 (in pixels)
            col1: pos in dim1 (in pixels)
            col2: ring index in calibrant object
        :param calibrant: instance of pyFAI.calibrant.Calibrant containing the d-Spacing

        :param dist: guessed sample-detector distance (optional, in m)
        :param poni1: guessed PONI coordinate along the Y axis (optional, in m)
        :param poni2: guessed PONI coordinate along the X axis (optional, in m)
        :param rot1: guessed tilt of the detector around the Y axis (optional, in rad)
        :param rot2: guessed tilt of the detector around the X axis (optional, in rad)
        :param rot3: guessed tilt of the detector around the incoming beam axis (optional, in rad)
        :param pixel1: Pixel size along the vertical direction of the detector (in m), almost mandatory
        :param pixel2: Pixel size along the horizontal direction of the detector (in m), almost mandatory
        :param splineFile: file describing the detector as 2 cubic splines. Replaces pixel1 & pixel2
        :param detector: name of the detector or Detector instance. Replaces splineFile, pixel1 & pixel2
        :param wavelength: wavelength in m (1.54e-10)


        """
        if data is None:
            self.data = None
        else:
            self.data = numpy.array(data, dtype=numpy.float64)
            if self.data.ndim != 2:
                raise RuntimeError("data is expected to be of shape (nb control-points, [3|4])")
            if self.data.shape[1] not in (3, 4):
                raise RuntimeError("data shape's last dim should be 3 for non weighted or 4 for weighted refinement")
            if self.data.shape[0] == 0:
                raise RuntimeError("expected at least one control point !")

        if (pixel1 is None) and (pixel2 is None) and (splineFile is None) and (detector is None):
            raise RuntimeError("Setting up the geometry refinement without knowing the detector makes little sense")
        super().__init__(dist, 0, 0,
                         rot1, rot2, rot3,
                         pixel1, pixel2, splineFile, detector,
                         wavelength=wavelength, **kwargs)

        if calibrant is None:
            self.calibrant = Calibrant()
        else:
            if isinstance(calibrant, Calibrant):
                self.calibrant = calibrant
            elif isinstance(calibrant, str):
                if calibrant in CALIBRANT_FACTORY:
                    self.calibrant = CALIBRANT_FACTORY(calibrant)
                else:
                    self.calibrant = Calibrant(filename=calibrant)
            else:
                self.calibrant = Calibrant(calibrant)

        self.calibrant.setWavelength_change2th(self.wavelength)

        if (poni1 is None) or (poni2 is None):
            self.guess_poni()
        else:
            if math.isfinite(poni1) and math.isfinite(poni2):
                self.poni1 = float(poni1)
                self.poni2 = float(poni2)
            else:
                self.guess_poni()

        self._dist_min = 0.0
        self._dist_max = 35.0
        self._poni1_min = -10000.0 * self.pixel1
        self._poni1_max = 15000.0 * self.pixel1
        self._poni2_min = -10000.0 * self.pixel2
        self._poni2_max = 15000.0 * self.pixel2
        self._rot1_min = -pi
        self._rot1_max = pi
        self._rot2_min = -pi
        self._rot2_max = pi
        self._rot3_min = -pi
        self._rot3_max = pi
        self._wavelength_min = 1e-15
        self._wavelength_max = 100.e-10

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        data = copy.deepcopy(self.data, memo=memo)
        dist=copy.deepcopy(self._dist, memo=memo)
        poni1=copy.deepcopy(self._poni1, memo=memo)
        poni2=copy.deepcopy(self._poni2, memo=memo)
        rot1=copy.deepcopy(self._rot1, memo=memo)
        rot2=copy.deepcopy(self._rot2, memo=memo)
        rot3=copy.deepcopy(self._rot3, memo=memo)
        pixel1=copy.deepcopy(self.detector.pixel1, memo=memo)
        pixel2=copy.deepcopy(self.detector.pixel2, memo=memo)
        splineFile=copy.deepcopy(self.detector.splineFile, memo=memo)
        detector = copy.deepcopy(self.detector, memo=memo)
        wavelength=copy.deepcopy(self.wavelength, memo=memo)
        calibrant=copy.deepcopy(self.calibrant, memo=memo)

        new = self.__class__(data=data,
                             dist=dist,
                             poni1=poni1,
                             poni2=poni2,
                             rot1=rot1,
                             rot2=rot2,
                             rot3=rot3,
                             pixel1=pixel1,
                             pixel2=pixel2,
                             splineFile=splineFile,
                             detector=detector,
                             wavelength=wavelength,
                             calibrant=calibrant
                              )
        numerical = ["_dist", "_poni1", "_poni2", "_rot1", "_rot2", "_rot3",
                     "chiDiscAtPi", "_dssa_order", "_wavelength",
                     '_oversampling', '_correct_solid_angle_for_spline',
                     '_transmission_normal',
                     "_dist_min", "_dist_max", "_poni1_min", "_poni1_max", "_poni2_min", "_poni2_max",
                     "_rot1_min", "_rot1_max", "_rot2_min", "_rot2_max", "_rot3_min", "_rot3_max",
                     "_wavelength_min", "_wavelength_max"]
        memo[id(self)] = new
        for key in numerical:
            old_value = self.__getattribute__(key)
            memo[id(old_value)] = old_value
            new.__setattr__(key, old_value)
        new_param = [new._dist, new._poni1, new._poni2,
                     new._rot1, new._rot2, new._rot3]
        memo[id(self.param)] = new_param
        new.param = new_param
        cached = {}
        memo[id(self._cached_array)] = cached
        for key, old_value in self._cached_array.copy().items():
            if "copy" in dir(old_value):
                new_value = old_value.copy()
                memo[id(old_value)] = new_value
        new._cached_array = cached
        return new

    def guess_poni(self, fixed=None):
        """PONI can be guessed by the centroid of the ring with lowest 2Theta

        It may try to fit an ellipse and sometimes it works
        """
        if self.data is None or len(self.data) == 0:
            logger.warning("No input data, not guessing the PONI")
            return
        if len(self.calibrant.dSpacing):
            # logger.warning(self.calibrant.__repr__())s
            tth = self.calc_2th(self.data[:, 2])
        else:  # assume rings are in decreasing dSpacing in the file
            tth = self.data[:, 2]
        asrt = tth.argsort()
        tth = tth[asrt]
        srtdata = self.data[asrt]
        tth_min = tth.min()
        smallRing = srtdata[tth < (tth_min + 1e-6)]
        smallRing1 = smallRing[:, 0]
        smallRing2 = smallRing[:, 1]
        smallRing_in_m = self.detector.calc_cartesian_positions(smallRing1,
                                                                smallRing2)
        nbpt = len(smallRing)
        worked = False
        if nbpt > 5:
            # If there are many control point on the inner-most ring, fit an ellipse
            try:
                ellipse = fit_ellipse(*smallRing_in_m[:2])
                direct_dist = ellipse.half_long_axis / numpy.tan(tth_min)
                tilt = numpy.arctan2(ellipse.half_long_axis - ellipse.half_short_axis, ellipse.half_short_axis)
                cos_tilt = numpy.cos(tilt)
                sin_tilt = numpy.sin(tilt)
                angle = (ellipse.angle + numpy.pi / 2.0) % numpy.pi
                cos_tpr = numpy.cos(angle)
                sin_tpr = numpy.sin(angle)
                dist = direct_dist * cos_tilt
                poni1 = ellipse.center_1 - direct_dist * sin_tilt * sin_tpr
                poni2 = ellipse.center_2 - direct_dist * sin_tilt * cos_tpr
                rot2 = numpy.arcsin(sin_tilt * sin_tpr)  # or pi-
                rot1 = numpy.arccos(min(1.0, max(-1.0, (cos_tilt / numpy.sqrt(1 - sin_tpr * sin_tpr * sin_tilt * sin_tilt)))))  # + or -
                if cos_tpr * sin_tilt > 0:
                    rot1 = -rot1
                rot3 = 0
            except ValueError:
                worked = False
            else:
                if numpy.isnan(dist + poni1 + poni2 + rot1 + rot2 + rot3):
                    worked = False
                else:
                    worked = True
                    self.update_values(dist=dist, poni1=poni1, poni2=poni2,
                                       rot1=rot1, rot2=rot2, rot3=rot3,
                                       fixed=fixed)
        if not worked:
            poni1 = smallRing_in_m[0].sum() / nbpt
            poni2 = smallRing_in_m[1].sum() / nbpt
            self.update_values(poni1=poni1, poni2=poni2, fixed=fixed)

    def update_values(self, dist=None, wavelength=None, poni1=None, poni2=None,
                      rot1=None, rot2=None, rot3=None, fixed=None):
        """Update values taking care of fixed parameters.
        """
        # TODO: Take care of ranges too
        if fixed is None:
            fixed = set([])
        if dist is not None and "dist" not in fixed:
            self.dist = dist
        if wavelength is not None and "wavelength" not in fixed:
            self.wavelength = wavelength
        if poni1 is not None and "poni1" not in fixed:
            self.poni1 = poni1
        if poni2 is not None and "poni2" not in fixed:
            self.poni2 = poni2
        if rot1 is not None and "rot1" not in fixed:
            self.rot1 = rot1
        if rot2 is not None and "rot2" not in fixed:
            self.rot2 = rot2
        if rot3 is not None and "rot3" not in fixed:
            self.rot3 = rot3

    def set_tolerance(self, value=10):
        """
        Set the tolerance for a refinement of the geometry; in percent of the original value

        :param value: Tolerance as a percentage

        """
        low = 1.0 - value / 100.
        hi = 1.0 + value / 100.
        self.dist_min = low * self.dist
        self.dist_max = hi * self.dist
        if abs(self.poni1) > (value / 100.) ** 2:
            self.poni1_min = min(low * self.poni1, hi * self.poni1)
            self.poni1_max = max(low * self.poni1, hi * self.poni1)
        else:
            self.poni1_min = -(value / 100.) ** 2
            self.poni1_max = (value / 100.) ** 2
        if abs(self.poni2) > (value / 100.) ** 2:
            self.poni2_min = min(low * self.poni2, hi * self.poni2)
            self.poni2_max = max(low * self.poni2, hi * self.poni2)
        else:
            self.poni2_min = -(value / 100.) ** 2
            self.poni2_max = (value / 100.) ** 2
        if abs(self.rot1) > (value / 100.) ** 2:
            self.rot1_min = min(low * self.rot1, hi * self.rot1)
            self.rot1_max = max(low * self.rot1, hi * self.rot1)
        else:
            self.rot1_min = -(value / 100.) ** 2
            self.rot1_max = (value / 100.) ** 2
        if abs(self.rot2) > (value / 100.) ** 2:
            self.rot2_min = min(low * self.rot2, hi * self.rot2)
            self.rot2_max = max(low * self.rot2, hi * self.rot2)
        else:
            self.rot2_min = -(value / 100.) ** 2
            self.rot2_max = (value / 100.) ** 2
        if abs(self.rot3) > (value / 100.) ** 2:
            self.rot3_min = min(low * self.rot3, hi * self.rot3)
            self.rot3_max = max(low * self.rot3, hi * self.rot3)
        else:
            self.rot3_min = -(value / 100.) ** 2
            self.rot3_max = (value / 100.) ** 2
        self.wavelength_min = low * self.wavelength
        self.wavelength_max = hi * self.wavelength

    def calc_2th(self, rings, wavelength=None):
        """
        :param rings: indices of the rings. starts at 0 and self.dSpacing should be long enough !!!
        :param wavelength: wavelength in meter
        """
        if wavelength is None:
            wavelength = self.wavelength
        if wavelength is None or wavelength <= 0.0:
            return numpy.array([numpy.finfo("float32").max] * len(rings))
        rings = numpy.ascontiguousarray(rings, dtype=numpy.int32)

        if wavelength != self.calibrant.wavelength:
            self.calibrant.setWavelength_change2th(wavelength)
        ary = self.calibrant.get_2th()
        if len(ary) < rings.max():
            # complete turn ~ 2pi ~ 7: help the optimizer to find the right way
            ary += [10.0 * (rings.max() - len(ary))] * (1 + rings.max() - len(ary))
        tth = numpy.array(ary, dtype=numpy.float64)
        if rings.max() >= len(tth):
            raise IndexError("Ring indices %s are not all available at this wavelength (%s)" % (numpy.unique(rings), wavelength))
        return tth[rings]

    def calc_param7(self, param, free, const):
        """Calculate the "legacy" 6/7 parameters from a number of free and fixed parameters"""
        param7 = [   ]
        for name in self.PARAM_ORDER:
            if name in free:
                value = param[free.index(name)]
                if name == "wavelength":
                    param7.append(value * 1e-10)
                else:
                    param7.append(value)
            else:
                param7.append(const[name])
        return param7

    def residu1(self, param, d1, d2, rings):
        return self.tth(d1, d2, param) - self.calc_2th(rings, self.wavelength)

    def residu1_wavelength(self, param, d1, d2, rings):
        return self.tth(d1, d2, param) - self.calc_2th(rings, param[6] * 1e-10)

    def residu2(self, param, d1, d2, rings):
        # dot product is faster ...
        # return (self.residu1(param, d1, d2, rings) ** 2).sum()
        t = self.residu1(param, d1, d2, rings)
        return numpy.dot(t, t)

    def residu2_weighted(self, param, d1, d2, rings, weight):
        # return (weight * self.residu1(param, d1, d2, rings) ** 2).sum()
        t = weight * self.residu1(param, d1, d2, rings)
        return numpy.dot(t, t)

    def residu2_wavelength(self, param, d1, d2, rings):
        # return (self.residu1_wavelength(param, d1, d2, rings) ** 2).sum()
        t = self.residu1_wavelength(param, d1, d2, rings)
        return numpy.dot(t, t)

    def residu2_wavelength_weighted(self, param, d1, d2, rings, weight):
        # return (weight * self.residu1_wavelength(param, d1, d2, rings) ** 2).sum()
        t = weight * self.residu1_wavelength(param, d1, d2, rings)
        return numpy.dot(t, t)

    def residu3(self, param, free, const, d1, d2, rings, weights=None):
        "Preform the calculation of $sum_(2\theta_e-2\theta_i)²$"
        param7 = self.calc_param7(param, free, const)
        delta_theta = self.tth(d1, d2, param7[:6]) - self.calc_2th(rings, param7[6])
        if weights is not None:
            delta_theta *= weights
        return numpy.dot(delta_theta, delta_theta)

    def refine1(self):
        self.param = numpy.array([self._dist, self._poni1, self._poni2,
                                  self._rot1, self._rot2, self._rot3],
                                 dtype=numpy.float64)
        new_param, rc = leastsq(self.residu1, self.param,
                                args=(self.data[:, 0],
                                      self.data[:, 1],
                                      self.data[:, 2]))
        oldDeltaSq = self.chi2(tuple(self.param))
        newDeltaSq = self.chi2(tuple(new_param))
        logger.info("Least square retcode=%s %s --> %s",
                    rc, oldDeltaSq, newDeltaSq)
        if newDeltaSq < oldDeltaSq:
            i = abs(self.param - new_param).argmax()
            d = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
            logger.info("maxdelta on %s: %s --> %s ",
                        d[i], self.param[i], new_param[i])
            self.set_param(new_param)
            return newDeltaSq
        else:
            return oldDeltaSq

    def refine3(self, maxiter=1000000, fix=None):
        """
        Same as refine2 except it does not rely on upper_bound == lower_bound to fix parameters

        This is a work around the regression introduced with scipy 1.5

        :param maxiter: maximum number of iteration for finding the solution
        :param fix: parameters to be fixed. Does not assume the wavelength to be fixed by default
        :return: $sum_(2\theta_e-2\theta_i)²$
        """
        npt, ncol = self.data.shape
        if  ncol >= 3:
            pos0 = self.data[:, 0]
            pos1 = self.data[:, 1]
            ring = self.data[:, 2].astype(numpy.int32)
        if ncol == 4:
            weight = self.data[:, 3]
        else:
            weight = None

        fix = [] if fix is None else fix

        free = []
        param = []
        bounds = []
        const = {}
        for name in self.PARAM_ORDER:
            value = getattr(self, name)
            if name in fix:
                const[name] = value
            else:
                minmax = (getattr(self, "_%s_min" % name), getattr(self, "_%s_max" % name))
                if name == "wavelength":
                    # enforces an upper limit to the wavelength depending on the number of rings.
                    max_wavelength = self.calibrant.get_max_wavelength(ring.max())
                    value = min(value, max_wavelength)
                    value = value * 1e10
                    minmax = (1e10 * minmax[0], 1e10 * min(minmax[1], max_wavelength))
                free.append(name)
                param.append(value)
                bounds.append(minmax)
        param = numpy.array(param)

        old_delta_theta2 = self.residu3(param, free, const, pos0, pos1, ring, weight) / npt

        new_param = fmin_slsqp(self.residu3, param, iter=maxiter,
                               args=(free, const, pos0, pos1, ring, weight),
                               bounds=bounds,
                               acc=1.0e-12,
                               iprint=(logger.getEffectiveLevel() <= logging.INFO))
        # new_param7 = self.calc_param7(new_param, free, const)

        new_delta_theta2 = self.residu3(new_param, free, const, pos0, pos1, ring, weight) / npt

        logger.info("Constrained Least square %s --> %s", old_delta_theta2, new_delta_theta2)

        if new_delta_theta2 < old_delta_theta2:
            i = abs(param - new_param).argmax()

            logger.info("maxdelta on %s: %s --> %s ",
                        free[i], param[i], new_param[i])

            param7 = self.calc_param7(new_param, free, const)
            self.set_param(param7)
            return new_delta_theta2
        else:
            return old_delta_theta2

    def refine2(self, maxiter=1000000, fix=None):
        if not fix:
            fix = ["wavelength"]
        return self.refine3(maxiter=maxiter, fix=fix)

    def refine2_wavelength(self, maxiter=1000000, fix=None):
        """Refine all parameters including the wavelength.

        This implies that it enforces an upper limit to the wavelength depending
        on the number of rings.
        """
        if fix is None:
            fix = []
        return self.refine3(maxiter=maxiter, fix=fix)

    def simplex(self, maxiter=1000000):
        self.param = numpy.array([self.dist, self.poni1, self.poni2,
                                  self.rot1, self.rot2, self.rot3],
                                 dtype=numpy.float64)
        new_param = fmin(self.residu2, self.param,
                         args=(self.data[:, 0],
                               self.data[:, 1],
                               self.data[:, 2]),
                         maxiter=maxiter,
                         xtol=1.0e-12)
        oldDeltaSq = self.chi2(tuple(self.param)) / self.data.shape[0]
        newDeltaSq = self.chi2(tuple(new_param)) / self.data.shape[0]
        logger.info("Simplex %s --> %s", oldDeltaSq, newDeltaSq)
        if newDeltaSq < oldDeltaSq:
            i = abs(self.param - new_param).argmax()
            d = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
            logger.info("maxdelta on %s : %s --> %s ",
                        d[i], self.param[i], new_param[i])
            self.set_param(new_param)
            return newDeltaSq
        else:
            return oldDeltaSq

    def anneal(self, maxiter=1000000):
        self.param = [self.dist, self.poni1, self.poni2,
                      self.rot1, self.rot2, self.rot3]
        result = anneal(self.residu2, self.param,
                        args=(self.data[:, 0],
                              self.data[:, 1],
                              self.data[:, 2]),
                        lower=[self._dist_min,
                               self._poni1_min,
                               self._poni2_min,
                               self._rot1_min,
                               self._rot2_min,
                               self._rot3_min],
                        upper=[self._dist_max,
                               self._poni1_max,
                               self._poni2_max,
                               self._rot1_max,
                               self._rot2_max,
                               self._rot3_max],
                        maxiter=maxiter)
        new_param = result[0]
        oldDeltaSq = self.chi2() / self.data.shape[0]
        newDeltaSq = self.chi2(new_param) / self.data.shape[0]
        logger.info("Anneal  %s --> %s", oldDeltaSq, newDeltaSq)
        if newDeltaSq < oldDeltaSq:
            i = abs(self.param - new_param).argmax()
            d = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
            logger.info("maxdelta on %s : %s --> %s ",
                        d[i], self.param[i], new_param[i])
            self.set_param(new_param)
            return newDeltaSq
        else:
            return oldDeltaSq

    def chi2(self, param=None):
        if param is None:
            param = self.param[:]
        return self.residu2(param,
                            self.data[:, 0], self.data[:, 1], self.data[:, 2])

    def chi2_wavelength(self, param=None):
        if param is None:
            param = self.param
            if len(param) == 6:
                param.append(1e10 * self.wavelength)
        return self.residu2_wavelength(param,
                                       self.data[:, 0],
                                       self.data[:, 1],
                                       self.data[:, 2])

    def curve_fit(self, with_rot=True):
        """Refine the geometry and provide confidence interval
        Use curve_fit from scipy.optimize to not only refine the geometry (unconstrained fit)

        :param with_rot: include rotation intro error measurment
        :return: std_dev, confidence
        """
        if not curve_fit:
            import scipy
            logger.error("curve_fit method needs a newer scipy: at lease scipy 0.9, you are running: %s", scipy.version.version)
        d1 = self.data[:, 0]
        d2 = self.data[:, 1]
        size = d1.size
        x = d1, d2
        rings = self.data[:, 2].astype(numpy.int32)

        def f_with_rot(x, *param):
            return self.tth(x[0], x[1], numpy.concatenate((param, [self.rot3])))

        def f_no_rot(x, *param):
            return self.tth(x[0], x[1], numpy.concatenate((param, [self.rot1, self.rot2, self.rot3])))

        y = self.calc_2th(rings, self.wavelength)
        param0 = numpy.array([self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3], dtype=numpy.float64)
        ref = self.residu2(param0, d1, d2, rings)
        print("param0: %s %s" % (param0, ref))
        if with_rot:
            popt, pcov = curve_fit(f_with_rot, x, y, param0[:-1])
            popt = numpy.concatenate((popt, [self.rot3]))
        else:
            popt, pcov = curve_fit(f_no_rot, x, y, param0[:-3])
            popt = numpy.concatenate((popt, [self.rot1, self.rot2, self.rot3]))
        obt = self.residu2(popt, d1, d2, rings)
        print("param1: %s %s" % (popt, obt))
        print(pcov)
        err = numpy.sqrt(numpy.diag(pcov))
        print("err: %s" % err)
        if obt < ref:
            self.set_param(popt)
        error = {}
        confidence = {}
        for k, v in zip(("dist", "poni1", "poni2", "rot1", "rot2", "rot3"), err):
            error[k] = v
            confidence[k] = 1.96 * v / numpy.sqrt(size)

        print("Std dev  as sqrt of the diag of covariance:\n%s" % error)
        print("Confidence as 1.95 sigma/sqrt(n):\n%s" % confidence)
        return error, confidence

    def confidence(self, with_rot=True):
        """Confidence interval obtained from the second derivative of the error function
        next to its minimum value.

        Note the confidence interval increases with the number of points which is "surprizing"

        :param with_rot: if true include rot1 & rot2 in the parameter set.
        :return: std_dev, confidence
        """
        epsilon = 1e-5
        d1 = self.data[:, 0]
        d2 = self.data[:, 1]
        r = self.data[:, 2].astype(numpy.int32)
        param0 = numpy.array([self.dist, self.poni1, self.poni2, self.rot1, self.rot2, self.rot3], dtype=numpy.float64)
        ref = self.residu2(param0, d1, d2, r)
        print(ref)
        if with_rot:
            size = 5
        else:
            size = 3
        hessian = numpy.zeros((size, size), dtype=numpy.float64)

        delta = abs(epsilon * param0)
        delta[abs(param0) < epsilon] = epsilon
        print(delta)
        for i in range(size):
            # Diagonal terms:
            deltai = delta[i]
            param = param0.copy()
            param[i] += deltai
            value_plus = self.residu2(param, d1, d2, r)
            param = param0.copy()
            param[i] -= deltai
            value_moins = self.residu2(param, d1, d2, r)
            hessian[i, i] = (value_plus + value_moins - 2.0 * ref) / (deltai ** 2)

            for j in range(i + 1, size):
                # if i == j: continue
                deltaj = delta[j]
                param = param0.copy()
                param[i] += deltai
                param[j] += deltaj
                value_plus_plus = self.residu2(param, d1, d2, r)
                param = param0.copy()
                param[i] -= deltai
                param[j] -= deltaj
                value_moins_moins = self.residu2(param, d1, d2, r)
                param = param0.copy()
                param[i] += deltai
                param[j] -= deltaj
                value_plus_moins = self.residu2(param, d1, d2, r)
                param = param0.copy()
                param[i] -= deltai
                param[j] += deltaj
                value_moins_plus = self.residu2(param, d1, d2, r)
                hessian[j, i] = hessian[i, j] = (value_plus_plus + value_moins_moins - value_plus_moins - value_moins_plus) / (4.0 * deltai * deltaj)
        print(hessian)
        w, v = numpy.linalg.eigh(hessian)
        print("eigen val: %s" % w)
        print("eigen vec: %s" % v)
        cov = numpy.linalg.inv(hessian)
        print(cov)
        err = numpy.sqrt(numpy.diag(cov))
        print("err: %s" % err)
        error = {}
        for k, v in zip(("dist", "poni1", "poni2", "rot1", "rot2", "rot3"), err):
            error[k] = v
        confidence = {}
        for i, k in enumerate(("dist", "poni1", "poni2", "rot1", "rot2", "rot3")):
            if i < size:
                confidence[k] = numpy.sqrt(ref / hessian[i, i])
        print("std_dev as sqrt of the diag of inv hessian:\n%s" % error)
        print("Convidence as sqrt of the error function /  hessian:\n%s" % confidence)
        return error, confidence

    @deprecated
    def roca(self):
        """
        run roca to optimise the parameter set
        """
        tmpf = tempfile.NamedTemporaryFile()
        for line in self.data:
            tmpf.write("%s %s %s %s" % (line[2], line[0], line[1], os.linesep))
        tmpf.flush()
        roca = subprocess.Popen(
            [ROCA, "debug=8", "maxdev=1", "input=" + tmpf.name,
             str(self.pixel1), str(self.pixel2),
             str(self.poni1 / self.pixel1), str(self.poni2 / self.pixel2),
             str(self.dist), str(self.rot1), str(self.rot2), str(self.rot3)],
            stdout=subprocess.PIPE)
        new_param = [self.dist, self.poni1, self.poni2,
                     self.rot1, self.rot2, self.rot3]
        for line in roca.stdout:
            word = line.split()
            if len(word) == 3:
                if word[0] == "cen1":
                    new_param[1] = float(word[1]) * self.pixel1
                if word[0] == "cen2":
                    new_param[2] = float(word[1]) * self.pixel2
                if word[0] == "dis":
                    new_param[0] = float(word[1])
                if word[0] == "rot1":
                    new_param[3] = float(word[1])
                if word[0] == "rot2":
                    new_param[4] = float(word[1])
                if word[0] == "rot3":
                    new_param[5] = float(word[1])
        print("Roca %s --> %s" % (self.chi2() / self.data.shape[0], self.chi2(new_param) / self.data.shape[0]))
        if self.chi2(tuple(new_param)) < self.chi2(tuple(self.param)):
            self.param = new_param
            self.dist, self.poni1, self.poni2, \
                self.rot1, self.rot2, self.rot3 = tuple(new_param)

        tmpf.close()

    def set_dist_max(self, value):
        if isinstance(value, float):
            self._dist_max = value
        else:
            self._dist_max = float(value)

    def get_dist_max(self):
        return self._dist_max

    dist_max = property(get_dist_max, set_dist_max)

    def set_dist_min(self, value):
        if isinstance(value, float):
            self._dist_min = value
        else:
            self._dist_min = float(value)

    def get_dist_min(self):
        return self._dist_min

    dist_min = property(get_dist_min, set_dist_min)

    def set_poni1_min(self, value):
        if isinstance(value, float):
            self._poni1_min = value
        else:
            self._poni1_min = float(value)

    def get_poni1_min(self):
        return self._poni1_min

    poni1_min = property(get_poni1_min, set_poni1_min)

    def set_poni1_max(self, value):
        if isinstance(value, float):
            self._poni1_max = value
        else:
            self._poni1_max = float(value)

    def get_poni1_max(self):
        return self._poni1_max

    poni1_max = property(get_poni1_max, set_poni1_max)

    def set_poni2_min(self, value):
        if isinstance(value, float):
            self._poni2_min = value
        else:
            self._poni2_min = float(value)

    def get_poni2_min(self):
        return self._poni2_min

    poni2_min = property(get_poni2_min, set_poni2_min)

    def set_poni2_max(self, value):
        if isinstance(value, float):
            self._poni2_max = value
        else:
            self._poni2_max = float(value)

    def get_poni2_max(self):
        return self._poni2_max

    poni2_max = property(get_poni2_max, set_poni2_max)

    def set_rot1_min(self, value):
        if isinstance(value, float):
            self._rot1_min = value
        else:
            self._rot1_min = float(value)

    def get_rot1_min(self):
        return self._rot1_min

    rot1_min = property(get_rot1_min, set_rot1_min)

    def set_rot1_max(self, value):
        if isinstance(value, float):
            self._rot1_max = value
        else:
            self._rot1_max = float(value)

    def get_rot1_max(self):
        return self._rot1_max

    rot1_max = property(get_rot1_max, set_rot1_max)

    def set_rot2_min(self, value):
        if isinstance(value, float):
            self._rot2_min = value
        else:
            self._rot2_min = float(value)

    def get_rot2_min(self):
        return self._rot2_min

    rot2_min = property(get_rot2_min, set_rot2_min)

    def set_rot2_max(self, value):
        if isinstance(value, float):
            self._rot2_max = value
        else:
            self._rot2_max = float(value)

    def get_rot2_max(self):
        return self._rot2_max

    rot2_max = property(get_rot2_max, set_rot2_max)

    def set_rot3_min(self, value):
        if isinstance(value, float):
            self._rot3_min = value
        else:
            self._rot3_min = float(value)

    def get_rot3_min(self):
        return self._rot3_min

    rot3_min = property(get_rot3_min, set_rot3_min)

    def set_rot3_max(self, value):
        if isinstance(value, float):
            self._rot3_max = value
        else:
            self._rot3_max = float(value)

    def get_rot3_max(self):
        return self._rot3_max

    rot3_max = property(get_rot3_max, set_rot3_max)

    def set_wavelength_min(self, value):
        if isinstance(value, float):
            self._wavelength_min = value
        else:
            self._wavelength_min = float(value)

    def get_wavelength_min(self):
        return self._wavelength_min

    wavelength_min = property(get_wavelength_min, set_wavelength_min)

    def set_wavelength_max(self, value):
        if isinstance(value, float):
            self._wavelength_max = value
        else:
            self._wavelength_max = float(value)

    def get_wavelength_max(self):
        return self._wavelength_max

    wavelength_max = property(get_wavelength_max, set_wavelength_max)
