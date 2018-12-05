# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "04/12/2018"

import logging
import numpy
import collections

from silx.image import marchingsquares
import pyFAI.utils
from ...geometryRefinement import GeometryRefinement
from .model.GeometryConstraintsModel import GeometryConstraintsModel
from ..peak_picker import PeakPicker
from ..utils import timeutils

_logger = logging.getLogger(__name__)


class GeometryRefinementContext(object):
    """Store the full context of the GeometryRefinement object

    Right now, GeometryRefinement store the bound but do not store the fixed
    constraints. It make the context difficult to manage and to trust.
    """

    PARAMETERS = ["wavelength", "dist", "poni1", "poni2", "rot1", "rot2", "rot3"]

    def __init__(self, *args, **kwargs):
        self.__geoRef = GeometryRefinement(*args, **kwargs)
        fixed = pyFAI.utils.FixedParameters()
        fixed.add("wavelength")
        self.__fixed = fixed

        self.__bounds = {}
        attrs = ("wavelength", "dist", "poni1", "poni2", "rot1", "rot2", "rot3")
        for name in attrs:
            min_getter = getattr(self.__geoRef, "get_%s_min" % name)
            max_getter = getattr(self.__geoRef, "get_%s_max" % name)
            minValue, maxValue = min_getter(), max_getter()
            self.__bounds[name] = minValue, maxValue

    def __getattr__(self, name):
        return object.__getattribute__(self.__geoRef, name)

    def __setattr__(self, name, value):
        if "__" in name:
            return super(GeometryRefinementContext, self).__setattr__(name, value)
        return object.__setattr__(self.__geoRef, name, value)

    def bounds(self):
        return self.__bounds

    def fixed(self):
        return self.__fixed

    def setFixed(self, fixed):
        self.__fixed = fixed

    def setBounds(self, bounds):
        self.__bounds = bounds

    def setParams(self, params):
        """Set the fit parameter values from the list of values"""
        for value, name in zip(params, self.PARAMETERS):
            setattr(self.__geoRef, name, value)

    def getParams(self):
        """Returns list of parameters"""
        return [getattr(self.__geoRef, p) for p in self.PARAMETERS]

    def chi2(self):
        if "wavelength" in self.__fixed:
            chi2 = self.__geoRef.chi2()
        else:
            chi2 = self.__geoRef.chi2_wavelength()
        return chi2

    def refine(self, maxiter):
        attrs = ["wavelength", "dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        for name in attrs:
            if name in self.__fixed:
                continue
            min_setter = getattr(self.__geoRef, "set_%s_min" % name)
            max_setter = getattr(self.__geoRef, "set_%s_max" % name)
            if name in self.__bounds:
                minValue, maxValue = self.__bounds[name]
            else:
                minValue, maxValue = -float("inf"), float("inf")
            min_setter(minValue)
            max_setter(maxValue)

        if "wavelength" in self.__fixed:
            deltaS = self.__geoRef.refine2(maxiter, self.__fixed)
        else:
            deltaS = self.__geoRef.refine2_wavelength(maxiter, self.__fixed)
        return deltaS


class RingCalibration(object):

    def __init__(self, image, mask, calibrant, detector, wavelength, peaks, method):
        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        self.__calibrant.set_wavelength(wavelength)
        self.__detector = detector
        self.__wavelength = wavelength
        self.__rms = None
        self.__previousRms = None
        self.__defaultConstraints = None

        self.__init(peaks, method)

    def __initGeoRef(self):
        """
        Tries to initialise the GeometryRefinement (dist, poni, rot)
        Returns a dictionary of key value pairs
        """
        defaults = {"dist": 0.1, "poni1": 0.0, "poni2": 0.0,
                    "rot1": 0.0, "rot2": 0.0, "rot3": 0.0}
        if self.__detector:
            try:
                p1, p2, _p3 = self.__detector.calc_cartesian_positions()
                defaults["poni1"] = p1.max() / 2.
                defaults["poni2"] = p2.max() / 2.
            except Exception as err:
                _logger.warning(err)
        # if ai:
        #    for key in defaults.keys():  # not PARAMETERS which holds wavelength
        #        val = getattr(self.ai, key, None)
        #        if val is not None:
        #            defaults[key] = val
        return defaults

    def __init(self, peaks, method, constraintsModel=None):

        if len(peaks) == 0:
            self.__peakPicker = None
            self.__geoRef = None
            return

        scores = []
        defaultParams = self.__initGeoRef()

        geoRef = GeometryRefinementContext(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaultParams)
        self.__geoRef = geoRef

        # Store the default constraints
        self.__defaultConstraints = GeometryConstraintsModel()
        self.toGeometryConstraintsModel(self.__defaultConstraints)

        # First attempt

        geoRef = GeometryRefinementContext(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaultParams)
        self.__geoRef = geoRef
        if constraintsModel is not None:
            assert(constraintsModel.isValid())
            self.fromGeometryConstraintsModel(constraintsModel)
        rms = geoRef.refine(1000000)
        score = geoRef.chi2()
        parameters = geoRef.getParams()
        scores.append((score, parameters, rms))

        # Second attempt

        geoRef = GeometryRefinementContext(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaultParams)
        self.__geoRef = geoRef
        geoRef.guess_poni()
        if constraintsModel is not None:
            assert(constraintsModel.isValid())
            self.fromGeometryConstraintsModel(constraintsModel)
        rms = geoRef.refine(1000000)
        score = geoRef.chi2()
        parameters = geoRef.getParams()
        scores.append((score, parameters, rms))

        # Use the better one
        scores.sort()
        _score, parameters, rms = scores[0]
        geoRef.setParams(parameters)

        self.__rms = rms
        self.__previousRms = None

        peakPicker = PeakPicker(data=self.__image,
                                calibrant=self.__calibrant,
                                wavelength=self.__wavelength,
                                detector=self.__detector,
                                method=method)

        self.__peakPicker = peakPicker
        self.__geoRef = geoRef

    def init(self, peaks, method, constraintsModel):
        self.__init(peaks, method, constraintsModel=constraintsModel)

    def update(self, image, mask, calibrant, detector, wavelength=None):
        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        self.__detector = detector
        if wavelength is not None:
            self.__wavelength = wavelength

    def getPyfaiGeometry(self):
        return self.__geoRef

    def __computeRms(self):
        if self.__geoRef is None:
            return None
        chi2 = self.__geoRef.chi2()
        return numpy.sqrt(chi2 / self.__geoRef.data.shape[0])

    def refine(self, max_iter=500, seconds=10):
        """
        Contains the common geometry refinement part
        """
        self.__calibrant.set_wavelength(self.__wavelength)
        self.__peakPicker.points.setWavelength_change2th(self.__wavelength)

        self.__previousRms = self.__rms
        residual = previous_residual = float("+inf")

        print("Initial residual: %s" % previous_residual)

        count = 0
        timer = timeutils.Timer(seconds=10)

        while count < max_iter and not timer.isTimeout():
            residual = self.__geoRef.refine(10000)
            if residual >= previous_residual:
                break
            previous_residual = residual
            count += 1

        self.__rms = residual
        print("Final residual: %s (after %s iterations)" % (residual, count))

        self.__geoRef.del_ttha()
        self.__geoRef.del_dssa()
        self.__geoRef.del_chia()

    def getRms(self):
        """Returns the RMS (root mean square) computed from the current fitting.

        The unit is the radian.
        """
        if self.__rms is None:
            self.__rms = self.__computeRms()
        return self.__rms

    def getPreviousRms(self):
        """Returns the previous RMS computed before the last fitting.

        The unit is the radian.
        """
        return self.__previousRms

    def getTwoThetaArray(self):
        """
        Returns the 2th array corresponding to the calibrated image
        """
        # 2th array is cached insided
        tth = self.__geoRef.twoThetaArray(self.__peakPicker.shape)
        return tth

    def getRings(self):
        """
        Returns polygons of rings

        :returns: List of ring angle with the associated polygon
        :rtype: List[Tuple[float,List[numpy.ndarray]]]
        """
        tth = self.__geoRef.twoThetaArray(self.__peakPicker.shape)

        result = collections.OrderedDict()

        tth_max = tth.max()
        tth_min = tth.min()
        if not self.__calibrant:
            return result

        angles = [i for i in self.__calibrant.get_2th()
                  if (i is not None) and (i >= tth_min) and (i <= tth_max)]
        if len(angles) == 0:
            return result

        ms = marchingsquares.MarchingSquaresMergeImpl(tth, self.__mask, use_minmax_cache=True)
        rings = []
        for angle in angles:
            polygons = ms.find_contours(angle)
            rings.append((angle, polygons))

        return rings

    def getBeamCenter(self):
        try:
            f2d = self.__geoRef.getFit2D()
            x, y = f2d["centerX"], f2d["centerY"]
        except TypeError:
            return None

        # Check if this pixel really contains the beam center
        # If the detector contains gap, it is not always the case
        ax, ay = numpy.array([x]), numpy.array([y])
        tth = self.__geoRef.tth(ay, ax)[0]
        if tth >= 0.001:
            return None
        return y, x

    def getPoni(self):
        """"Returns the PONI coord in image coordinate.

        That's an approximation of the PONI coordinate at pixel precision
        """
        solidAngle = self.__geoRef.solidAngleArray(shape=self.__image.shape)
        index = numpy.argmax(solidAngle)
        coord = numpy.unravel_index(index, solidAngle.shape)
        dmin = self.__geoRef.dssa.min()
        dmax = self.__geoRef.dssa.max()
        if dmax > 1 - (dmax - dmin) * 0.001:
            return coord
        else:
            return None

    def toGeometryModel(self, model):
        model.lockSignals()
        if self.__geoRef is None:
            model.wavelength().setValue(None)
            model.distance().setValue(None)
            model.poni1().setValue(None)
            model.poni2().setValue(None)
            model.rotation1().setValue(None)
            model.rotation2().setValue(None)
            model.rotation3().setValue(None)
        else:
            model.wavelength().setValue(self.__geoRef.wavelength)
            model.distance().setValue(self.__geoRef.dist)
            model.poni1().setValue(self.__geoRef.poni1)
            model.poni2().setValue(self.__geoRef.poni2)
            model.rotation1().setValue(self.__geoRef.rot1)
            model.rotation2().setValue(self.__geoRef.rot2)
            model.rotation3().setValue(self.__geoRef.rot3)
        model.unlockSignals()

    def fromGeometryModel(self, model, resetResidual=True):
        wavelength = model.wavelength().value()
        self.__calibrant.setWavelength_change2th(wavelength)
        self.__geoRef.wavelength = wavelength
        self.__geoRef.dist = model.distance().value()
        self.__geoRef.poni1 = model.poni1().value()
        self.__geoRef.poni2 = model.poni2().value()
        self.__geoRef.rot1 = model.rotation1().value()
        self.__geoRef.rot2 = model.rotation2().value()
        self.__geoRef.rot3 = model.rotation3().value()
        if resetResidual:
            self.__previousRms = None
            self.__rms = None

    def toGeometryConstraintsModel(self, contraintsModel, reachFromGeoRef=True):
        if reachFromGeoRef is False:
            raise NotImplementedError("Not implemented")
        attrs = [
            ("wavelength", contraintsModel.wavelength()),
            ("dist", contraintsModel.distance()),
            ("poni1", contraintsModel.poni1()),
            ("poni2", contraintsModel.poni2()),
            ("rot1", contraintsModel.rotation1()),
            ("rot2", contraintsModel.rotation2()),
            ("rot3", contraintsModel.rotation3()),
        ]
        bounds = self.__geoRef.bounds()
        fixed = self.__geoRef.fixed()
        for name, constraint in attrs:
            minValue, maxValue = bounds[name]
            constraint.setRangeConstraint(minValue, maxValue)
            if name in fixed:
                constraint.setFixed()

    def defaultGeometryConstraintsModel(self):
        """Returns the default constraints

        Not the one used, but the one initially set to the refinement engine.
        """
        assert(self.__defaultConstraints is not None)
        return self.__defaultConstraints

    def fromGeometryConstraintsModel(self, constraintsModel):
        attrs = [
            ("wavelength", constraintsModel.wavelength()),
            ("dist", constraintsModel.distance()),
            ("poni1", constraintsModel.poni1()),
            ("poni2", constraintsModel.poni2()),
            ("rot1", constraintsModel.rotation1()),
            ("rot2", constraintsModel.rotation2()),
            ("rot3", constraintsModel.rotation3()),
        ]
        fixed = pyFAI.utils.FixedParameters()
        bounds = {}
        for name, constraint in attrs:
            if constraint.isFixed():
                fixed.add(name)
            elif constraint.isRangeConstrained():
                minValue, maxValue = constraint.range()
                if minValue is None:
                    minValue = -float("inf")
                if maxValue is None:
                    maxValue = +float("inf")
                bounds[name] = minValue, maxValue
        self.__geoRef.setFixed(fixed)
        self.__geoRef.setBounds(bounds)
