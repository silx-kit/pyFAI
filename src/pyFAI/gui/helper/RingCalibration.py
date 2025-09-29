# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2025 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "12/09/2025"

import logging
import numpy
import collections

from ... import units
from ...geometryRefinement import GeometryRefinement
from ..model.GeometryConstraintsModel import GeometryConstraintsModel
from ..peak_picker import PeakPicker
from ..utils import timeutils
from ...containers import FixedParameters

_logger = logging.getLogger(__name__)
inf = numpy.inf


class GeometryRefinementContext(object):
    """Store the full context of the GeometryRefinement object

    Right now, GeometryRefinement store the bound but do not store the fixed
    constraints. It make the context difficult to manage and to trust.
    """

    PARAMETERS = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]

    def __init__(self, *args, **kwargs):
        _logger.debug("GeometryRefinementContext.__init__")
        self.__geoRef = GeometryRefinement(*args, **kwargs)
        self.__fixed = FixedParameters(["rot3", "wavelength"])

        self.__bounds = {}
        for name in self.PARAMETERS:
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
            param = numpy.array([self._dist, self._poni1, self._poni2,
                                 self._rot1, self._rot2, self._rot3],
                                dtype=numpy.float64)
            chi2 = self.__geoRef.chi2(param)
        else:
            param = numpy.array([self._dist, self._poni1, self._poni2,
                                 self._rot1, self._rot2, self._rot3,
                                 1e10 * self.wavelength],
                                dtype=numpy.float64)
            chi2 = self.__geoRef.chi2_wavelength(param)
        return chi2

    def guess_poni(self):
        self.__geoRef.guess_poni(self.__fixed)

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
                minValue, maxValue = -inf, inf
            min_setter(minValue)
            max_setter(maxValue)

        try:
            deltaS = self.__geoRef.refine3(maxiter, self.__fixed)
        except Exception:
            _logger.error("Error while refining the geometry", exc_info=True)
            return inf
        else:
            return deltaS


class RingCalibration:

    def __init__(self, image, mask, calibrant, detector, wavelength, peaks, method):
        self.__image = image
        self.__mask = mask
        calibrant.wavelength = wavelength
        self.__calibrant = calibrant
        self.__detector = detector
        self.__wavelength = wavelength
        self.__defaultConstraints = None

        self.__isValid = True
        try:
            self.__init(peaks, method)
        except Exception:
            _logger.error("Error while initializing the calibration", exc_info=True)
            self.__isValid = False

    def isValid(self):
        """
        Returns true if it can be use to calibrate the data.
        """
        return self.__isValid

    def __createDefaultParams(self, geometry=None):
        """
        Tries to initialise the GeometryRefinement (dist, poni, rot)
        Returns a dictionary of key value pairs
        """
        defaults = {"dist": None, "poni1": None, "poni2": None,
                    "rot1": None, "rot2": None, "rot3": None}
        if geometry is not None:
            defaults["dist"] = geometry.distance().value()
            defaults["poni1"] = geometry.poni1().value()
            defaults["poni2"] = geometry.poni2().value()
            defaults["rot1"] = geometry.rotation1().value()
            defaults["rot2"] = geometry.rotation2().value()
            defaults["rot3"] = geometry.rotation3().value()

        if self.__detector:
            try:
                p1, p2, _p3 = self.__detector.calc_cartesian_positions()
                if defaults["poni1"] is None:
                    defaults["poni1"] = p1.max() / 2.0
                if defaults["poni2"] is None:
                    defaults["poni2"] = p2.max() / 2.0
            except Exception as err:
                _logger.warning(err)

        if defaults["dist"] is None:
            defaults["dist"] = 0.1
        if defaults["poni1"] is None:
            defaults["poni1"] = 0.0
        if defaults["poni2"] is None:
            defaults["poni2"] = 0.0
        if defaults["rot1"] is None:
            defaults["rot1"] = 0.0
        if defaults["rot2"] is None:
            defaults["rot2"] = 0.0
        if defaults["rot3"] is None:
            defaults["rot3"] = 0.0

        return defaults

    def __init(self, peaks, method, geometry=None, constraintsModel=None):

        if len(peaks) == 0:
            self.__peakPicker = None
            self.__geoRef = None
            return

        scores = []
        defaultParams = self.__createDefaultParams(geometry=geometry)

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
            if not constraintsModel.isValid():
                raise RuntimeError("Constrain model is invalid")
            self.fromGeometryConstraintsModel(constraintsModel)
        score = geoRef.refine(1000000)
        scores.append((score, geoRef, "without-guess"))

        # Second attempt

        geoRef = GeometryRefinementContext(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaultParams)
        self.__geoRef = geoRef
        if constraintsModel is not None:
            if not constraintsModel.isValid():
                raise RuntimeError("Constrain model is invalid")
            self.fromGeometryConstraintsModel(constraintsModel)
        geoRef.guess_poni()
        score = geoRef.refine(1000000)
        scores.append((score, geoRef, "with-guess"))

        # Use the best one
        scores.sort(key=lambda x: x[0])
        score, geoRef, _ = scores[0]
        scores = None

        peakPicker = PeakPicker(data=self.__image,
                                calibrant=self.__calibrant,
                                wavelength=self.__wavelength,
                                detector=self.__detector,
                                method=method)

        if score == inf:
            self.__isValid = False
        self.__peakPicker = peakPicker
        self.__geoRef = geoRef

    def init(self, peaks, method, geometry, constraintsModel):
        self.__init(peaks, method, geometry, constraintsModel=constraintsModel)

    def update(self, image, mask, calibrant, detector, wavelength=None):
        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        self.__detector = detector
        if wavelength is not None:
            self.__wavelength = wavelength

    def getPyfaiGeometry(self):
        return self.__geoRef

    def refine(self, max_iter=500, seconds=10):
        """
        Contains the common geometry refinement part
        """
        self.__calibrant.setWavelength_change2th(self.__wavelength)
        self.__peakPicker.points.setWavelength_change2th(self.__wavelength)

        residual = previous_residual = float("+inf")

        count = 0
        timer = timeutils.Timer(seconds=seconds)

        while count < max_iter and not timer.isTimeout():
            residual = self.__geoRef.refine(10000)
            if residual >= previous_residual:
                break
            previous_residual = residual
            count += 1

        if residual == inf:
            self.__isValid = False

        print("Final residual: %s (after %s iterations)" % (residual, count))

        self.__geoRef.reset()

    def getRms(self):
        """Returns the RMS (root mean square) computed from the current fitting.

        The unit is the radian.
        """
        if self.__geoRef is None:
            return None
        try:
            chi2 = self.__geoRef.chi2()
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
            return inf
        return numpy.sqrt(chi2 / self.__geoRef.data.shape[0])

    def getTwoThetaArray(self):
        """
        Returns the 2th array corresponding to the calibrated image
        """
        # 2th array is cached insided
        tth = self.__geoRef.center_array(self.__peakPicker.shape, unit=units.TTH_RAD, scale=False)
        return tth

    def getMask(self):
        """
        Returns the mask used to compute the tth.
        """
        return self.__mask

    def getRings(self):
        """
        Returns polygons of rings

        :returns: List of ring angles available
        :rtype: List[float]
        """
        tth = self.__geoRef.center_array(self.__peakPicker.shape, unit=units.TTH_RAD, scale=False)

        result = collections.OrderedDict()

        tth_max = tth.max()
        tth_min = tth.min()
        if not self.__calibrant:
            return result

        angles = [i for i in self.__calibrant.get_2th()
                  if (i is not None) and (i >= tth_min) and (i <= tth_max)]
        if len(angles) == 0:
            return result

        return angles

    def getIndexedRings(self):
        """
        Returns polygons of rings

        :returns: List of tuples with ring (index, angle)  available
        :rtype: dict[ringId] = angle
        """
        tth = self.__geoRef.center_array(self.__peakPicker.shape, unit=units.TTH_RAD, scale=False)

        result = collections.OrderedDict()

        tth_max = tth.max()
        tth_min = tth.min()
        if not self.__calibrant:
            return result

        angles = collections.OrderedDict([(j, i) for (j,i) in enumerate(self.__calibrant.get_2th())
                  if (i is not None) and (i >= tth_min) and (i <= tth_max)])
        if len(angles) == 0:
            return result

        return angles


    def getBeamCenter(self):
        epsilon = 0.001
        try:
            f2d = self.__geoRef.getFit2D()
            x, y = f2d["centerX"], f2d["centerY"]
        except TypeError:
            return None

        tth = self.__geoRef.tth(numpy.array([y]), numpy.array([x]))[0]
        if tth >= epsilon:
            # Check if this pixel really contains the beam center
            # If the detector contains gap, it is not always the case
            tth_array = self.__geoRef.center_array(unit=units.TTH_RAD, scale=False)
            pos = tth_array.argmin()
            width = self.__geoRef.detector.shape[-1]
            y = pos // width
            x = pos %  width

            #This is for sub-pixel refinement...
            from ...ext.bilinear import Bilinear
            bili = Bilinear(-tth_array)
            y, x = bili.local_maxi((y,x))

            tth = self.__geoRef.tth(numpy.array([y]), numpy.array([x]))[0]
            if tth >=epsilon:
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
        self.__geoRef.reset()

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
        if self.__defaultConstraints is None:
            raise RuntimeError("No default geometry constrains model")
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
        fixed = FixedParameters()
        bounds = {}
        for name, constraint in attrs:
            if constraint.isFixed():
                fixed.add(name)
            elif constraint.isRangeConstrained():
                minValue, maxValue = constraint.range()
                if minValue is None:
                    minValue = -inf
                if maxValue is None:
                    maxValue = +inf
                bounds[name] = minValue, maxValue
        self.__geoRef.setFixed(fixed)
        self.__geoRef.setBounds(bounds)
