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
__date__ = "18/10/2018"

import logging
import numpy
import collections

from silx.image import marchingsquares
import pyFAI.utils
from ...geometryRefinement import GeometryRefinement
from ..peak_picker import PeakPicker
from ..utils import timeutils

_logger = logging.getLogger(__name__)


class RingCalibration(object):

    def __init__(self, image, mask, calibrant, detector, wavelength, peaks, method):
        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        self.__calibrant.set_wavelength(wavelength)
        self.__detector = detector
        self.__wavelength = wavelength
        self.__init(peaks, method)

        fixed = pyFAI.utils.FixedParameters()
        fixed.add("wavelength")
        self.__fixed = fixed
        self.__rms = None
        self.__previousRms = None
        self.__peakResidual = None

    def __initgeoRef(self):
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

    def __init(self, peaks, method):
        defaults = self.__initgeoRef()
        fixed = pyFAI.utils.FixedParameters()
        fixed.add("wavelength")

        if len(peaks) == 0:
            self.__peakPicker = None
            self.__geoRef = None
            return

        geoRef = GeometryRefinement(data=peaks,
                                    wavelength=self.__wavelength,
                                    detector=self.__detector,
                                    calibrant=self.__calibrant,
                                    **defaults)
        self.__rms = geoRef.refine2(1000000, fix=fixed)
        self.__peakResidual = self.__rms
        self.__previousRms = None

        peakPicker = PeakPicker(data=self.__image,
                                calibrant=self.__calibrant,
                                wavelength=self.__wavelength,
                                detector=self.__detector,
                                method=method)

        self.__peakPicker = peakPicker
        self.__geoRef = geoRef

    def init(self, peaks, method):
        self.__init(peaks, method)

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
        if "wavelength" in self.__fixed:
            chi2 = self.__geoRef.chi2()
        else:
            chi2 = self.__geoRef.chi2_wavelength()
        return numpy.sqrt(chi2 / self.__geoRef.data.shape[0])

    def __refine(self, maxiter=1000000, fix=None):
        if "wavelength" in self.__fixed:
            return self.__geoRef.refine2(maxiter, fix)
        else:
            return self.__geoRef.refine2_wavelength(maxiter, fix)

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
        timeout = timeutils.Timeout(seconds=10)

        while count < max_iter and not timeout:
            residual = self.__refine(10000, fix=self.__fixed)
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

    def getPeakResidual(self):
        """Returns the residual computed from the peak selection."""
        return self.__peakResidual

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

    def fromGeometryConstriansModel(self, contraintsModel):
        # FIXME take care of range values
        fixed = pyFAI.utils.FixedParameters()
        if contraintsModel.wavelength().isFixed():
            fixed.add("wavelength")
        if contraintsModel.distance().isFixed():
            fixed.add("dist")
        if contraintsModel.poni1().isFixed():
            fixed.add("poni1")
        if contraintsModel.poni2().isFixed():
            fixed.add("poni2")
        if contraintsModel.rotation1().isFixed():
            fixed.add("rot1")
        if contraintsModel.rotation2().isFixed():
            fixed.add("rot2")
        if contraintsModel.rotation3().isFixed():
            fixed.add("rot3")
        self.__fixed = fixed
