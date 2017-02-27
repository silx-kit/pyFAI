# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "27/02/2017"

import logging
import numpy
import pyFAI.utils
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.peak_picker import PeakPicker
try:
    from .third_party import six
except (ImportError, Exception):
    import six

_logger = logging.getLogger(__name__)


class RingCalibration(object):

    def __init__(self, image, calibrant, detector, wavelength, peaks, method):
        self.__image = image
        self.__calibrant = calibrant
        self.__calibrant.set_wavelength(wavelength)
        self.__detector = detector
        self.__wavelength = wavelength
        self.__init(peaks, method)

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

        geoRef = GeometryRefinement(data=peaks,
                                    wavelength=self.__wavelength,
                                    detector=self.__detector,
                                    calibrant=self.__calibrant,
                                    **defaults)
        geoRef.refine2(1000000, fix=fixed)

        peakPicker = PeakPicker(data=self.__image,
                                calibrant=self.__calibrant,
                                wavelength=self.__wavelength,
                                detector=self.__detector,
                                method=method)

        self.__peakPicker = peakPicker
        self.__geoRef = geoRef
        self.__fixed = fixed

    def init(self, peaks, method):
        self.__init(peaks, method)

    def refine(self, max_iter=1000):
        """
        Contains the common geometry refinement part
        """
        print("Before refinement, the geometry is:")
        print(self.__geoRef)

        previous = six.MAXSIZE
        finished = False

        while not finished:
            count = 0
            if "wavelength" in self.__fixed:
                while (previous > self.__geoRef.chi2()) and (count < max_iter):
                    if (count == 0):
                        previous = six.MAXSIZE
                    else:
                        previous = self.__geoRef.chi2()
                    self.__geoRef.refine2(1000000, fix=self.__fixed)
                    print(self.__geoRef)
                    count += 1
            else:
                while previous > self.__geoRef.chi2_wavelength() and (count < max_iter):
                    if (count == 0):
                        previous = six.MAXSIZE
                    else:
                        previous = self.__geoRef.chi2()
                    self.__geoRef.refine2_wavelength(1000000, fix=self.__fixed)
                    print(self.__geoRef)
                    count += 1
                self.__peakPicker.points.setWavelength_change2th(self.__geoRef.wavelength)

            self.__geoRef.del_ttha()
            self.__geoRef.del_dssa()
            self.__geoRef.del_chia()

            finished = True
            if not finished:
                previous = six.MAXSIZE

    def getRings(self):
        """
        Overlay a contour-plot
        """
        tth = self.__geoRef.twoThetaArray(self.__peakPicker.shape)

        tth_max = tth.max()
        tth_min = tth.min()
        if self.__calibrant:
            angles = [i for i in self.__calibrant.get_2th()
                      if (i is not None) and (i >= tth_min) and (i <= tth_max)]
            if not angles:
                angles = None
        else:
            angles = None

        # FIXME use documentaed function
        import matplotlib._cntr
        x, y = numpy.mgrid[:tth.shape[0], :tth.shape[1]]
        contour = matplotlib._cntr.Cntr(x, y, tth)

        rings = []
        for angle in angles:
            res = contour.trace(angle)
            nseg = len(res) // 2
            segments, _codes = res[:nseg], res[nseg:]
            rings.append(segments)

        return rings

    def getBeamCenter(self):
        try:
            f2d = self.__geoRef.getFit2D()
            return f2d["centerY"], f2d["centerX"]
        except TypeError:
            return None

    def toGeometryModel(self, model):
        model.wavelength().setValue(self.__geoRef.wavelength * 1e10)
        model.distance().setValue(self.__geoRef.dist)
        model.poni1().setValue(self.__geoRef.poni1)
        model.poni2().setValue(self.__geoRef.poni2)
        model.rotation1().setValue(self.__geoRef.rot1)
        model.rotation2().setValue(self.__geoRef.rot2)
        model.rotation3().setValue(self.__geoRef.rot3)
