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
__date__ = "22/11/2018"

import logging
import numpy
from silx.image import marchingsquares

import pyFAI.utils
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.geometry import Geometry
from ..peak_picker import PeakPicker

_logger = logging.getLogger(__name__)


class RingExtractor(object):

    def __init__(self, image, mask, calibrant, detector, wavelength):
        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        self.__calibrant.setWavelength_change2th(wavelength)
        # self.__calibrant.set_wavelength(wavelength)
        self.__detector = detector
        self.__wavelength = wavelength
        self.__geoRef = None

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

    def __createGeoRefFromPeaks(self, peaks):
        """
        Contains the geometry refinement part specific to Calibration
        Sets up the initial guess when starting pyFAI-calib
        """

        fixed = pyFAI.utils.FixedParameters()
        fixed.add("wavelength")

        scores = []
        PARAMETERS = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3", "wavelength"]

        # First attempt
        defaults = self.__initGeoRef()
        geoRef = GeometryRefinement(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaults)
        geoRef.refine2(1000000, fix=fixed)
        score = geoRef.chi2()
        parameters = [getattr(geoRef, p) for p in PARAMETERS]
        scores.append((score, parameters))

        # Second attempt
        defaults = self.__initGeoRef()
        geoRef = GeometryRefinement(
            data=peaks,
            wavelength=self.__wavelength,
            detector=self.__detector,
            calibrant=self.__calibrant,
            **defaults)
        geoRef.guess_poni()
        geoRef.refine2(1000000, fix=fixed)
        score = geoRef.chi2()
        parameters = [getattr(geoRef, p) for p in PARAMETERS]
        scores.append((score, parameters))

        # Third attempt
        # FIXME use the geometry from the computed model

        # Choose the best scoring method: At this point we might also ask
        # a user to just type the numbers in?
        scores.sort()
        _score, parameters = scores[0]
        for parval, parname in zip(parameters, PARAMETERS):
            setattr(geoRef, parname, parval)

        return geoRef

    def __createGeoRefFromGeometry(self, geometryModel):
        geoRef = Geometry(
            dist=geometryModel.distance().value(),
            wavelength=geometryModel.wavelength().value(),
            poni1=geometryModel.poni1().value(),
            poni2=geometryModel.poni2().value(),
            rot1=geometryModel.rotation1().value(),
            rot2=geometryModel.rotation2().value(),
            rot3=geometryModel.rotation3().value(),
            detector=self.__detector)
        return geoRef

    def extract(self, peaks=None, geometryModel=None, method="massif", maxRings=None, pointPerDegree=1.0):
        """
        Performs an automatic keypoint extraction:
        Can be used in recalib or in calib after a first calibration has been performed.

        # FIXME pts_per_deg
        """
        assert(numpy.logical_xor(peaks is not None, geometryModel is not None))

        if peaks is not None:
            # Energy from from experiment settings
            wavelength = self.__wavelength
            self.__calibrant.setWavelength_change2th(wavelength)
            geoRef = self.__createGeoRefFromPeaks(peaks)
        elif geometryModel is not None:
            # Fitted energy
            assert(geometryModel.isValid())
            wavelength = geometryModel.wavelength().value()
            self.__calibrant.setWavelength_change2th(wavelength)
            geoRef = self.__createGeoRefFromGeometry(geometryModel)

        self.__geoRef = geoRef

        peakPicker = PeakPicker(data=self.__image,
                                mask=self.__mask,
                                calibrant=self.__calibrant,
                                wavelength=wavelength,
                                detector=self.__detector,
                                method=method)

        peakPicker.reset()
        peakPicker.init(method, False)

        tth = numpy.array([i for i in self.__calibrant.get_2th() if i is not None])
        tth = numpy.unique(tth)
        tth_min = numpy.zeros_like(tth)
        tth_max = numpy.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 4.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth

        ttha = geoRef.get_ttha()
        chia = geoRef.get_chia()
        if (ttha is None) or (ttha.shape != peakPicker.data.shape):
            ttha = geoRef.twoThetaArray(peakPicker.data.shape)
        if (chia is None) or (chia.shape != peakPicker.data.shape):
            chia = geoRef.chiArray(peakPicker.data.shape)

        rings = 0
        peakPicker.sync_init()
        if maxRings is None:
            maxRings = tth.size
        ms = marchingsquares.MarchingSquaresMergeImpl(ttha, self.__mask, use_minmax_cache=True)

        for i in range(tth.size):
            if rings >= maxRings:
                break
            mask = numpy.logical_and(ttha >= tth_min[i], ttha < tth_max[i])
            # if self.mask is not None:
            #     mask = numpy.logical_and(mask, numpy.logical_not(self.mask))
            size = mask.sum(dtype=int)
            if (size > 0):
                rings += 1
                peakPicker.massif_contour(mask)
                # if self.gui:
                #     update_fig(self.peakPicker.fig)
                sub_data = peakPicker.data.ravel()[numpy.where(mask.ravel())]
                mean = sub_data.mean(dtype=numpy.float64)
                std = sub_data.std(dtype=numpy.float64)
                upper_limit = mean + std
                mask2 = numpy.logical_and(peakPicker.data > upper_limit, mask)
                size2 = mask2.sum(dtype=int)
                if size2 < 1000:
                    upper_limit = mean
                    mask2 = numpy.logical_and(peakPicker.data > upper_limit, mask)
                    size2 = mask2.sum()
                # length of the arc:
                # Coords in points are y, x
                points = ms.find_pixels(tth[i])

                seeds = set((i[0], i[1]) for i in points if mask2[i[0], i[1]])
                # max number of points: 360 points for a full circle
                azimuthal = chia[points[:, 0].clip(0, peakPicker.data.shape[0]), points[:, 1].clip(0, peakPicker.data.shape[1])]
                nb_deg_azim = numpy.unique(numpy.rad2deg(azimuthal).round()).size
                keep = int(nb_deg_azim * pointPerDegree)
                if keep == 0:
                    continue
                dist_min = len(seeds) / 2.0 / keep
                # why 3.0, why not ?

                msg = "Extracting datapoint for ring %s (2theta = %.2f deg); "\
                    "searching for %i pts out of %i with I>%.1f, dmin=%.1f"
                _logger.info(msg, i, numpy.degrees(tth[i]), keep, size2, upper_limit, dist_min)
                _res = peakPicker.peaks_from_area(mask=mask2, Imin=upper_limit, keep=keep, method=method, ring=i, dmin=dist_min, seed=seeds)

        # self.peakPicker.points.save(self.basename + ".npt")
        # if self.weighted:
        #     self.data = self.peakPicker.points.getWeightedList(self.peakPicker.data)
        # else:
        #     self.data = peakPicker.points.getList()
        return peakPicker.points.getList()

    def toGeometryModel(self, model):
        model.lockSignals()
        model.wavelength().setValue(self.__geoRef.wavelength)
        model.distance().setValue(self.__geoRef.dist)
        model.poni1().setValue(self.__geoRef.poni1)
        model.poni2().setValue(self.__geoRef.poni2)
        model.rotation1().setValue(self.__geoRef.rot1)
        model.rotation2().setValue(self.__geoRef.rot2)
        model.rotation3().setValue(self.__geoRef.rot3)
        model.unlockSignals()
