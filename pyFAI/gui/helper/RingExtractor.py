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
__date__ = "22/03/2019"

import logging
import numpy

from silx.gui import qt
from silx.image import marchingsquares

import pyFAI.utils
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.geometry import Geometry
from ..peak_picker import PeakPicker
from . import model_transform

_logger = logging.getLogger(__name__)


class RingExtractorThread(qt.QThread):
    """Job to process data and collect peaks according to a diffraction ring
    modelization.
    """

    sigProcessLocationChanged = qt.Signal(object)

    def __init__(self, parent):
        """Constructor"""
        super(RingExtractorThread, self).__init__(parent=parent)

        self.__image = None
        self.__mask = None
        self.__calibrant = None
        self.__detector = None
        self.__wavelength = None
        self.__geoRef = None

        self.__maxRings = None
        self.__ringNumbers = None
        self.__pointPerDegree = None
        self.__peaksModel = None
        self.__geometryModel = None

        self.__error = None
        self.__keys = {}

    def errorString(self):
        """Returns the error message in case of failure"""
        return self.__error

    def isAborted(self):
        """
        Returns whether the theard was aborted or not.

        .. note:: Aborted thead are not finished theads.
        """
        return self.__isAborted

    def run(self):
        self.__isAborted = False
        try:
            result = self.runProcess()
        except Exception as e:
            _logger.error("Backtrace", exc_info=True)
            self.__error = str(e)
            self.__isAborted = True
        else:
            if not result:
                self.__error = "Task was aborted"
                self.__isAborted = True

    def setUserData(self, name, value):
        """Store key-value information from caller to be retrived when the
        processing finish."""
        self.__keys[name] = value

    def userData(self, name):
        """Returns a stored user data."""
        return self.__keys[name]

    def setPeaksModel(self, peaksModel):
        """Define a set of peaks as source of the diffraction ring
        modelization"""
        self.__peaksModel = peaksModel

    def setGeometryModel(self, geometryModel):
        """Define a geometry model as source of the diffraction ring
        modelization"""
        self.__geometryModel = geometryModel

    def setMaxRings(self, maxRings):
        """Set max ring to extract"""
        self.__maxRings = maxRings

    def setRingNumbers(self, ringNumbers):
        """Specify a set of rings to extract

        :param List[int] ringNumbers: List of number (1 is the 1st ring)
        """
        if ringNumbers is None:
            self.__ringNumbers = None
        else:
            self.__ringNumbers = [n - 1 for n in ringNumbers]

    def setPointPerDegree(self, pointPerDegree):
        """Specify the amount of peak to extract per degree"""
        self.__pointPerDegree = pointPerDegree

    def setExperimentSettings(self, experimentSettings, copy):
        """
        Set the experiment data.

        :param ..model.ExperimentSettingsModel.ExperimentSettingsModel experimentSettings:
            Contains the modelization of the problem
        :param bool copy: If true copy the data for a thread safe processing
        """
        image = experimentSettings.image().value()
        mask = experimentSettings.mask().value()
        calibrant = experimentSettings.calibrantModel().calibrant()
        detector = experimentSettings.detector()
        wavelength = experimentSettings.wavelength().value()

        if copy:
            if image is not None:
                image = image.copy()
            if mask is not None:
                mask = mask.copy()

        self.__image = image
        self.__mask = mask
        self.__calibrant = calibrant
        if self.__calibrant is not None:
            self.__calibrant.setWavelength_change2th(wavelength)
        self.__detector = detector
        self.__wavelength = wavelength

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

    def runProcess(self):
        """Extract the peaks.

        :raises ValueError: If a mandatory setting is not initialized.
        :rtype: bool
        :returns: True if successed
        """
        if self.__detector is None:
            raise ValueError("No detector defined")
        if self.__calibrant is None:
            raise ValueError("No calibrant defined")
        if self.__wavelength is None:
            raise ValueError("No wavelength defined")

        peaksModel = self.__peaksModel
        geometryModel = self.__geometryModel

        if peaksModel is not None and geometryModel is not None:
            raise ValueError("Computation have to be done from peaks or from geometry")

        if peaksModel is not None:
            peaks = model_transform.createPeaksArray(peaksModel)
            geometryModel = None

        elif geometryModel is not None:
            peaks = None
            if not geometryModel.isValid():
                raise ValueError("The fitted model is not valid. Extraction cancelled.")

        result = self._extract(peaks=peaks, geometryModel=geometryModel)
        self.__newPeaksRaw = result
        return True

    def resultPeaks(self):
        """Returns the extracted peaks.

        :rtype: dict
        """
        if len(self.__newPeaksRaw) == 0:
            return {}

        # Index to result per ring number
        raw = numpy.array(self.__newPeaksRaw)
        newPeaks = {}
        ringNumbers = numpy.unique(raw[:, 2])
        countPeaks = 0
        for ringNumber in ringNumbers:
            coords = raw[raw[:, 2] == ringNumber][:, 0:2]
            ringNumber = int(ringNumber) + 1
            newPeaks[ringNumber] = coords
            countPeaks += len(coords)
        assert(countPeaks == len(raw))
        return newPeaks

    def _updateProcessingLocation(self, mask):
        self.sigProcessLocationChanged.emit(mask)

    def _extract(self, peaks=None, geometryModel=None):
        """
        Performs an automatic keypoint extraction:
        Can be used in recalib or in calib after a first calibration has been performed.

        :param List[int] ringNumbers: If set, extraction will only be done on
            rings number contained in this list (the number 0 identify the first
            ring)
        """
        assert(numpy.logical_xor(peaks is not None, geometryModel is not None))
        method = "massif"
        maxRings = self.__maxRings
        ringNumbers = self.__ringNumbers
        pointPerDegree = self.__pointPerDegree

        if ringNumbers is not None:
            ringNumbers = set(ringNumbers)

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
            if ringNumbers is not None:
                if i not in ringNumbers:
                    continue
            if rings >= maxRings:
                break
            mask = numpy.logical_and(ttha >= tth_min[i], ttha < tth_max[i])
            size = mask.sum(dtype=int)
            if (size > 0):
                rings += 1
                self._updateProcessingLocation(mask)
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
                # Coords in points are y, x
                points = ms.find_pixels(tth[i])

                seeds = set((i[0], i[1]) for i in points if mask2[i[0], i[1]])
                # Max number of points: 360 points for a full circle
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
