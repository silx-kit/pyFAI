# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""Helper to transform or extract information from the abstratc model.
"""

from __future__ import division


__authors__ = ["V. Valls"]
__license__ = "MIT"


from pyFAI.control_points import ControlPoints
from pyFAI.gui.model.CalibrationModel import CalibrationModel
from pyFAI.gui.model.PeakSelectionModel import PeakSelectionModel
from pyFAI.gui.model.PeakModel import PeakModel
from pyFAI.gui.CalibrationContext import CalibrationContext


def createControlPoints(model):
    """Create ControlPoints object from the calibration model

    :rtype: pyFAI.control_points.ControlPoints
    """
    if not isinstance(model, CalibrationModel):
        raise TypeError("Unexpected model type")

    calibrant = model.experimentSettingsModel().calibrantModel().calibrant()
    wavelength = model.experimentSettingsModel().wavelength().value()
    controlPoints = ControlPoints(calibrant=calibrant, wavelength=wavelength)
    for peakModel in model.peakSelectionModel():
        ringNumber = peakModel.ringNumber() - 1
        points = peakModel.coords()
        controlPoints.append(points=points, ring=ringNumber)
    return controlPoints


def filterControlPoints(filterCallback, peakSelectionModel, removedPeaks=None):
    """Filter each peaks of the model using a callback

    :param Callable[int,int,bool] filter: Filter returning true is the
        peak have to stay in the result.
    :param PeakSelectionModel peakSelectionModel: Model to filter
    :param List[Tuple[int,int]] removedPeaks: Provide a list to feed it with
        removed peaks from the model.
    """
    peakSelectionModel.lockSignals()
    for peakGroup in peakSelectionModel:
        changed = False
        newCoords = []
        for coord in peakGroup.coords():
            if filterCallback(coord[0], coord[1]):
                newCoords.append(coord)
            else:
                if removedPeaks is not None:
                    removedPeaks.append(coord)
                changed = True
        if changed:
            peakGroup.setCoords(newCoords)
    peakSelectionModel.unlockSignals()


def _findUnusedId(peakSelectionModel):
    """
    :rtype: int
    """
    # reach the bigger name
    names = ["% 8s" % p.name() for p in peakSelectionModel]
    if len(names) > 0:
        names = list(sorted(names))
        bigger = names[-1].strip()
        number = 0
        for c in bigger:
            number = number * 26 + (ord(c) - ord('a'))
    else:
        number = -1
    return number + 1


def _convertIdToName(number):
    """
    :rtype: str
    """
    # compute the next one
    name = ""
    if number == 0:
        name = "a"
    else:
        n = number
        while n > 0:
            c = n % 26
            n = n // 26
            name = chr(c + ord('a')) + name
    return name


def createRing(points, peakSelectionModel, context=None):
    """Create a new ring from a group of points

    :rtype: PeakModel
    """

    if context is None:
        context = CalibrationContext.instance()

    number = _findUnusedId(peakSelectionModel)
    name = _convertIdToName(number)
    color = context.getMarkerColor(number)

    # TODO: color and name should be removed from the model
    # TODO: As result this function should be removed
    peakModel = PeakModel(peakSelectionModel)
    peakModel.setName(name)
    peakModel.setColor(color)
    peakModel.setCoords(points)
    peakModel.setRingNumber(1)

    return peakModel


def initPeaksFromControlPoints(peakSelectionModel, controlPoints, context=None):
    """Initialize peak selection model using control points object

    :rtype: pyFAI.control_points.ControlPoints
    """
    if not isinstance(peakSelectionModel, PeakSelectionModel):
        raise TypeError("Unexpected model type")
    if not isinstance(controlPoints, ControlPoints):
        raise TypeError("Unexpected model type")

    if context is None:
        context = CalibrationContext.instance()

    peakSelectionModel.clear()
    for label in controlPoints.get_labels():
        group = controlPoints.get(lbl=label)
        peakModel = createRing(group.points, peakSelectionModel=peakSelectionModel, context=context)
        peakModel.setRingNumber(group.ring + 1)
        peakModel.setName(label)
        peakSelectionModel.append(peakModel)


def geometryModelToGeometry(geometryModel, geometry):
    geometry.dist = geometryModel.distance().value()
    geometry.poni1 = geometryModel.poni1().value()
    geometry.poni2 = geometryModel.poni2().value()
    geometry.rot1 = geometryModel.rotation1().value()
    geometry.rot2 = geometryModel.rotation2().value()
    geometry.rot3 = geometryModel.rotation3().value()
    geometry.wavelength = geometryModel.wavelength().value()
