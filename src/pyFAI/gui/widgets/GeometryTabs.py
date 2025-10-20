# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2023 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls", "J. Kieffer"]
__license__ = "MIT"
__date__ = "03/02/2023"

from silx.gui import qt
from ...utils import get_ui_file
from ...geometry import fit2d
from ..utils import units
from ..model.DataModel import DataModel
from ..model.GeometryModel import GeometryModel
from ..model.Fit2dGeometryModel import Fit2dGeometryModel


class GeometryTabs(qt.QWidget):
    """Widget to display a selected geometry with various representation:
    * pyFAI
    * Fit2D
    * More to come.

    This widget is meant to be used in GeometryDialog and in IntegrationTask in calib2
    """

    def __init__(self, parent=None):
        super(GeometryTabs, self).__init__(parent)
        filename = get_ui_file("geometry-tabs.ui")
        qt.loadUi(filename, self)

        self._geometry = GeometryModel()
        self.__fit2dGeometry = Fit2dGeometryModel()
        self.__detector = None
        self.__originalGeometry = None
        self.__updatingModel = False

        # Create shared units
        angleUnit = DataModel()
        angleUnit.setValue(units.Unit.RADIAN)
        lengthUnit = DataModel()
        lengthUnit.setValue(units.Unit.METER)
        pixelUnit = DataModel()
        pixelUnit.setValue(units.Unit.PIXEL)

        # Connect pyFAI widgets to units
        self._pyfaiDistance.setDisplayedUnitModel(lengthUnit)
        self._pyfaiDistance.setModelUnit(units.Unit.METER)
        self._pyfaiDistanceUnit.setUnitModel(lengthUnit)
        self._pyfaiDistanceUnit.setUnitEditable(True)
        self._pyfaiPoni1.setDisplayedUnitModel(lengthUnit)
        self._pyfaiPoni1.setModelUnit(units.Unit.METER)
        self._pyfaiPoni1Unit.setUnitModel(lengthUnit)
        self._pyfaiPoni1Unit.setUnitEditable(True)
        self._pyfaiPoni2.setDisplayedUnitModel(lengthUnit)
        self._pyfaiPoni2.setModelUnit(units.Unit.METER)
        self._pyfaiPoni2Unit.setUnitModel(lengthUnit)
        self._pyfaiPoni2Unit.setUnitEditable(True)
        self._pyfaiRotation1.setDisplayedUnitModel(angleUnit)
        self._pyfaiRotation1.setModelUnit(units.Unit.RADIAN)
        self._pyfaiRotation1Unit.setUnitModel(angleUnit)
        self._pyfaiRotation1Unit.setUnitEditable(True)
        self._pyfaiRotation2.setDisplayedUnitModel(angleUnit)
        self._pyfaiRotation2.setModelUnit(units.Unit.RADIAN)
        self._pyfaiRotation2Unit.setUnitModel(angleUnit)
        self._pyfaiRotation2Unit.setUnitEditable(True)
        self._pyfaiRotation3.setDisplayedUnitModel(angleUnit)
        self._pyfaiRotation3.setModelUnit(units.Unit.RADIAN)
        self._pyfaiRotation3Unit.setUnitModel(angleUnit)
        self._pyfaiRotation3Unit.setUnitEditable(True)

        # Connect fit2d widgets to units
        self._fit2dDistance.setDisplayedUnit(units.Unit.MILLIMETER)
        self._fit2dDistance.setModelUnit(units.Unit.MILLIMETER)
        self._fit2dDistanceUnit.setUnit(units.Unit.MILLIMETER)
        self._fit2dCenterX.setDisplayedUnitModel(pixelUnit)
        self._fit2dCenterX.setModelUnit(units.Unit.PIXEL)
        self._fit2dCenterXUnit.setUnit(units.Unit.PIXEL)
        self._fit2dCenterY.setDisplayedUnitModel(pixelUnit)
        self._fit2dCenterY.setModelUnit(units.Unit.PIXEL)
        self._fit2dCenterYUnit.setUnit(units.Unit.PIXEL)
        self._fit2dTilt.setDisplayedUnit(units.Unit.DEGREE)
        self._fit2dTilt.setModelUnit(units.Unit.DEGREE)
        self._fit2dTiltUnit.setUnit(units.Unit.DEGREE)
        self._fit2dTiltPlan.setDisplayedUnit(units.Unit.DEGREE)
        self._fit2dTiltPlan.setModelUnit(units.Unit.DEGREE)
        self._fit2dTiltPlanUnit.setUnit(units.Unit.DEGREE)

        # Connect fit2d model-widget
        self._fit2dDistance.setModel(self.__fit2dGeometry.distance())
        self._fit2dCenterX.setModel(self.__fit2dGeometry.centerX())
        self._fit2dCenterY.setModel(self.__fit2dGeometry.centerY())
        self._fit2dTilt.setModel(self.__fit2dGeometry.tilt())
        self._fit2dTiltPlan.setModel(self.__fit2dGeometry.tiltPlan())

        self._pyfaiDistance.setModel(self._geometry.distance())
        self._pyfaiPoni1.setModel(self._geometry.poni1())
        self._pyfaiPoni2.setModel(self._geometry.poni2())
        self._pyfaiRotation1.setModel(self._geometry.rotation1())
        self._pyfaiRotation2.setModel(self._geometry.rotation2())
        self._pyfaiRotation3.setModel(self._geometry.rotation3())

        self._geometry.changed.connect(self.__updateFit2dFromPyfai)
        self.__fit2dGeometry.changed.connect(self.__updatePyfaiFromFit2d)

    def isReadOnly(self):
        """
        Returns True if the dialog is in read only.

        In read-only mode, the geometry is displayed, but the user can't edited
        it.

        By default, this returns false.

        :rtype: bool
        """
        return self._pyfaiDistance.isReadOnly()

    def setReadOnly(self, readOnly):
        """
        Enable or disable the read-only mode.

        :param bool readOnly: True to enable the read-only mode.
        """
        self._pyfaiDistance.setReadOnly(readOnly)
        self._pyfaiPoni1.setReadOnly(readOnly)
        self._pyfaiPoni2.setReadOnly(readOnly)
        self._pyfaiRotation1.setReadOnly(readOnly)
        self._pyfaiRotation2.setReadOnly(readOnly)
        self._pyfaiRotation3.setReadOnly(readOnly)

        self._fit2dDistance.setReadOnly(readOnly)
        self._fit2dCenterX.setReadOnly(readOnly)
        self._fit2dCenterY.setReadOnly(readOnly)
        self._fit2dTilt.setReadOnly(readOnly)
        self._fit2dTiltPlan.setReadOnly(readOnly)

    def __createPyfaiGeometry(self):
        geometry = self._geometry
        if not geometry.isValid(checkWaveLength=False):
            raise RuntimeError("The geometry is not valid")
        dist = geometry.distance().value()
        poni1 = geometry.poni1().value()
        poni2 = geometry.poni2().value()
        rot1 = geometry.rotation1().value()
        rot2 = geometry.rotation2().value()
        rot3 = geometry.rotation3().value()
        wavelength = geometry.wavelength().value()
        result = fit2d.PoniFile({"dist":dist,
                                 "poni1":poni1,
                                 "poni2":poni2,
                                 "rot1":rot1,
                                 "rot2":rot2,
                                 "rot3":rot3,
                                 "detector":self.__detector,
                                 "wavelength":wavelength})
        return result

    def __updatePyfaiFromFit2d(self):
        if self.__updatingModel:
            return
        self.__updatingModel = True
        geometry = self.__fit2dGeometry
        error = None
        distance = None
        poni1 = None
        poni2 = None
        rotation1 = None
        rotation2 = None
        rotation3 = None

        if geometry is None:
            error = "No geometry to compute pyFAI geometry."
            pass
        elif self.__detector is None:
            error = "No detector defined. It is needed to compute the pyFAI geometry."
        elif not geometry.isValid():
            error = "The current geometry is not valid to compute the pyFAI one."
        else:
            fit2dGeometry = fit2d.Fit2dGeometry(geometry.distance().value(),
                                                geometry.centerX().value(),
                                                geometry.centerY().value(),
                                                geometry.tilt().value(),
                                                geometry.tiltPlan().value(),
                                                detector=self.__detector)
            try:
                pyFAIGeometry = fit2d.convert_from_Fit2d(fit2dGeometry)
            except Exception:
                error = "This geometry can't be modelized with pyFAI."
            else:
                distance = pyFAIGeometry.dist
                poni1 = pyFAIGeometry.poni1
                poni2 = pyFAIGeometry.poni2
                rotation1 = pyFAIGeometry.rot1
                rotation2 = pyFAIGeometry.rot2
                rotation3 = pyFAIGeometry.rot3

        self._fit2dError.setVisible(error is not None)
        self._fit2dError.setText(error)
        self._geometry.lockSignals()
        self._geometry.distance().setValue(distance)
        self._geometry.poni1().setValue(poni1)
        self._geometry.poni2().setValue(poni2)
        self._geometry.rotation1().setValue(rotation1)
        self._geometry.rotation2().setValue(rotation2)
        self._geometry.rotation3().setValue(rotation3)
        self._geometry.unlockSignals()
        self.__updatingModel = False

    def __updateFit2dFromPyfai(self):
        if self.__updatingModel:
            return
        self.__updatingModel = True
        geometry = self._geometry
        error = None
        distance = None
        centerX = None
        centerY = None
        tiltPlan = None
        tilt = None

        if geometry is None:
            error = "No geometry to compute Fit2D geometry."
            pass
        elif self.__detector is None:
            error = "No detector defined. It is needed to compute the Fit2D geometry."
        elif not geometry.isValid(checkWaveLength=False):
            error = "The current geometry is not valid to compute the Fit2D one."
        else:
            pyFAIGeometry = self.__createPyfaiGeometry()
            try:
                result = fit2d.convert_to_Fit2d(pyFAIGeometry)
            except Exception:
                error = "This geometry can't be modelized with Fit2D."
            else:
                distance = result.directDist
                centerX = result.centerX
                centerY = result.centerY
                tilt = result.tilt
                tiltPlan = result.tiltPlanRotation

        self._fit2dError.setVisible(error is not None)
        self._fit2dError.setText(error)
        self.__fit2dGeometry.lockSignals()
        self.__fit2dGeometry.distance().setValue(distance)
        self.__fit2dGeometry.centerX().setValue(centerX)
        self.__fit2dGeometry.centerY().setValue(centerY)
        self.__fit2dGeometry.tilt().setValue(tilt)
        self.__fit2dGeometry.tiltPlan().setValue(tiltPlan)
        self.__fit2dGeometry.unlockSignals()
        self.__updatingModel = False

    def resetToOriginalGeometry(self):
        if self.__originalGeometry is None:
            return
        self._geometry.setFrom(self.__originalGeometry)

    def isDirty(self):
        """Tell if the geometry has changed"""
        return self._geometry != self.__originalGeometry

    def detector(self):
        return self.__detector

    def setDetector(self, detector):
        """Set the used detector.

        This information is needed to display the Fit2D geometry.
        """
        self.__detector = detector
        self.__updateFit2dFromPyfai()

    def setGeometryModel(self, geometryModel):
        """Set the geometry to display.

        :param ~pyFAI.gui.model.GeometryModel geometryModel: A geometry.
        """
        if not isinstance(geometryModel, GeometryModel):
            raise RuntimeError("geometryModel is not a pyFAI.gui.model.GeometryModel")
        if self._geometry is geometryModel:
            return
        # first setting
        if geometryModel.isValid() and self.__originalGeometry is None:
            self.__originalGeometry = geometryModel.copy()
        self._geometry.setFrom(geometryModel)

    def geometryModel(self):
        """Returns the geometry model

        :rtype: ~pyFAI.gui.model.GeometryModel
        """
        return self._geometry
