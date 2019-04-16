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
__date__ = "22/02/2019"

from silx.gui import qt

from pyFAI.utils import get_ui_file
from ..utils import units
from ..model.DataModel import DataModel
from ..model.GeometryModel import GeometryModel
from ..model.Fit2dGeometryModel import Fit2dGeometryModel
from pyFAI.geometry import Geometry


class GeometryDialog(qt.QDialog):
    """Dialog to display a selected geometry
    """

    def __init__(self, parent=None):
        super(GeometryDialog, self).__init__(parent)
        filename = get_ui_file("geometry-dialog.ui")
        qt.loadUi(filename, self)

        self.__geometry = GeometryModel()
        self.__fit2dGeometry = Fit2dGeometryModel()
        self.__detector = None
        self.__originalGeometry = None
        self.__updatingModel = False

        # Connect buttons
        self._buttonBox.rejected.connect(self.reject)
        self._buttonBox.accepted.connect(self.accept)

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
        self._fit2dDistance.setDisplayedUnitModel(lengthUnit)
        self._fit2dDistance.setModelUnit(units.Unit.MILLIMETER)
        self._fit2dDistanceUnit.setUnit(units.Unit.MILLIMETER)
        # self._fit2dDistanceUnit.setUnitModel(lengthUnit)
        # self._fit2dDistanceUnit.setUnitEditable(True)
        self._fit2dCenterX.setDisplayedUnitModel(pixelUnit)
        self._fit2dCenterX.setModelUnit(units.Unit.PIXEL)
        self._fit2dCenterXUnit.setUnit(units.Unit.PIXEL)
        # self._fit2dCenterXUnit.setUnitModel(pixelUnit)
        # self._fit2dCenterXUnit.setUnitEditable(True)
        self._fit2dCenterY.setDisplayedUnitModel(pixelUnit)
        self._fit2dCenterY.setModelUnit(units.Unit.PIXEL)
        self._fit2dCenterYUnit.setUnit(units.Unit.PIXEL)
        # self._fit2dCenterYUnit.setUnitModel(pixelUnit)
        # self._fit2dCenterYUnit.setUnitEditable(True)
        self._fit2dTilt.setDisplayedUnitModel(angleUnit)
        self._fit2dTilt.setModelUnit(units.Unit.DEGREE)
        self._fit2dTiltUnit.setUnit(units.Unit.DEGREE)
        # self._fit2dTiltUnit.setUnitModel(angleUnit)
        # self._fit2dTiltUnit.setUnitEditable(True)
        self._fit2dTiltPlan.setDisplayedUnitModel(angleUnit)
        self._fit2dTiltPlan.setModelUnit(units.Unit.DEGREE)
        self._fit2dTiltPlanUnit.setUnit(units.Unit.DEGREE)
        # self._fit2dTiltPlanUnit.setUnitModel(angleUnit)
        # self._fit2dTiltPlanUnit.setUnitEditable(True)

        # Connect fit2d model-widget
        self._fit2dDistance.setModel(self.__fit2dGeometry.distance())
        self._fit2dCenterX.setModel(self.__fit2dGeometry.centerX())
        self._fit2dCenterY.setModel(self.__fit2dGeometry.centerY())
        self._fit2dTilt.setModel(self.__fit2dGeometry.tilt())
        self._fit2dTiltPlan.setModel(self.__fit2dGeometry.tiltPlan())

        self._pyfaiDistance.setModel(self.__geometry.distance())
        self._pyfaiPoni1.setModel(self.__geometry.poni1())
        self._pyfaiPoni2.setModel(self.__geometry.poni2())
        self._pyfaiRotation1.setModel(self.__geometry.rotation1())
        self._pyfaiRotation2.setModel(self.__geometry.rotation2())
        self._pyfaiRotation3.setModel(self.__geometry.rotation3())

        self.__geometry.changed.connect(self.__updateFit2dFromPyfai)
        self.__fit2dGeometry.changed.connect(self.__updatePyfaiFromFit2d)
        self.__geometry.changed.connect(self.__updateButtons)

        # NOTE: All the buttons have to be create here.
        # Changing available buttons on the focus event create a segfault
        types = (qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel |
                 qt.QDialogButtonBox.Reset | qt.QDialogButtonBox.Close)
        self._buttonBox.setStandardButtons(types)
        resetButton = self._buttonBox.button(qt.QDialogButtonBox.Reset)
        resetButton.clicked.connect(self.__resetToOriginalGeometry)

        self.__updateButtons()

    def accept(self):
        self.__originalGeometry = None
        return qt.QDialog.accept(self)

    def reject(self):
        self.__originalGeometry = None
        return qt.QDialog.reject(self)

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
        geometry = self.__geometry
        if not geometry.isValid():
            raise RuntimeError("The geometry is not valid")
        dist = geometry.distance().value()
        poni1 = geometry.distance().value()
        poni2 = geometry.distance().value()
        rot1 = geometry.distance().value()
        rot2 = geometry.distance().value()
        rot3 = geometry.distance().value()
        wavelength = geometry.distance().value()
        result = Geometry(dist=dist,
                          poni1=poni1,
                          poni2=poni2,
                          rot1=rot1,
                          rot2=rot2,
                          rot3=rot3,
                          detector=self.__detector,
                          wavelength=wavelength)
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
            pyFAIGeometry = Geometry(detector=self.__detector)
            try:
                f2d_distance = geometry.distance().value()
                f2d_centerX = geometry.centerX().value()
                f2d_centerY = geometry.centerY().value()
                f2d_tiltPlan = geometry.tiltPlan().value()
                f2d_tilt = geometry.tilt().value()
                pyFAIGeometry.setFit2D(directDist=f2d_distance,
                                       centerX=f2d_centerX,
                                       centerY=f2d_centerY,
                                       tilt=f2d_tilt,
                                       tiltPlanRotation=f2d_tiltPlan)
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
        self.__geometry.lockSignals()
        self.__geometry.distance().setValue(distance)
        self.__geometry.poni1().setValue(poni1)
        self.__geometry.poni2().setValue(poni2)
        self.__geometry.rotation1().setValue(rotation1)
        self.__geometry.rotation2().setValue(rotation2)
        self.__geometry.rotation3().setValue(rotation3)
        self.__geometry.unlockSignals()
        self.__updatingModel = False

    def __updateFit2dFromPyfai(self):
        if self.__updatingModel:
            return
        self.__updatingModel = True
        geometry = self.__geometry
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
        elif not geometry.isValid():
            error = "The current geometry is not valid to compute the Fit2D one."
        else:
            pyFAIGeometry = self.__createPyfaiGeometry()
            try:
                result = pyFAIGeometry.getFit2D()
            except Exception:
                error = "This geometry can't be modelized with Fit2D."
            else:
                distance = result["directDist"]
                centerX = result["centerX"]
                centerY = result["centerY"]
                tilt = result["tilt"]
                tiltPlan = result["tiltPlanRotation"]

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

    def __resetToOriginalGeometry(self):
        if self.__originalGeometry is None:
            return
        self.__geometry.setFrom(self.__originalGeometry)

    def __updateButtons(self):
        """Update the state of the dialog's buttons"""
        haveChanges = self.__geometry != self.__originalGeometry
        existing = [qt.QDialogButtonBox.Ok, qt.QDialogButtonBox.Cancel, qt.QDialogButtonBox.Reset, qt.QDialogButtonBox.Close]
        if haveChanges:
            available = set([qt.QDialogButtonBox.Ok, qt.QDialogButtonBox.Cancel, qt.QDialogButtonBox.Reset])
        else:
            available = set([qt.QDialogButtonBox.Close])
        for buttonType in existing:
            button = self._buttonBox.button(buttonType)
            isVisible = buttonType in available
            button.setVisible(isVisible)

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
        assert(isinstance(geometryModel, GeometryModel))
        if self.__geometry is geometryModel:
            return
        self.__originalGeometry = geometryModel
        self.__geometry.changed.disconnect(self.__updateButtons)
        self.__geometry.setFrom(geometryModel)
        self.__geometry.changed.connect(self.__updateButtons)
        self.__updateButtons()

    def geometryModel(self):
        """Returns the geometry model

        :rtype: ~pyFAI.gui.model.GeometryModel
        """
        return self.__geometry
