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
__date__ = "07/12/2018"

from silx.gui import qt

from pyFAI.utils import get_ui_file
from ..utils import units
from ..calibration.model.DataModel import DataModel
from ..calibration.model.GeometryModel import GeometryModel
from pyFAI.geometry import Geometry


class GeometryDialog(qt.QDialog):
    """Dialog to display a selected geometry
    """

    def __init__(self, parent=None):
        super(GeometryDialog, self).__init__(parent)
        filename = get_ui_file("geometry-dialog.ui")
        qt.loadUi(filename, self)

        self.__geometry = None
        self.__detector = None

        # Connect close button
        self._buttonBox.rejected.connect(self.reject)

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
        self.__fit2dDistance = DataModel()
        self.__fit2dCenterX = DataModel()
        self.__fit2dCenterY = DataModel()
        self.__fit2dTilt = DataModel()
        self.__fit2dTiltPlan = DataModel()
        self._fit2dDistance.setModel(self.__fit2dDistance)
        self._fit2dCenterX.setModel(self.__fit2dCenterX)
        self._fit2dCenterY.setModel(self.__fit2dCenterY)
        self._fit2dTilt.setModel(self.__fit2dTilt)
        self._fit2dTiltPlan.setModel(self.__fit2dTiltPlan)

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

    def __updateFid2dModel(self):
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
        elif not geometry.isValid():
            error = "The current geometry is not valid to compute the Fit2D one."
        elif self.__detector is None:
            error = "No detector defined. It is needed to compute the Fit2D geometry."
            self._fit2dError.setText("")
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
        self.__fit2dDistance.setValue(distance)
        self.__fit2dCenterX.setValue(centerX)
        self.__fit2dCenterY.setValue(centerY)
        self.__fit2dTilt.setValue(tilt)
        self.__fit2dTiltPlan.setValue(tiltPlan)

    def setDetector(self, detector):
        """Set the used detector.

        This information is needed to display the Fit2D geometry.
        """
        self.__detector = detector
        self.__updateFid2dModel()

    def setGeometryModel(self, geometryModel):
        """Set the geometry to display.

        :param ~pyFAI.gui.calibration.model.GeometryModel geometryModel: A geometry.
        """
        assert(isinstance(geometryModel, GeometryModel))
        if self.__geometry is geometryModel:
            return
        if self.__geometry is not None:
            self.__geometry.changed.disconnect(self.__updateFid2dModel)
        self.__geometry = geometryModel
        if self.__geometry is not None:
            self.__geometry.changed.connect(self.__updateFid2dModel)
        self._pyfaiDistance.setModel(self.__geometry.distance())
        self._pyfaiPoni1.setModel(self.__geometry.poni1())
        self._pyfaiPoni2.setModel(self.__geometry.poni2())
        self._pyfaiRotation1.setModel(self.__geometry.rotation1())
        self._pyfaiRotation2.setModel(self.__geometry.rotation2())
        self._pyfaiRotation3.setModel(self.__geometry.rotation3())
        self.__updateFid2dModel()

    def geometryModel(self):
        """Returns the geometry model

        :rtype: ~pyFAI.gui.calibration.model.GeometryModel
        """
        return self.__geometry
