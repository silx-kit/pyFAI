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
__date__ = "03/03/2017"

import logging
from .model.DataModel import DataModel
from pyFAI.gui import qt
import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

import silx.gui.plot
from . import utils

_logger = logging.getLogger(__name__)


class EnablableDataModel(DataModel):

    def __init__(self, parent, model):
        DataModel.__init__(self, parent=parent)
        self.__model = model
        self.__model.changed.connect(self.__modelChanged)
        self.__isEnabled = False
        self.__modelChanged()

    def setEnabled(self, isEnabled):
        if self.__isEnabled == isEnabled:
            return
        self.__isEnabled = isEnabled
        if self.__isEnabled:
            self.__model.setValue(self.value())
        else:
            self.__model.setValue(None)
        self.wasChanged()

    def isEnabled(self):
        return self.__isEnabled

    def __modelChanged(self):
        value = self.__model.value()
        if self.value() == value:
            return
        self.lockSignals()
        self.setEnabled(value is not None)
        if value is not None:
            self.setValue(value)
        self.unlockSignals()

    def setValue(self, value):
        super(EnablableDataModel, self).setValue(value)
        if self.__isEnabled:
            self.__model.setValue(value)


class IntegrationTask(AbstractCalibrationTask):

    def __init__(self):
        super(IntegrationTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-result.ui"), self)

        self.__integrationUpToDate = True
        self.__ringLegends = []
        self.__plot1d = silx.gui.plot.Plot1D(self)
        self.__plot1d.setGraphXLabel("Radial")
        self.__plot1d.setGraphYLabel("Intensity")
        self.__plot2d = silx.gui.plot.Plot2D(self)
        self.__plot2d.setGraphXLabel("Radial")
        self.__plot2d.setGraphYLabel("Azimuthal")
        colormap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        self.__plot2d.setDefaultColormap(colormap)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.__plot2d)
        layout.addWidget(self.__plot1d)
        self._integrateButton.clicked.connect(self.__invalidateIntegration)
        self._radialUnit.setUnits(pyFAI.units.RADIAL_UNITS.values())
        self.__polarizationModel = None
        self._polarizationFactorCheck.clicked[bool].connect(self.__polarizationFactorChecked)
        self.widgetShow.connect(self.__widgetShow)

    def __polarizationFactorChecked(self, checked):
        self.__polarizationModel.setEnabled(checked)
        self._polarizationFactor.setEnabled(checked)

    def __polarizationModelChanged(self):
        old = self._polarizationFactorCheck.blockSignals(True)
        isEnabled = self.__polarizationModel.isEnabled()
        self._polarizationFactorCheck.setChecked(isEnabled)
        self._polarizationFactor.setEnabled(isEnabled)
        self._polarizationFactorCheck.blockSignals(old)

    def __invalidateIntegration(self):
        if self.isVisible():
            self.__integrate()
            self.__integrationUpToDate = True
        else:
            self.__integrationUpToDate = False

    def __widgetShow(self):
        if not self.__integrationUpToDate:
            self.__integrate()
            self.__integrationUpToDate = True

    def __integrate(self):
        model = self.model()
        if model is None:
            return
        image = model.experimentSettingsModel().image().value()
        if image is None:
            return
        mask = model.experimentSettingsModel().mask().value()
        detector = model.experimentSettingsModel().detectorModel().detector()
        if detector is None:
            return
        geometry = model.fittedGeometry()
        if not geometry.isValid():
            return
        radialUnit = model.integrationSettingsModel().radialUnit().value()
        if radialUnit is None:
            return
        polarizationFactor = model.experimentSettingsModel().polarizationFactor().value()

        wavelength = geometry.wavelength().value()
        wavelength = wavelength / 1e10
        distance = geometry.distance().value()
        poni1 = geometry.poni1().value()
        poni2 = geometry.poni2().value()
        rotation1 = geometry.rotation1().value()
        rotation2 = geometry.rotation2().value()
        rotation3 = geometry.rotation3().value()

        ai = AzimuthalIntegrator(
            dist=distance,
            poni1=poni1,
            poni2=poni2,
            rot1=rotation1,
            rot2=rotation2,
            rot3=rotation3,
            detector=detector,
            wavelength=wavelength)

        numberPoint1D = 1024
        numberPointRadial = 400
        numberPointAzimuthal = 360

        # FIXME error model, method

        result1d = ai.integrate1d(
            data=image,
            npt=numberPoint1D,
            unit=radialUnit,
            mask=mask,
            polarization_factor=polarizationFactor)

        result2d = ai.integrate2d(
            data=image,
            npt_rad=numberPointRadial,
            npt_azim=numberPointAzimuthal,
            unit=radialUnit,
            polarization_factor=polarizationFactor)

        # Add a marker for each rings on the plots
        calibrant = model.experimentSettingsModel().calibrantModel().calibrant()
        if calibrant:
            for legend in self.__ringLegends:
                self.__plot1d.removeMarker(legend)
                self.__plot2d.removeMarker(legend)
            self.__ringLegends = []

            colors = utils.getFreeColorRange(self.__plot2d.getDefaultColormap())
            rings = calibrant.get_2th()
            rings = filter(lambda x: x <= result1d.radial[-1], rings)
            try:
                rings = utils.from2ThRad(rings, radialUnit, wavelength, ai)
            except ValueError:
                message = "Convertion to unit %s not supported. Ring marks ignored"
                _logger.warning(message, radialUnit)
                rings = []
            for i, angle in enumerate(rings):
                legend = "ring_%i" % (i + 1)
                color = colors[i % len(colors)]
                htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
                self.__plot1d.addXMarker(x=angle, color=htmlColor, legend=legend)
                self.__plot2d.addXMarker(x=angle, color=htmlColor, legend=legend)
                self.__ringLegends.append(legend)

        # FIXME set axes
        self.__plot1d.addCurve(
            legend="result1d",
            x=result1d.radial,
            y=result1d.intensity)

        # Assume that axes are linear
        origin = (result2d.radial[0], result2d.azimuthal[0])
        scaleX = (result2d.radial[-1] - result2d.radial[0]) / result2d.intensity.shape[1]
        scaleY = (result2d.azimuthal[-1] - result2d.azimuthal[0]) / result2d.intensity.shape[0]
        self.__plot2d.addImage(
            legend="result2d",
            data=result2d.intensity,
            origin=origin,
            scale=(scaleX, scaleY))

    def _updateModel(self, model):
        experimentSettings = model.experimentSettingsModel()
        integrationSettings = model.integrationSettingsModel()
        self.__polarizationModel = EnablableDataModel(self, experimentSettings.polarizationFactor())
        if self.__polarizationModel.value() is None:
            self.__polarizationModel.setValue(0.9)
        # connect widgets
        self.__polarizationModelChanged()
        self._polarizationFactor.setModel(self.__polarizationModel)
        self._radialUnit.setModel(integrationSettings.radialUnit())
        # connect model
        self.__polarizationModel.changed.connect(self.__polarizationModelChanged)
        experimentSettings.mask().changed.connect(self.__invalidateIntegration)
        experimentSettings.polarizationFactor().changed.connect(self.__invalidateIntegration)
        model.fittedGeometry().changed.connect(self.__invalidateIntegration)
        integrationSettings.radialUnit().changed.connect(self.__invalidateIntegration)
