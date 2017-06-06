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
__date__ = "01/06/2017"

import logging
import numpy
from collections import OrderedDict
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


class IntegrationProcess(object):

    def __init__(self, model):
        self.__isValid = self._init(model)

    def _init(self, model):
        self.__isValid = True
        if model is None:
            return False
        image = model.experimentSettingsModel().image().value()
        if image is None:
            return False
        mask = model.experimentSettingsModel().mask().value()
        detector = model.experimentSettingsModel().detector()
        if detector is None:
            return
        geometry = model.fittedGeometry()
        if not geometry.isValid():
            return False
        self.__radialUnit = model.integrationSettingsModel().radialUnit().value()
        if self.__radialUnit is None:
            return False
        self.__polarizationFactor = model.experimentSettingsModel().polarizationFactor().value()

        self.__calibrant = model.experimentSettingsModel().calibrantModel().calibrant()

        if mask is not None:
            mask = numpy.array(mask)
        if image is not None:
            image = numpy.array(image)

        # FIXME calibrant and detector have to be cloned
        self.__detector = detector
        self.__image = image
        self.__mask = mask

        wavelength = geometry.wavelength().value()
        self.__wavelength = wavelength / 1e10
        self.__distance = geometry.distance().value()
        self.__poni1 = geometry.poni1().value()
        self.__poni2 = geometry.poni2().value()
        self.__rotation1 = geometry.rotation1().value()
        self.__rotation2 = geometry.rotation2().value()
        self.__rotation3 = geometry.rotation3().value()
        return True

    def isValid(self):
        return self.__isValid

    def run(self):
        ai = AzimuthalIntegrator(
            dist=self.__distance,
            poni1=self.__poni1,
            poni2=self.__poni2,
            rot1=self.__rotation1,
            rot2=self.__rotation2,
            rot3=self.__rotation3,
            detector=self.__detector,
            wavelength=self.__wavelength)

        numberPoint1D = 1024
        numberPointRadial = 400
        numberPointAzimuthal = 360

        # FIXME error model, method

        self.__result1d = ai.integrate1d(
            data=self.__image,
            npt=numberPoint1D,
            unit=self.__radialUnit,
            mask=self.__mask,
            polarization_factor=self.__polarizationFactor)

        self.__result2d = ai.integrate2d(
            data=self.__image,
            npt_rad=numberPointRadial,
            npt_azim=numberPointAzimuthal,
            unit=self.__radialUnit,
            mask=self.__mask,
            polarization_factor=self.__polarizationFactor)

        if self.__calibrant:

            rings = self.__calibrant.get_2th()
            rings = filter(lambda x: x <= self.__result1d.radial[-1], rings)
            rings = list(rings)
            try:
                rings = utils.from2ThRad(rings, self.__radialUnit, self.__wavelength, ai)
            except ValueError:
                message = "Convertion to unit %s not supported. Ring marks ignored"
                _logger.warning(message, self.__radialUnit)
                rings = []
        else:
            rings = []
        self.__ringAngles = rings

    def ringAngles(self):
        return self.__ringAngles

    def result1d(self):
        return self.__result1d

    def result2d(self):
        return self.__result2d


class IntegrationTask(AbstractCalibrationTask):

    def __init__(self):
        super(IntegrationTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-result.ui"), self)
        self.initNextStep()

        self.__integrationUpToDate = True
        self.__ringLegends = []
        self.__plot1d = silx.gui.plot.Plot1D(self)
        self.__plot1d.setGraphXLabel("Radial")
        self.__plot1d.setGraphYLabel("Intensity")
        self.__plot1d.setGraphGrid(True)
        self.__plot2d = silx.gui.plot.Plot2D(self)
        self.__plot2d.setGraphXLabel("Radial")
        self.__plot2d.setGraphYLabel("Azimuthal")

        self.__defaultColorMap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        self.__plot2d.setDefaultColormap(self.__defaultColorMap)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.__plot2d)
        layout.addWidget(self.__plot1d)
        self._radialUnit.setUnits(pyFAI.units.RADIAL_UNITS.values())
        self.__polarizationModel = None
        self._polarizationFactorCheck.clicked[bool].connect(self.__polarizationFactorChecked)
        self.widgetShow.connect(self.__widgetShow)

        self._integrateButton.beforeExecuting.connect(self.__integrate)
        self._integrateButton.setDisabledWhenWaiting(True)
        self._integrateButton.finished.connect(self.__integratingFinished)

        self._savePoniButton.clicked.connect(self.__saveAsPoni)
        self._saveJsonButton.clicked.connect(self.__saveAsJson)

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
            if not self._integrateButton.isWaiting():
                self._integrateButton.executeCallable()
            else:
                # integration is processing
                # but data are already outdated
                self.__integrationUpToDate = False
        else:
            # We can process data later
            self.__integrationUpToDate = False

    def __widgetShow(self):
        if not self.__integrationUpToDate:
            self._integrateButton.executeCallable()

    def __integrate(self):
        self.__integrationProcess = IntegrationProcess(self.model())
        if not self.__integrationProcess.isValid():
            self.__integrationProcess = None
            return
        self.__updateGUIWhileIntegrating()
        self._integrateButton.setCallable(self.__integrationProcess.run)
        self.__integrationUpToDate = True

    def __integratingFinished(self):
        self.__updateGUIWithIntegrationResult(self.__integrationProcess)
        self.__integrationProcess = None
        if not self.__integrationUpToDate:
            # Maybe it was invalidated while priocessing
            self._integrateButton.executeCallable()

    def __updateGUIWhileIntegrating(self):
        self.__defaultColorMap = self.__plot2d.getDefaultColormap()
        colormap = {
            'name': "Greys",
            'normalization': 'log',
            'autoscale': True,
        }
        self.__plot2d.setDefaultColormap(colormap)

        # Update active image
        image = self.__plot2d.getActiveImage()
        if image is not None:
            # Update image: This do not preserve pixmap
            self.__plot2d.addImage(
                image.getData(copy=False),
                legend=image.getLegend(),
                info=image.getInfo(),
                colormap=colormap,
                replace=False,
                resetzoom=False)

    def __updateGUIWithIntegrationResult(self, integrationProcess):
        colormap = self.__defaultColorMap
        self.__plot2d.setDefaultColormap(colormap)

        # Add a marker for each rings on the plots
        ringAngles = integrationProcess.ringAngles()
        for legend in self.__ringLegends:
            self.__plot1d.removeMarker(legend)
            self.__plot2d.removeMarker(legend)
        self.__ringLegends = []
        colors = utils.getFreeColorRange(self.__plot2d.getDefaultColormap())
        for i, angle in enumerate(ringAngles):
            legend = "ring_%i" % (i + 1)
            color = colors[i % len(colors)]
            htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot1d.addXMarker(x=angle, color=htmlColor, legend=legend)
            self.__plot2d.addXMarker(x=angle, color=htmlColor, legend=legend)
            self.__ringLegends.append(legend)

        # FIXME set axes
        result1d = integrationProcess.result1d()
        if hasattr(self.__plot1d, "addHistogram"):
            # silx >= 0.5
            # Removing item fixes bug in silx 0.5 when histogram data changed
            self.__plot1d.remove(legend="result1d", kind="histogram")
            self.__plot1d.addHistogram(
                legend="result1d",
                align="right",
                edges=result1d.radial,
                color="blue",
                histogram=result1d.intensity)
        else:
            # older
            self.__plot1d.addCurve(
                legend="result1d",
                x=result1d.radial,
                y=result1d.intensity)

        # Assume that axes are linear
        result2d = integrationProcess.result2d()
        origin = (result2d.radial[0], result2d.azimuthal[0])
        scaleX = (result2d.radial[-1] - result2d.radial[0]) / result2d.intensity.shape[1]
        scaleY = (result2d.azimuthal[-1] - result2d.azimuthal[0]) / result2d.intensity.shape[0]
        self.__plot2d.addImage(
            legend="result2d",
            data=result2d.intensity,
            origin=origin,
            scale=(scaleX, scaleY),
            colormap=colormap)

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

    def createSaveDialog(self, title, poni=False, json=False):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)

        extensions = OrderedDict()
        if poni:
            extensions["PONI files"] = "*.poni"
        if json:
            extensions["JSON files"] = "*.json"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        return dialog

    def __saveAsPoni(self):
        # FIXME test the validity of the geometry before opening the dialog
        dialog = self.createSaveDialog("Save as PONI file", poni=True)
        result = dialog.exec_()
        if not result:
            return
        filename = dialog.selectedFiles()[0]

        pyfaiGeometry = pyFAI.geometry.Geometry()

        geometry = self.model().fittedGeometry()
        pyfaiGeometry.dist = geometry.distance().value()
        pyfaiGeometry.poni1 = geometry.poni1().value()
        pyfaiGeometry.poni2 = geometry.poni2().value()
        pyfaiGeometry.rot1 = geometry.rotation1().value()
        pyfaiGeometry.rot2 = geometry.rotation2().value()
        pyfaiGeometry.rot3 = geometry.rotation3().value()

        experimentSettingsModel = self.model().experimentSettingsModel()
        detector = experimentSettingsModel.detector()
        pyfaiGeometry.detector = detector
        wavelength = experimentSettingsModel.wavelength().value() * 1e-10
        pyfaiGeometry.wavelength = wavelength

        pyfaiGeometry.save(filename)

    def __saveAsJson(self):
        pass
