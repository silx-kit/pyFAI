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
__date__ = "20/08/2018"

import logging
import numpy

from silx.gui import qt
from silx.gui import icons
import silx.gui.plot
from silx.gui.plot.tools import PositionInfo

import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.RingCalibration import RingCalibration
from . import utils
from .helper.SynchronizeRawView import SynchronizeRawView
from .CalibrationContext import CalibrationContext
from ..widgets.UnitLabel import UnitLabel
from .model.DataModel import DataModel
from .QuantityEdit import QuantityEdit
from . import units

_logger = logging.getLogger(__name__)

_iconVariableFixed = None
_iconVariableConstrained = None
_iconVariableConstrainedOut = None


class FitParamView(qt.QObject):

    def __init__(self, parent, label, internalUnit, displayedUnit=None):
        qt.QObject.__init__(self, parent=parent)
        self.__label = qt.QLabel(parent)
        self.__label.setText(label)
        self.__quantity = QuantityEdit(parent)
        self.__quantity.setAlignment(qt.Qt.AlignRight)
        self.__unit = UnitLabel(parent)
        self.__unit.setUnitEditable(True)

        if displayedUnit is None:
            displayedUnit = internalUnit

        self.__quantity.setModelUnit(internalUnit)

        if isinstance(displayedUnit, units.Unit):
            model = DataModel()
            model.setValue(displayedUnit)
            displayedUnit = model
        elif isinstance(displayedUnit, DataModel):
            pass
        else:
            raise TypeError("Unsupported type %s" % type(displayedUnit))
        self.__unit.setUnitModel(displayedUnit)
        self.__quantity.setDisplayedUnitModel(displayedUnit)

        self.__constraints = qt.QToolButton(parent)
        self.__constraints.setAutoRaise(True)
        self.__constraints.clicked.connect(self.__constraintsClicked)
        self.__model = None
        self.__constraintsModel = None

        global _iconVariableFixed, _iconVariableConstrained, _iconVariableConstrainedOut
        if _iconVariableFixed is None:
            _iconVariableFixed = icons.getQIcon("pyfai:gui/icons/variable-fixed")
        if _iconVariableConstrained is None:
            _iconVariableConstrained = icons.getQIcon("pyfai:gui/icons/variable-constrained")
        if _iconVariableConstrainedOut is None:
            _iconVariableConstrainedOut = icons.getQIcon("pyfai:gui/icons/variable-constrained-out")

    def model(self):
        return self.__model

    def setModel(self, model):
        self.__quantity.setModel(model)
        self.__model = model

    def setConstraintsModel(self, model):
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.disconnect(self.__constraintsModelChanged)
        self.__constraintsModel = model
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.connect(self.__constraintsModelChanged)
            self.__constraintsModelChanged()

    def __constraintsModelChanged(self):
        constraint = self.__constraintsModel
        if constraint.isFixed():
            icon = _iconVariableFixed
        else:
            icon = _iconVariableConstrained
        self.__constraints.setIcon(icon)

    def __constraintsClicked(self):
        constraint = self.__constraintsModel
        # FIXME implement popup with range
        constraint.setFixed(not constraint.isFixed())

    def widgets(self):
        return [self.__label, self.__quantity, self.__unit, self.__constraints]


class _RingPlot(silx.gui.plot.PlotWidget):

    def __init__(self, parent=None):
        silx.gui.plot.PlotWidget.__init__(self, parent=parent)
        self.__markerColors = {}
        self.getXAxis().sigLimitsChanged.connect(self.__axesChanged)
        self.getYAxis().sigLimitsChanged.connect(self.__axesChanged)
        self.sigPlotSignal.connect(self.__plotSignalReceived)
        self.__axisOfCurrentView = None
        self.__tth = None
        self.__rings = []
        self.__ringItems = {}
        self.__angleUnderMouse = None
        self.__displayedAngles = []
        self.__processing = None

        if hasattr(self, "centralWidget"):
            self.centralWidget().installEventFilter(self)

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Leave:
            self.__mouseLeave()
            return True
        return False

    def __plotSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)

    def __getClosestAngle(self, angle):
        """
        Returns the closest ring index and ring angle
        """
        # TODO: Could be done in log(n) using bisect search
        result = None
        iresult = None
        minDistance = float("inf")
        for ringId, data in enumerate(self.__rings):
            ringAngle, _polygons = data
            distance = abs(angle - ringAngle)
            if distance < minDistance:
                minDistance = distance
                result = ringAngle
                iresult = ringId
        return iresult, result

    def __mouseLeave(self):
        if self.__angleUnderMouse is None:
            return
        if self.__angleUnderMouse not in self.__displayedAngles:
            items = self.__ringItems.get(self.__angleUnderMouse, [])
            for item in items:
                item.setVisible(False)
        self.__angleUnderMouse = None

    def __mouseMoved(self, x, y):
        """Called when mouse move over the plot."""
        pos = int(x), int(y)
        if self.__tth is None:
            return
        x, y = self.__clampOnImage(pos)
        angle = self.__tth[y, x]
        ringId, angle = self.__getClosestAngle(angle)

        if angle == self.__angleUnderMouse:
            return

        if self.__angleUnderMouse not in self.__displayedAngles:
            items = self.__ringItems.get(self.__angleUnderMouse, [])
            for item in items:
                item.setVisible(False)

        self.__angleUnderMouse = angle

        if angle is not None:
            items = self.__getItemsFromAngle(ringId, angle)
            for item in items:
                item.setVisible(True)

    def markerColorList(self):
        colormap = self.getDefaultColormap()

        name = colormap['name']
        if name not in self.__markerColors:
            colors = self.createMarkerColors()
            self.__markerColors[name] = colors
        else:
            colors = self.__markerColors[name]
        return colors

    def createMarkerColors(self):
        colormap = self.getDefaultColormap()
        return utils.getFreeColorRange(colormap)

    def __clampOnImage(self, pos):
        x, y = pos
        x, y = int(x), int(y)
        if x < 0:
            x = 0
        elif x >= self.__tth.shape[1]:
            x = self.__tth.shape[1] - 1
        if y < 0:
            y = 0
        elif y >= self.__tth.shape[0]:
            y = self.__tth.shape[0] - 1
        return x, y

    def __getTwoTheraRange(self):
        if self.__tth is None:
            return None, None
        xmin, xmax = self.getXAxis().getLimits()
        xmin, xmax = int(xmin) - 1, int(xmax) + 1
        ymin, ymax = self.getYAxis().getLimits()
        ymin, ymax = int(ymin) - 1, int(ymax) + 1

        xmin, ymin = self.__clampOnImage((xmin, ymin))
        xmax, ymax = self.__clampOnImage((xmax, ymax))

        view = self.__tth[ymin:ymax + 1, xmin:xmax + 1]
        vmin, vmax = view.min(), view.max()
        return vmin, vmax

    def __axesChanged(self, minValue, maxValue):
        axisOfCurrentView = self.getXAxis().getLimits(), self.getYAxis().getLimits()
        if self.__axisOfCurrentView == axisOfCurrentView:
            return
        self.__updateRings()

    def __getAvailableAngles(self, minTth, maxTth):
        result = []
        for ringId, data in enumerate(self.__rings):
            angle, _polygons = data
            if minTth is None or maxTth is None:
                result.append(ringId, angle)
            if minTth <= angle <= maxTth:
                result.append((ringId, angle))
        return result

    def __updateRings(self):
        minTth, maxTth = self.__getTwoTheraRange()
        angles = self.__getAvailableAngles(minTth, maxTth)

        if len(angles) < 20:
            step = 1
        elif len(angles) < 100:
            step = 2
        elif len(angles) < 200:
            step = 5
        elif len(angles) < 500:
            step = 10
        elif len(angles) < 1000:
            step = 20
        elif len(angles) < 5000:
            step = 100
        else:
            step = int(len(angles) / 50)

        self.__displayedAngles = set([])

        for items in self.__ringItems.values():
            for item in items:
                item.setVisible(False)

        for angleId in range(0, len(angles), step):
            ringId, ringAngle = angles[angleId]
            self.__displayedAngles.add(ringAngle)
            items = self.__getItemsFromAngle(ringId, ringAngle)
            for item in items:
                item.setVisible(True)

    def __getItemsFromAngle(self, ringId, ringAngle):
        items = self.__ringItems.get(ringAngle, None)
        if items is not None:
            return items

        colors = self.markerColorList()

        polyline = self.__rings[ringId][1]
        color = colors[ringId % len(colors)]
        numpyColor = numpy.array([color.redF(), color.greenF(), color.blueF()])
        items = []
        for lineId, line in enumerate(polyline):
            y, x = line[:, 0], line[:, 1]
            legend = "ring-%i-%i" % (ringId, lineId)
            self.addCurve(
                x=x,
                y=y,
                selectable=False,
                legend=legend,
                resetzoom=False,
                color=numpyColor,
                linewidth=1,
                linestyle=":",
                copy=False)
            item = self.getCurve(legend)
            items.append(item)
        self.__ringItems[ringAngle] = items
        return items

    def setRings(self, rings, tth=None):
        self.__tth = tth
        self.__rings = rings
        for items in self.__ringItems.values():
            for item in items:
                self.removeCurve(item.getLegend())
        self.__ringItems = {}
        self.__updateRings()

    def unsetProcessing(self):
        self.__processing.deleteLater()

    def setProcessing(self):
        self.__processing = utils.createProcessingWidgetOverlay(self)


class GeometryTask(AbstractCalibrationTask):

    def __init__(self):
        super(GeometryTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-geometry.ui"), self)
        self.initNextStep()
        self.widgetShow.connect(self.__widgetShow)

        self.__plot = self.__createPlot()

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        layout = qt.QGridLayout(self._settings)
        self.__wavelength = FitParamView(self, "Wavelength:", units.Unit.METER_WL, units.Unit.ANGSTROM)
        self.addParameterToLayout(layout, self.__wavelength)

        layout = qt.QGridLayout(self._geometry)
        self.__distance = FitParamView(self, "Distance:", units.Unit.METER)
        self.__poni1 = FitParamView(self, "PONI1:", units.Unit.METER)
        self.__poni2 = FitParamView(self, "PONI2:", units.Unit.METER)

        userAngleUnit = CalibrationContext.instance().getAngleUnit()

        self.__rotation1 = FitParamView(self, "Rotation 1:", units.Unit.RADIAN, userAngleUnit)
        self.__rotation2 = FitParamView(self, "Rotation 2:", units.Unit.RADIAN, userAngleUnit)
        self.__rotation3 = FitParamView(self, "Rotation 3:", units.Unit.RADIAN, userAngleUnit)
        self.addParameterToLayout(layout, self.__distance)
        self.addParameterToLayout(layout, self.__poni1)
        self.addParameterToLayout(layout, self.__poni2)
        self.addParameterToLayout(layout, self.__rotation1)
        self.addParameterToLayout(layout, self.__rotation2)
        self.addParameterToLayout(layout, self.__rotation3)

        self._fitButton.clicked.connect(self.__fitGeometryLater)
        self._fitButton.setDisabledWhenWaiting(True)
        self._resetButton.clicked.connect(self.__resetGeometryLater)
        self.__calibration = None
        self.__peaksInvalidated = False
        self.__fitting = False
        self.__wavelengthInvalidated = False

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

    def addParameterToLayout(self, layout, param):
        # an empty grid returns 1
        row = layout.rowCount()
        widgets = param.widgets()
        for i, widget in enumerate(widgets):
            layout.addWidget(widget, row, i)

    def __createPlot(self):
        plot = _RingPlot(parent=self._imageHolder)
        plot.setKeepDataAspectRatio(True)
        self.__createPlotToolBar(plot)
        statusBar = self.__createPlotStatusBar(plot)
        plot.setStatusBar(statusBar)
        plot.setAxesDisplayed(False)

        colormap = CalibrationContext.instance().getRawColormap()
        plot.setDefaultColormap(colormap)

        return plot

    def __createPlotToolBar(self, plot):
        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        colormapDialog = CalibrationContext.instance().getColormapDialog()
        toolBar.getColormapAction().setColorDialog(colormapDialog)
        plot.addToolBar(toolBar)

    def __createPlotStatusBar(self, plot):

        converters = [
            ('X', lambda x, y: x),
            ('Y', lambda x, y: y),
            ('Value', self.__getImageValue)]

        hbox = qt.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)

        info = PositionInfo(plot=plot, converters=converters)
        info.setSnappingMode(True)
        statusBar = qt.QStatusBar(plot)
        statusBar.setSizeGripEnabled(False)
        statusBar.addWidget(info)
        return statusBar

    def __createCalibration(self):
        image = self.model().experimentSettingsModel().image().value()
        mask = self.model().experimentSettingsModel().mask().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()

        peaks = []
        for peakModel in self.model().peakSelectionModel():
            ringNumber = peakModel.ringNumber()
            for coord in peakModel.coords():
                peaks.append([coord[0], coord[1], ringNumber - 1])
        peaks = numpy.array(peaks)

        calibration = RingCalibration(image,
                                      mask,
                                      calibrant,
                                      detector,
                                      wavelength,
                                      peaks=peaks,
                                      method="massif")

        return calibration

    def __invalidateWavelength(self):
        self.__wavelengthInvalidated = True

    def __getCalibration(self):
        if self.__calibration is None:
            self.__calibration = self.__createCalibration()

        # It have to be updated only if it changes
        image = self.model().experimentSettingsModel().image().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        mask = self.model().experimentSettingsModel().mask().value()
        if self.__wavelengthInvalidated:
            self.__wavelengthInvalidated = False
            wavelength = self.model().experimentSettingsModel().wavelength().value()
        else:
            wavelength = None
        self.__calibration.update(image, mask, calibrant, detector, wavelength)

        return self.__calibration

    def __invalidatePeakSelection(self):
        self.__peaksInvalidated = True

    def __initGeometryFromPeaks(self):
        if self.__peaksInvalidated:
            # recompute the geometry from the peaks
            # FIXME numpy array can be allocated first
            peaks = []
            for peakModel in self.model().peakSelectionModel():
                ringNumber = peakModel.ringNumber()
                for coord in peakModel.coords():
                    peaks.append([coord[0], coord[1], ringNumber - 1])
            peaks = numpy.array(peaks)

            calibration = self.__getCalibration()
            calibration.init(peaks, "massif")
            calibration.toGeometryModel(self.model().peakGeometry())
            self.__peaksInvalidated = False

        self.model().fittedGeometry().setFrom(self.model().peakGeometry())

    def __initGeometryLater(self):
        self.__plot.setProcessing()
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__initGeometry)

    def __resetGeometryLater(self):
        self.__plot.setProcessing()
        self._resetButton.setWaiting(True)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__resetGeometry)

    def __fitGeometryLater(self):
        self.__plot.setProcessing()
        self._fitButton.setWaiting(True)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__fitGeometry)

    def __unsetProcessing(self):
        self.__plot.unsetProcessing()
        self._resetButton.setWaiting(False)
        self._fitButton.setWaiting(False)

    def __initGeometry(self):
        self.__initGeometryFromPeaks()
        self.__formatResidual()
        self.__unsetProcessing()

    def __resetGeometry(self):
        calibration = self.__getCalibration()
        self.__initGeometryFromPeaks()
        # write result to the fitted model
        model = self.model().fittedGeometry()
        calibration.toGeometryModel(model)
        self.__formatResidual()
        self.__unsetProcessing()

    def __fitGeometry(self):
        self.__fitting = True
        self._fitButton.setWaiting(True)
        calibration = self.__getCalibration()
        if self.__peaksInvalidated:
            self.__initGeometryFromPeaks()
        else:
            calibration.fromGeometryModel(self.model().fittedGeometry())
        calibration.fromGeometryConstriansModel(self.model().geometryConstraintsModel())
        calibration.refine()
        # write result to the fitted model
        model = self.model().fittedGeometry()
        calibration.toGeometryModel(model)
        self._fitButton.setWaiting(False)
        self.__fitting = False
        self.__unsetProcessing()

    def __formatResidual(self):
        calibration = self.__getCalibration()
        previousResidual = calibration.getPreviousResidual()
        residual = calibration.getResidual()
        text = '%.6e' % residual
        if previousResidual is not None:
            if residual == previousResidual:
                diff = "(no changes)"
            else:
                diff = '(%+.2e)' % (residual - previousResidual)
                if residual < previousResidual:
                    diff = '<font color="green">%s</font>' % diff
                else:
                    diff = '<font color="red">%s</font>' % diff
            text = '%s %s' % (text, diff)
        self._currentResidual.setText(text)

    def __geometryUpdated(self):
        calibration = self.__getCalibration()
        model = self.model().fittedGeometry()
        if model.isValid():
            resetResidual = self.__fitting is not True
            calibration.fromGeometryModel(model, resetResidual=resetResidual)
            self.__updateDisplay()
            self.__formatResidual()

    def __updateDisplay(self):
        calibration = self.__getCalibration()

        rings = calibration.getRings()
        tth = calibration.getTwoThetaArray()
        self.__plot.setRings(rings, tth)

        center = calibration.getBeamCenter()
        if center is None:
            self.__plot.removeMarker(legend="center")
        else:
            color = self.__plot.markerColorList()[0]
            htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot.addMarker(
                text="Beam",
                y=center[0],
                x=center[1],
                legend="center",
                color=htmlColor,
                symbol="+")

        poni = calibration.getPoni()
        if poni is None:
            self.__plot.removeMarker(legend="poni")
        else:
            color = self.__plot.markerColorList()[0]
            htmlColor = "#%02X%02X%02X" % (color.red(), color.green(), color.blue())
            self.__plot.addMarker(
                text="PONI",
                y=poni[0],
                x=poni[1],
                legend="poni",
                color=htmlColor,
                symbol="+")

    def __getImageValue(self, x, y):
        """Get value of top most image at position (x, y).

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or 'n/a'
        """
        value = 'n/a'

        image = self.__plot.getImage("image")
        if image is None:
            return value
        data = image.getData(copy=False)
        ox, oy = image.getOrigin()
        sx, sy = image.getScale()
        row, col = (y - oy) / sy, (x - ox) / sx
        if row >= 0 and col >= 0:
            # Test positive before cast otherwise issue with int(-0.5) = 0
            row, col = int(row), int(col)
            if (row < data.shape[0] and col < data.shape[1]):
                value = data[row, col]
        return value

    def _updateModel(self, model):
        self.__synchronizeRawView.registerModel(model.rawPlotView())
        settings = model.experimentSettingsModel()
        settings.image().changed.connect(self.__imageUpdated)
        settings.wavelength().changed.connect(self.__invalidateWavelength)

        geometry = model.fittedGeometry()

        self.__distance.setModel(geometry.distance())
        self.__wavelength.setModel(geometry.wavelength())
        self.__poni1.setModel(geometry.poni1())
        self.__poni2.setModel(geometry.poni2())
        self.__rotation1.setModel(geometry.rotation1())
        self.__rotation2.setModel(geometry.rotation2())
        self.__rotation3.setModel(geometry.rotation3())

        constrains = model.geometryConstraintsModel()
        self.__distance.setConstraintsModel(constrains.distance())
        self.__wavelength.setConstraintsModel(constrains.wavelength())
        self.__poni1.setConstraintsModel(constrains.poni1())
        self.__poni2.setConstraintsModel(constrains.poni2())
        self.__rotation1.setConstraintsModel(constrains.rotation1())
        self.__rotation2.setConstraintsModel(constrains.rotation2())
        self.__rotation3.setConstraintsModel(constrains.rotation3())

        model.fittedGeometry().changed.connect(self.__geometryUpdated)
        model.peakSelectionModel().changed.connect(self.__invalidatePeakSelection)

        self.__imageUpdated()

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image")
        if image is not None:
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()

    def __widgetShow(self):
        if self.__peaksInvalidated:
            self.__initGeometryLater()
