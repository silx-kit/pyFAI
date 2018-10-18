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
__date__ = "17/10/2018"

import logging
import numpy

from silx.gui import qt
from silx.gui import icons
import silx.gui.plot

import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.RingCalibration import RingCalibration
from . import utils
from .helper.SynchronizeRawView import SynchronizeRawView
from .CalibrationContext import CalibrationContext
from ..widgets.UnitLabel import UnitLabel
from ..widgets.QuantityLabel import QuantityLabel
from ..widgets.QuantityEdit import QuantityEdit
from .model.DataModel import DataModel
from ..utils import units
from .helper.MarkerManager import MarkerManager

_logger = logging.getLogger(__name__)


class FitParamView(qt.QObject):

    _iconVariableFixed = None
    _iconVariableConstrained = None
    _iconVariableConstrainedOut = None

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

        if self._iconVariableFixed is None:
            self._iconVariableFixed = icons.getQIcon("pyfai:gui/icons/variable-fixed")
        if self._iconVariableConstrained is None:
            self._iconVariableConstrained = icons.getQIcon("pyfai:gui/icons/variable-constrained")
        if self._iconVariableConstrainedOut is None:
            self._iconVariableConstrainedOut = icons.getQIcon("pyfai:gui/icons/variable-constrained-out")

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
            icon = self._iconVariableFixed
        else:
            icon = self._iconVariableConstrained
        self.__constraints.setIcon(icon)

    def __constraintsClicked(self):
        constraint = self.__constraintsModel
        # FIXME implement popup with range
        constraint.setFixed(not constraint.isFixed())

    def widgets(self):
        return [self.__label, self.__quantity, self.__unit, self.__constraints]


class _StatusBar(qt.QStatusBar):

    def __init__(self, parent=None):
        qt.QStatusBar.__init__(self, parent)

        angleUnitModel = CalibrationContext.instance().getAngleUnit()

        self.__position = QuantityLabel(self)
        self.__position.setPrefix(u"<b>Pos</b>: ")
        self.__position.setFormatter(u"{value[0]: >4.2F}×{value[1]:4.2F} px")
        # TODO: Could it be done using a custom layout? Instead of setElasticSize
        self.__position.setElasticSize(True)
        self.addWidget(self.__position)
        self.__pixel = QuantityLabel(self)
        self.__pixel.setPrefix(u"<b>Pixel</b>: ")
        self.__pixel.setFormatter(u"{value}")
        self.__pixel.setFloatFormatter(u"{value: >4.3F}")
        self.__pixel.setElasticSize(True)
        self.addWidget(self.__pixel)
        self.__chi = QuantityLabel(self)
        self.__chi.setPrefix(u"<b>χ</b>: ")
        self.__chi.setFormatter(u"{value: >4.3F}")
        self.__chi.setInternalUnit(units.Unit.RADIAN)
        self.__chi.setDisplayedUnit(units.Unit.RADIAN)
        self.__chi.setDisplayedUnitModel(angleUnitModel)
        self.__chi.setUnitEditable(True)
        self.__chi.setElasticSize(True)
        self.addWidget(self.__chi)
        self.__2theta = QuantityLabel(self)
        self.__2theta.setPrefix(u"<b>2θ</b>: ")
        self.__2theta.setFormatter(u"{value: >4.3F}")
        self.__2theta.setInternalUnit(units.Unit.RADIAN)
        self.__2theta.setDisplayedUnitModel(angleUnitModel)
        self.__2theta.setUnitEditable(True)
        self.__2theta.setElasticSize(True)
        self.addWidget(self.__2theta)

        self.clearValues()

    def clearValues(self):
        self.setValues(None, None, float("nan"), float("nan"), float("nan"))

    def setValues(self, x, y, pixel, chi, tth):
        if x is None:
            pos = None
        else:
            pos = x, y
        self.__position.setValue(pos)
        self.__pixel.setValue(pixel)
        self.__chi.setValue(chi)
        self.__2theta.setValue(tth)


class _RingPlot(silx.gui.plot.PlotWidget):

    sigMouseMove = qt.Signal(float, float)

    sigMouseLeave = qt.Signal()

    def __init__(self, parent=None):
        silx.gui.plot.PlotWidget.__init__(self, parent=parent)
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

        markerModel = CalibrationContext.instance().getCalibrationModel().markerModel()
        self.__markerManager = MarkerManager(self, markerModel, pixelBasedPlot=True)

        handle = self.getWidgetHandle()
        handle.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        handle.customContextMenuRequested.connect(self.__plotContextMenu)

        if hasattr(self, "centralWidget"):
            self.centralWidget().installEventFilter(self)

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Leave:
            self.__mouseLeave()
            return True
        return False

    def markerManager(self):
        return self.__markerManager

    def __plotContextMenu(self, pos):
        plot = self
        from silx.gui.plot.actions.control import ZoomBackAction
        zoomBackAction = ZoomBackAction(plot=plot, parent=plot)

        menu = qt.QMenu(self)

        menu.addAction(zoomBackAction)
        menu.addSeparator()
        menu.addAction(self.__markerManager.createMarkPixelAction(menu, pos))
        menu.addAction(self.__markerManager.createMarkGeometryAction(menu, pos))
        action = self.__markerManager.createRemoveClosestMaskerAction(menu, pos)
        if action is not None:
            menu.addAction(action)

        handle = plot.getWidgetHandle()
        menu.exec_(handle.mapToGlobal(pos))

    def __plotSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)
            self.sigMouseMove.emit(x, y)

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
        self.sigMouseLeave.emit()
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

        polyline = self.__rings[ringId][1]
        color = CalibrationContext.instance().getMarkerColor(ringId, mode="numpy")
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
                color=color,
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

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-geometry.ui"), self)
        icon = icons.getQIcon("pyfai:gui/icons/task-fit-geometry")
        self.setWindowIcon(icon)

        self.initNextStep()
        self.widgetShow.connect(self.__widgetShow)

        self.__plot = self.__createPlot()
        self.__plot.sigMouseMove.connect(self.__mouseMoved)
        self.__plot.sigMouseLeave.connect(self.__mouseLeft)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        userAngleUnit = CalibrationContext.instance().getAngleUnit()
        userLengthUnit = CalibrationContext.instance().getLengthUnit()
        userWavelengthUnit = CalibrationContext.instance().getWavelengthUnit()

        layout = qt.QGridLayout(self._settings)
        self.__wavelength = FitParamView(self, "Wavelength:", units.Unit.METER_WL, userWavelengthUnit)
        self.addParameterToLayout(layout, self.__wavelength)

        layout = qt.QGridLayout(self._geometry)
        self.__distance = FitParamView(self, "Distance:", units.Unit.METER, userLengthUnit)
        self.__poni1 = FitParamView(self, "PONI1:", units.Unit.METER, userLengthUnit)
        self.__poni2 = FitParamView(self, "PONI2:", units.Unit.METER, userLengthUnit)

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
        self.__statusBar = statusBar
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

    def __mouseMoved(self, x, y):
        value = None

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

        if value is None:
            self.__mouseLeft()
            return

        if self.__calibration is None:
            self.__mouseLeft()
            return

        geometry = self.__calibration.getPyfaiGeometry()
        if geometry is not None:
            ax, ay = numpy.array([x]), numpy.array([y])
            chi = geometry.chi(ay, ax)[0]
            tth = geometry.tth(ay, ax)[0]
            self.__statusBar.setValues(x, y, value, chi, tth)
        else:
            self.__statusBar.setValues(x, y, value, None, None)

    def __mouseLeft(self):
        self.__statusBar.clearValues()

    def __createPlotStatusBar(self, plot):
        statusBar = _StatusBar(self)
        statusBar.setSizeGripEnabled(False)
        return statusBar

    def __createCalibration(self):
        image = self.model().experimentSettingsModel().image().value()
        mask = self.model().experimentSettingsModel().mask().value()
        calibrant = self.model().experimentSettingsModel().calibrantModel().calibrant()
        detector = self.model().experimentSettingsModel().detector()
        wavelength = self.model().experimentSettingsModel().wavelength().value()
        if calibrant is None:
            return None
        if detector is None:
            return None
        if wavelength is None:
            return None

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
        if self.__calibration is None:
            return None

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
            if calibration is None:
                return
            calibration.init(peaks, "massif")
            calibration.toGeometryModel(self.model().peakGeometry())
            self.__peaksInvalidated = False

        self.model().fittedGeometry().setFrom(self.model().peakGeometry())

    def __initGeometryLater(self):
        self.__plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(10, self.__initGeometry)

    def __resetGeometryLater(self):
        self.__plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        self._resetButton.setWaiting(True)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__resetGeometry)

    def __fitGeometryLater(self):
        self.__plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)
        self._fitButton.setWaiting(True)
        # Wait for Qt repaint first
        qt.QTimer.singleShot(1, self.__fitGeometry)

    def __unsetProcessing(self):
        self.__plot.unsetProcessing()
        qt.QApplication.restoreOverrideCursor()
        self._resetButton.setWaiting(False)
        self._fitButton.setWaiting(False)

    def __initGeometry(self):
        self.__initGeometryFromPeaks()
        self.__formatResidual()
        self.__unsetProcessing()

    def __resetGeometry(self):
        calibration = self.__getCalibration()
        if calibration is None:
            self.__unsetProcessing()
            return
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
        if calibration is None:
            self.__unsetProcessing()
            self._fitButton.setWaiting(False)
            return
        if self.__peaksInvalidated:
            self.__initGeometryFromPeaks()
        else:
            calibration.fromGeometryModel(self.model().fittedGeometry(), resetResidual=False)
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
        if calibration is None:
            text = ""
        else:
            previousRms = calibration.getPreviousRms()
            rms = calibration.getRms()
            if rms is not None:
                angleUnit = CalibrationContext.instance().getAngleUnit().value()
                rms = units.convert(rms, units.Unit.RADIAN, angleUnit)
                text = '%.6e' % rms
                if previousRms is not None:
                    previousRms = units.convert(previousRms, units.Unit.RADIAN, angleUnit)
                    if numpy.isclose(rms, previousRms):
                        diff = "(no changes)"
                    else:
                        diff = '(%+.2e)' % (rms - previousRms)
                        if rms < previousRms:
                            diff = '<font color="green">%s</font>' % diff
                        else:
                            diff = '<font color="red">%s</font>' % diff
                    text = '%s %s' % (text, diff)
                text = "%s %s" % (text, angleUnit.symbol)
            else:
                text = ""
        self._currentResidual.setText(text)

    def __geometryUpdated(self):
        calibration = self.__getCalibration()
        if calibration is None:
            return
        model = self.model().fittedGeometry()
        if model.isValid():
            resetResidual = self.__fitting is not True
            calibration.fromGeometryModel(model, resetResidual=resetResidual)
            self.__updateDisplay()
            self.__formatResidual()

        geometry = calibration.getPyfaiGeometry()
        self.__plot.markerManager().updatePhysicalMarkerPixels(geometry)

    def __updateDisplay(self):
        calibration = self.__getCalibration()
        if calibration is None:
            return

        rings = calibration.getRings()
        tth = calibration.getTwoThetaArray()
        self.__plot.setRings(rings, tth)

        center = calibration.getBeamCenter()
        if center is None:
            self.__plot.removeMarker(legend="center")
        else:
            color = CalibrationContext.instance().getMarkerColor(0, mode="html")
            self.__plot.addMarker(
                text="Beam",
                y=center[0],
                x=center[1],
                legend="center",
                color=color,
                symbol="+")

        poni = calibration.getPoni()
        if poni is None:
            self.__plot.removeMarker(legend="poni")
        else:
            color = CalibrationContext.instance().getMarkerColor(0, mode="html")
            self.__plot.addMarker(
                text="PONI",
                y=poni[0],
                x=poni[1],
                legend="poni",
                color=color,
                symbol="+")

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
        if image is not None:
            self.__plot.addImage(image, legend="image", copy=False)
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()
        else:
            self.__plot.removeImage("image")

    def __widgetShow(self):
        if self.__peaksInvalidated:
            self.__initGeometryLater()
