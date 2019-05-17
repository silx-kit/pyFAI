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
__date__ = "17/05/2019"

import logging
import numpy
import datetime

from silx.gui import qt
from silx.gui import icons
import silx.gui.plot

import pyFAI.utils
from pyFAI.utils import stringutil
from .AbstractCalibrationTask import AbstractCalibrationTask
from ..helper.RingCalibration import RingCalibration
from ..helper.SynchronizeRawView import SynchronizeRawView
from ..helper.SynchronizePlotBackground import SynchronizePlotBackground
from ..CalibrationContext import CalibrationContext
from ..widgets.QuantityLabel import QuantityLabel
from ..widgets.FitParamView import FitParamView
from ..model.GeometryConstraintsModel import GeometryConstraintsModel
from ..utils import units
from ..helper.MarkerManager import MarkerManager
from ..helper import ProcessingWidget
from ..helper import model_transform
from ..utils import unitutils
from ... import units as core_units

_logger = logging.getLogger(__name__)


class _StatusBar(qt.QStatusBar):

    def __init__(self, parent=None):
        qt.QStatusBar.__init__(self, parent)

        angleUnitModel = CalibrationContext.instance().getAngleUnit()
        scatteringUnitModel = CalibrationContext.instance().getScatteringVectorUnit()

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

        self.__q = QuantityLabel(self)
        self.__q.setPrefix(u"<b>q</b>: ")
        self.__q.setFormatter(u"{value: >4.3F}")
        self.__q.setInternalUnit(units.Unit.INV_ANGSTROM)
        self.__q.setDisplayedUnitModel(scatteringUnitModel)
        self.__q.setUnitEditable(True)
        self.__q.setElasticSize(True)
        self.addWidget(self.__q)

        self.clearValues()

    def clearValues(self):
        self.setValues(None, None, numpy.nan, numpy.nan, numpy.nan)

    def setValues(self, x, y, pixel, chi, tth):
        if x is None:
            pos = None
        else:
            pos = x, y
        self.__position.setValue(pos)
        self.__pixel.setValue(pixel)
        self.__chi.setValue(chi)
        tth = numpy.nan if tth is None else tth
        self.__2theta.setValue(tth)
        if not numpy.isnan(tth):
            # NOTE: wavelength could be updated, and the the display would not
            # be updated. But here it is safe enougth.
            wavelength = CalibrationContext.instance().getCalibrationModel().fittedGeometry().wavelength().value()
            q = unitutils.from2ThRad(tth, core_units.Q_A, wavelength)
            self.__q.setValue(q)
        else:
            self.__q.setValue(numpy.nan)


class CalibrationState(qt.QObject):
    """Store the state of a calibration"""

    changed = qt.Signal()

    def __init__(self, parent):
        qt.QObject.__init__(self, parent)
        self.reset()

    def reset(self):
        self.__geoRef = None
        self.__geometry = None
        self.__rings = None
        self.__rms = None
        self.__previousRms = None
        self.__tth = None
        self.__poni = None
        self.__beamCenter = None
        self.__empty = True
        self.changed.emit()

    def isEmpty(self):
        return self.__empty

    def getTwoThetaArray(self):
        return self.__tth

    def getRings(self):
        return self.__rings

    def getBeamCenter(self):
        return self.__beamCenter

    def getPoni(self):
        return self.__poni

    def getPreviousRms(self):
        return self.__previousRms

    def getRms(self):
        return self.__rms

    def getGeometryRefinement(self):
        return self.__geoRef

    def popGeometryRefinement(self):
        """Invalidate the object and remove the ownershit of the geometry
        refinment"""
        geoRef = self.__geoRef
        self.reset()
        return geoRef

    def update(self, calibration):
        """Update the state from a current calibration process.

        :param RingCalibration calibration: A calibration process
        """
        self.__geoRef = calibration.getPyfaiGeometry()
        self.__geometry = None
        self.__rings = calibration.getRings()
        self.__previousRms = self.__rms
        self.__rms = calibration.getRms()
        self.__tth = calibration.getTwoThetaArray()
        self.__poni = calibration.getPoni()
        self.__beamCenter = calibration.getBeamCenter()
        self.__empty = False
        self.changed.emit()


class _RingPlot(silx.gui.plot.PlotWidget):

    sigMouseMove = qt.Signal(float, float)

    sigMouseLeave = qt.Signal()

    def __init__(self, parent=None):
        silx.gui.plot.PlotWidget.__init__(self, parent=parent)
        self.getXAxis().sigLimitsChanged.connect(self.__axesChanged)
        self.getYAxis().sigLimitsChanged.connect(self.__axesChanged)
        self.sigPlotSignal.connect(self.__plotSignalReceived)
        self.__axisOfCurrentView = None
        self.__state = None
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

        self.__plotBackground = SynchronizePlotBackground(self)

        widget = self
        if hasattr(self, "centralWidget"):
            widget = widget.centralWidget()
        widget.installEventFilter(self)

    def setCalibrationState(self, state):
        if self.__state is not None:
            self.__state.changed.disconnect(self.__updateDisplay)
        self.__state = state
        if self.__state is not None:
            self.__state.changed.connect(self.__updateDisplay)

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Leave:
            self.__mouseLeave()
            return True

        if event.type() == qt.QEvent.ToolTip:
            if self.__tth is not None:
                pos = widget.mapFromGlobal(event.globalPos())
                coord = widget.pixelToData(pos.x(), pos.y(), axis="left", check=False)

                pos = coord[0], coord[1]
                x, y = self.__clampOnImage(pos)
                angle = self.__tth[y, x]
                ringId, angle = self.__getClosestAngle(angle)

                if ringId is not None:
                    message = "%s ring" % stringutil.to_ordinal(ringId + 1)
                    qt.QToolTip.showText(event.globalPos(), message)
                else:
                    qt.QToolTip.hideText()
                    event.ignore()

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

        # Do not dispaly all rings, but at least the 10 first
        firstRings = [a for a in angles if a[0] <= 10]
        sampledRings = [a for a in angles if (a[0] % step == 0)]
        displayedRings = set(firstRings + sampledRings)

        for ringId, ringAngle in displayedRings:
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
            y, x = line[:, 0] + 0.5, line[:, 1] + 0.5
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

    def __cleanupRings(self):
        for items in self.__ringItems.values():
            for item in items:
                self.removeCurve(item.getLegend())
        self.__ringItems = {}
        self.__tth = None
        self.__rings = []

    def __cleanupMarkers(self):
        try:
            self.removeMarker(legend="center")
        except Exception:
            pass
        try:
            self.removeMarker(legend="poni")
        except Exception:
            pass

    def __updateMarkers(self):
        state = self.__state
        center = state.getBeamCenter()
        if center is None:
            try:
                self.removeMarker(legend="center")
            except Exception:
                pass
        else:
            color = CalibrationContext.instance().getMarkerColor(0, mode="html")
            self.addMarker(
                text="Beam",
                y=center[0],
                x=center[1],
                legend="center",
                color=color,
                symbol="+")

        poni = state.getPoni()
        if poni is None:
            try:
                self.removeMarker(legend="poni")
            except Exception:
                pass
        else:
            color = CalibrationContext.instance().getMarkerColor(0, mode="html")
            self.addMarker(
                text="PONI",
                y=poni[0],
                x=poni[1],
                legend="poni",
                color=color,
                symbol="+")

    def __updateDisplay(self):
        """Update the display when the calibration state was updated."""
        state = self.__state

        self.__cleanupRings()
        self.__cleanupMarkers()
        if state.isEmpty():
            return

        rings = state.getRings()
        tth = state.getTwoThetaArray()
        self.__tth = tth
        self.__rings = rings
        self.__updateRings()
        self.__updateMarkers()

    def unsetProcessing(self):
        if self.__processing is not None:
            self.__processing.deleteLater()

    def setProcessing(self):
        self.__processing = ProcessingWidget.createProcessingWidgetOverlay(self)


class GeometryTask(AbstractCalibrationTask):

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-geometry.ui"), self)
        icon = icons.getQIcon("pyfai:gui/icons/task-fit-geometry")
        self.setWindowIcon(icon)

        self.__calibrationState = CalibrationState(self)
        self.__calibration = None
        self.__peaksInvalidated = False
        self.__fitting = False
        self.__wavelengthInvalidated = False

        self.initNextStep()
        self.widgetShow.connect(self.__widgetShow)

        self.__plot = self.__createPlot()
        self.__plot.setObjectName("plot-fit")
        self.__plot.sigMouseMove.connect(self.__mouseMoved)
        self.__plot.sigMouseLeave.connect(self.__mouseLeft)
        self.__plot.setCalibrationState(self.__calibrationState)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)
        self.__defaultConstraints = GeometryConstraintsModel()

        userAngleUnit = CalibrationContext.instance().getAngleUnit()
        userLengthUnit = CalibrationContext.instance().getLengthUnit()
        userWavelengthUnit = CalibrationContext.instance().getWavelengthUnit()

        layout = qt.QGridLayout(self._settings)
        self.__wavelength = FitParamView(self, "Wavelength", units.Unit.METER_WL, userWavelengthUnit)
        self.addParameterToLayout(layout, self.__wavelength)

        layout = qt.QGridLayout(self._geometry)
        self.__distance = FitParamView(self, "Distance", units.Unit.METER, userLengthUnit)
        self.__poni1 = FitParamView(self, "PONI1", units.Unit.METER, userLengthUnit)
        self.__poni2 = FitParamView(self, "PONI2", units.Unit.METER, userLengthUnit)

        self.__rotation1 = FitParamView(self, "Rotation 1", units.Unit.RADIAN, userAngleUnit)
        self.__rotation2 = FitParamView(self, "Rotation 2", units.Unit.RADIAN, userAngleUnit)
        self.__rotation3 = FitParamView(self, "Rotation 3", units.Unit.RADIAN, userAngleUnit)

        self.__wavelength.sigValueAccepted.connect(self.__geometryCustomed)
        self.__distance.sigValueAccepted.connect(self.__geometryCustomed)
        self.__poni1.sigValueAccepted.connect(self.__geometryCustomed)
        self.__poni2.sigValueAccepted.connect(self.__geometryCustomed)
        self.__rotation1.sigValueAccepted.connect(self.__geometryCustomed)
        self.__rotation2.sigValueAccepted.connect(self.__geometryCustomed)
        self.__rotation3.sigValueAccepted.connect(self.__geometryCustomed)

        self.__distance.setDefaultConstraintsModel(self.__defaultConstraints.distance())
        self.__wavelength.setDefaultConstraintsModel(self.__defaultConstraints.wavelength())
        self.__poni1.setDefaultConstraintsModel(self.__defaultConstraints.poni1())
        self.__poni2.setDefaultConstraintsModel(self.__defaultConstraints.poni2())
        self.__rotation1.setDefaultConstraintsModel(self.__defaultConstraints.rotation1())
        self.__rotation2.setDefaultConstraintsModel(self.__defaultConstraints.rotation2())
        self.__rotation3.setDefaultConstraintsModel(self.__defaultConstraints.rotation3())

        self.addParameterToLayout(layout, self.__distance)
        self.addParameterToLayout(layout, self.__poni1)
        self.addParameterToLayout(layout, self.__poni2)
        self.addParameterToLayout(layout, self.__rotation1)
        self.addParameterToLayout(layout, self.__rotation2)
        self.addParameterToLayout(layout, self.__rotation3)

        self._fitButton.clicked.connect(self.__fitGeometryLater)
        self._fitButton.setDisabledWhenWaiting(True)
        self._resetButton.clicked.connect(self.__resetGeometryLater)

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

        constraintLayout = qt.QHBoxLayout()
        defaultConstraintsButton = qt.QPushButton("Default contraints", self)
        defaultConstraintsButton.setToolTip("Remove all the custom constraints.")
        saxsConstraintsButton = qt.QPushButton("SAXS contraints", self)
        saxsConstraintsButton.setToolTip("Force all the rotations to zero.")
        constraintLayout.addWidget(defaultConstraintsButton)
        constraintLayout.addWidget(saxsConstraintsButton)
        layout.addLayout(constraintLayout, layout.rowCount(), 0, 1, -1)
        defaultConstraintsButton.clicked.connect(self.__setDefaultConstraints)
        saxsConstraintsButton.clicked.connect(self.__setSaxsConstraints)

        self._geometryHistoryCombo.activated.connect(self.__geometryPickedFromHistory)
        self._geometryHistoryCombo.setAngleUnit(userAngleUnit)

        self.__calibrationState.changed.connect(self.__updateResidual)
        self.__updateResidual()

    def __setDefaultConstraints(self):
        """Apply default contraints imposed by the refinment process"""
        calibrationModel = self.model()
        constraintsModel = calibrationModel.geometryConstraintsModel()
        constraintsModel.set(self.__defaultConstraints)

    def __setSaxsConstraints(self):
        """Apply default contraints use by SAXS experiments"""
        calibrationModel = self.model()
        constraintsModel = calibrationModel.geometryConstraintsModel()
        constraintsModel.lockSignals()
        constraintsModel.rotation1().setFixed(True)
        constraintsModel.rotation2().setFixed(True)
        constraintsModel.rotation3().setFixed(True)
        constraintsModel.unlockSignals()
        geometry = calibrationModel.fittedGeometry()
        geometry.lockSignals()
        geometry.rotation1().setValue(0)
        geometry.rotation2().setValue(0)
        geometry.rotation3().setValue(0)
        geometry.unlockSignals()

    def addParameterToLayout(self, layout, param):
        # an empty grid returns 1
        row = layout.rowCount()
        widgets = param.widgets()
        for i, widget in enumerate(widgets):
            if isinstance(widget, qt.QWidget):
                layout.addWidget(widget, row, i)
            else:
                layout.addLayout(widget, row, i)

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

        toolBar = qt.QToolBar(self)
        plot3dAction = qt.QAction(self)
        plot3dAction.setIcon(icons.getQIcon("pyfai:gui/icons/3d"))
        plot3dAction.setText("3D visualization")
        plot3dAction.setToolTip("Display a 3D visualization of the sample stage")
        plot3dAction.triggered.connect(self.__display3dDialog)
        toolBar.addAction(plot3dAction)
        plot.addToolBar(toolBar)

    def __mouseMoved(self, x, y):
        value = None

        image = self.__plot.getImage("image")
        if image is None:
            return
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

        geometry = self.__calibrationState.getGeometryRefinement()
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

    def __invalidateWavelength(self):
        self.__wavelengthInvalidated = True

    def __invalidateCalibration(self):
        self.__calibration = None

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

        peaksModel = self.model().peakSelectionModel()

        if len(peaksModel) == 0:
            return None

        peaks = model_transform.createPeaksArray(peaksModel)
        calibration = RingCalibration(image,
                                      mask,
                                      calibrant,
                                      detector,
                                      wavelength,
                                      peaks=peaks,
                                      method="massif")

        # Copy the default values
        self.__defaultConstraints.set(calibration.defaultGeometryConstraintsModel())
        return calibration

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

    def __initGeometryFromPeaks(self, useFittedGeometry=False):
        geometry = self.model().fittedGeometry()

        if self.__peaksInvalidated:
            # Recompute the geometry from the peaks
            peaksModel = self.model().peakSelectionModel()
            peaks = model_transform.createPeaksArray(peaksModel)
            calibration = self.__getCalibration()
            if calibration is None:
                return

            # Constraints defined by the GUI
            constraints = self.model().geometryConstraintsModel().copy(self)
            constraints.fillDefault(calibration.defaultGeometryConstraintsModel())

            if useFittedGeometry:
                initialGeometry = geometry
            else:
                initialGeometry = None

            calibration.init(peaks, "massif", initialGeometry, constraints)
            calibration.toGeometryModel(self.model().peakGeometry())
            self.__defaultConstraints.set(calibration.defaultGeometryConstraintsModel())
            self.__peaksInvalidated = False

        geometry.setFrom(self.model().peakGeometry())

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

        # Save this geometry into the history
        calibration = self.__getCalibration()
        geometry = self.model().fittedGeometry()
        rms = None
        if calibration is not None and calibration.isValid():
            rms = calibration.getRms()
        geometryHistory = self.model().geometryHistoryModel()
        geometryHistory.appendGeometry("Init", datetime.datetime.now(), geometry, rms)

        self.__unsetProcessing()

    def __resetGeometry(self):
        calibration = self.__getCalibration()
        if calibration is None:
            self.__unsetProcessing()
            return
        self.__initGeometryFromPeaks()
        # write result to the fitted geometry
        geometry = self.model().fittedGeometry()
        calibration.toGeometryModel(geometry)

        # Save this geometry into the history
        geometryHistory = self.model().geometryHistoryModel()
        geometryHistory.appendGeometry("Reset", datetime.datetime.now(), geometry, calibration.getRms())

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
            self.__initGeometryFromPeaks(useFittedGeometry=True)
        else:
            calibration.fromGeometryModel(self.model().fittedGeometry(), resetResidual=False)

        constraints = self.model().geometryConstraintsModel().copy(self)
        constraints.fillDefault(self.__defaultConstraints)
        calibration.fromGeometryConstraintsModel(constraints)

        calibration.refine()
        if calibration.isValid():
            # write result to the fitted model
            geometry = self.model().fittedGeometry()
            calibration.toGeometryModel(geometry)

            # Save this geometry into the history
            geometryHistory = self.model().geometryHistoryModel()
            geometryHistory.appendGeometry("Fitted", datetime.datetime.now(), geometry, calibration.getRms())
        else:
            self.__showDialogCalibrationDiverge()

        self._fitButton.setWaiting(False)
        self.__fitting = False
        self.__unsetProcessing()

    def __updateResidual(self):
        """
        Update the display of the residual.

        Called when the calibration state was updated.
        """
        state = self.__calibrationState
        rms = state.getRms()
        if rms is not None:
            angleUnit = CalibrationContext.instance().getAngleUnit().value()
            rms = units.convert(rms, units.Unit.RADIAN, angleUnit)
            text = stringutil.to_scientific_unicode(rms, digits=4)
            previousRms = state.getPreviousRms()
            if previousRms is not None:
                previousRms = units.convert(previousRms, units.Unit.RADIAN, angleUnit)
                if numpy.isclose(rms, previousRms):
                    diff = "no changes"
                else:
                    diff = stringutil.to_scientific_unicode(rms - previousRms, digits=2)
                    if rms < previousRms:
                        diff = '<font color="green">%s</font>' % diff
                    else:
                        diff = '<font color="red">%s</font>' % diff
                text = '%s (%s)' % (text, diff)
            text = "%s %s" % (text, angleUnit.symbol)
        else:
            text = ""
        self._currentResidual.setText(text)

    def __geometryCustomed(self):
        """
        Called when the geometry is manually customed.
        """
        # Save this geometry into the history
        geometry = self.model().fittedGeometry()
        geometryHistory = self.model().geometryHistoryModel()
        if len(geometryHistory) > 0:
            # Avoid duplication when it is possible
            last = geometryHistory[-1]
            if last.geometry() == geometry:
                return

        calibration = self.__getCalibration()
        calibration.fromGeometryModel(geometry, resetResidual=True)

        state = self.__calibrationState
        state.update(calibration)
        now = datetime.datetime.now()
        geometryHistory.appendGeometry("Customed", now, geometry, state.getRms())

    def __showDialogCalibrationDiverge(self):
        title = "Error while calibrating"
        message = ("It is not possible to calibrate/refine the geometry. " +
                   "The refinement <b>diverge</b>. " +
                   "It may be due to a mistake on specified wavelength, or selected peaks. " +
                   "<b>Check your input data</b>.")
        qt.QMessageBox.critical(self, title, message)

    def __geometryUpdated(self):
        calibration = self.__getCalibration()
        if calibration is None:
            self.__calibrationState.reset()
            return
        if not calibration.isValid():
            self.__showDialogCalibrationDiverge()
            self.__calibrationState.reset()
            return
        geometry = self.model().fittedGeometry()
        if geometry.isValid():
            resetResidual = self.__fitting is not True
            calibration.fromGeometryModel(geometry, resetResidual=resetResidual)
            self.__calibrationState.update(calibration)
        else:
            self.__calibrationState.reset()

        geoRef = calibration.getPyfaiGeometry()
        self.__plot.markerManager().updatePhysicalMarkerPixels(geoRef)

    def __geometryPickedFromHistory(self, index=None):
        item = self._geometryHistoryCombo.currentItem()
        if item is None:
            return

        # Unselect the selection
        old = self._geometryHistoryCombo.blockSignals(True)
        self._geometryHistoryCombo.setCurrentIndex(-1)
        self._geometryHistoryCombo.blockSignals(old)

        # Apply the selected geometry
        calibration = self.__getCalibration()
        if calibration is None:
            return

        pickedGeometry = item.geometry()
        calibration.fromGeometryModel(pickedGeometry, resetResidual=True)
        geometry = self.model().fittedGeometry()
        geometry.setFrom(pickedGeometry)

    def _updateModel(self, model):
        self.__synchronizeRawView.registerModel(model.rawPlotView())
        self._geometryHistoryCombo.setHistoryModel(model.geometryHistoryModel())
        settings = model.experimentSettingsModel()
        settings.maskedImage().changed.connect(self.__imageUpdated)
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

        settings.maskedImage().changed.connect(self.__invalidateCalibration)
        settings.image().changed.connect(self.__invalidateCalibration)
        settings.calibrantModel().changed.connect(self.__invalidateCalibration)
        settings.detectorModel().changed.connect(self.__invalidateCalibration)

        self.__imageUpdated()

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().maskedImage().value()
        if image is not None:
            self.__plot.addImage(image, legend="image", copy=False)
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()
        else:
            self.__plot.removeImage("image")

    def __widgetShow(self):
        if not self.__peaksInvalidated:
            # In case of the very first time
            geometry = self.model().fittedGeometry()
            peakPickingSelection = self.model().peakSelectionModel()
            self.__peaksInvalidated = len(peakPickingSelection) != 0 and not geometry.isValid()

        if self.__peaksInvalidated:
            self.__initGeometryLater()

    def __display3dDialog(self):
        from ..dialog.Detector3dDialog import Detector3dDialog
        dialog = Detector3dDialog(self)

        settings = self.model().experimentSettingsModel()
        detector = settings.detectorModel().detector()
        image = settings.image().value()
        mask = settings.mask().value()
        colormap = CalibrationContext.instance().getRawColormap()
        geometry = None

        fittedGeometry = self.model().fittedGeometry()
        if fittedGeometry.isValid():
            from pyFAI import geometry
            geometry = geometry.Geometry()
            model_transform.geometryModelToGeometry(fittedGeometry, geometry)

        dialog.setData(detector=detector,
                       image=image, mask=mask, colormap=colormap,
                       geometry=geometry)
        dialog.exec_()
