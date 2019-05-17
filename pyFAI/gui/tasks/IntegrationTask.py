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

from silx.gui import qt
import silx.gui.plot
import silx.gui.icons
import silx.io

import pyFAI.utils
from .AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from ..utils import unitutils
from ..model.DataModel import DataModel
from ..widgets.QuantityLabel import QuantityLabel
from ..CalibrationContext import CalibrationContext
from ... import units as core_units
from ..utils import units
from ..utils import validators
from ..helper.MarkerManager import MarkerManager
from ..helper.SynchronizePlotBackground import SynchronizePlotBackground
from ..helper import ProcessingWidget
from pyFAI.ext.invert_geometry import InvertGeometry
from ..utils import FilterBuilder
from ..utils import imageutils
from ...utils import stringutil
from ..dialog.IntegrationMethodDialog import IntegrationMethodDialog
from pyFAI import method_registry
from ..dialog import MessageBox
from pyFAI.io import ponifile


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
        self.__resetZoomPolicy = None
        self.__method = None
        self.__errorMessage = None

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
        self.__nPointsAzimuthal = model.integrationSettingsModel().nPointsAzimuthal().value()
        if self.__nPointsAzimuthal <= 0:
            self.__nPointsAzimuthal = 1
        self.__nPointsRadial = model.integrationSettingsModel().nPointsRadial().value()
        if self.__nPointsRadial <= 0:
            self.__nPointsRadial = 1
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

        self.__wavelength = geometry.wavelength().value()
        self.__distance = geometry.distance().value()
        self.__poni1 = geometry.poni1().value()
        self.__poni2 = geometry.poni2().value()
        self.__rotation1 = geometry.rotation1().value()
        self.__rotation2 = geometry.rotation2().value()
        self.__rotation3 = geometry.rotation3().value()
        return True

    def setDisplayMask(self, displayed):
        self.__displayMask = displayed

    def setMethod(self, method):
        self.__method = method

    def method(self):
        return self.__method

    def setResetZoomPolicy(self, policy):
        self.__resetZoomPolicy = policy

    def resetZoomPolicy(self):
        return self.__resetZoomPolicy

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

        # FIXME Add error model
        method = method_registry.Method(0, self.__method.split, self.__method.algo, self.__method.impl, None)
        method1d = method.fixed(dim=1)
        methods = method_registry.IntegrationMethod.select_method(method=method1d)
        if len(methods) == 0:
            method1d = method_registry.Method(1, method1d.split, "*", "*", None)
            _logger.warning("Downgrade 1D integration method to %s", method1d)
        else:
            method1d = methods[0].method

        method2d = method.fixed(dim=2)
        methods = method_registry.IntegrationMethod.select_method(method=method2d)
        if len(methods) == 0:
            method2d = method_registry.Method(2, method2d.split, "*", "*", None)
            _logger.warning("Downgrade 2D integration method to %s", method2d)
        else:
            method2d = methods[0].method

        try:
            self.__result1d = ai.integrate1d(
                method=method1d,
                data=self.__image,
                npt=self.__nPointsRadial,
                unit=self.__radialUnit,
                mask=self.__mask,
                polarization_factor=self.__polarizationFactor)

            self.__result2d = ai.integrate2d(
                method=method2d,
                data=self.__image,
                npt_rad=self.__nPointsRadial,
                npt_azim=self.__nPointsAzimuthal,
                unit=self.__radialUnit,
                mask=self.__mask,
                polarization_factor=self.__polarizationFactor)

            # Create an image masked where data exists
            self.__resultMask2d = None
            if self.__mask is not None:
                if self.__mask.shape == self.__image.shape:
                    maskData = numpy.ones(shape=self.__image.shape, dtype=numpy.float32)
                    maskData[self.__mask == 0] = float("NaN")

                    if self.__displayMask:
                        self.__resultMask2d = ai.integrate2d(
                            method=method2d,
                            data=maskData,
                            npt_rad=self.__nPointsRadial,
                            npt_azim=self.__nPointsAzimuthal,
                            unit=self.__radialUnit,
                            polarization_factor=self.__polarizationFactor)
                else:
                    _logger.warning("Inconsistency between image and mask sizes. %s != %s", self.__image.shape, self.__mask.shape)
        except Exception as e:
            _logger.debug("Error while integrating", exc_info=True)
            self.__errorMessage = e
            # TODO: Could be nice to  compute anyway other content (directDist...)
            return

        try:
            self.__directDist = ai.getFit2D()["directDist"]
        except Exception:
            # The geometry could not fit this param
            _logger.debug("Backtrace", exc_info=True)
            self.__directDist = None

        if self.__calibrant:

            rings = self.__calibrant.get_2th()
            try:
                rings = unitutils.from2ThRad(rings, self.__radialUnit, self.__wavelength, self.__directDist)
            except ValueError:
                message = "Convertion to unit %s not supported. Ring locations ignored."
                _logger.warning(message, self.__radialUnit)
                self.__errorMessage = message % self.__radialUnit
                rings = []

            # Filter the rings which are not part of the result
            minAngle, maxAngle = self.__result1d.radial[0], self.__result1d.radial[-1]
            rings = [(i, angle) for i, angle in enumerate(rings) if minAngle <= angle <= maxAngle]
        else:
            rings = []

        self.__rings = rings
        self.__ai = ai

    def rings(self):
        """
        Returns the list of displayable rings as a list of tuple id (zero based), angle

        :rtype: List
        """
        return self.__rings

    def errorMessage(self):
        return self.__errorMessage

    def result1d(self):
        return self.__result1d

    def result2d(self):
        return self.__result2d

    def resultMask2d(self):
        return self.__resultMask2d

    def radialUnit(self):
        return self.__radialUnit

    def wavelength(self):
        return self.__wavelength

    def directDist(self):
        return self.__directDist

    def geometry(self):
        """
        :rtype: pyFAI.geometry.Geometry
        """
        return self.__ai


def createSaveDialog(parent, title, poni=False, json=False, csv=False):
    """Util to create create a save dialog"""
    dialog = CalibrationContext.instance().createFileDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    dialog.setAcceptMode(qt.QFileDialog.AcceptSave)

    builder = FilterBuilder.FilterBuilder()
    if poni:
        builder.addFileFormat("PONI files", "poni")
    if json:
        builder.addFileFormat("JSON files", "json")
    if csv:
        builder.addFileFormat("CSV files", "csv")

    dialog.setNameFilters(builder.getFilters())
    return dialog


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

    def setValues(self, x, y, chi, tth):
        if x is None:
            pos = None
        else:
            pos = x, y

        if pos is None:
            self.__position.setVisible(False)
        else:
            self.__position.setVisible(True)
            self.__position.setValue(pos)

        if chi is None:
            self.__chi.setVisible(False)
        else:
            self.__chi.setVisible(True)
            self.__chi.setValue(chi)

        if tth is None:
            self.__2theta.setVisible(False)
            self.__q.setVisible(False)
        else:
            self.__2theta.setVisible(True)
            self.__2theta.setValue(tth)
            # NOTE: warelength could be updated, and the the display would not
            # be updated. But here it is safe enougth.
            wavelength = CalibrationContext.instance().getCalibrationModel().fittedGeometry().wavelength().value()
            q = unitutils.from2ThRad(tth, core_units.Q_A, wavelength)
            self.__q.setVisible(True)
            self.__q.setValue(q)

    def clearValues(self):
        self.__2theta.setValue(float("nan"))
        self.__q.setValue(float("nan"))


class IntegrationPlot(qt.QFrame):

    def __init__(self, parent=None):
        super(IntegrationPlot, self).__init__(parent)

        self.__plot1d, self.__plot2d = self.__createPlots(self)
        self.__statusBar = _StatusBar(self)
        self.__statusBar.setSizeGripEnabled(False)

        layout = qt.QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addWidget(self.__plot2d)
        layout.addWidget(self.__plot1d)
        layout.addWidget(self.__statusBar)
        self.__setResult(None)
        self.__processing1d = None
        self.__processing2d = None
        self.__ringItems = {}
        self.__axisOfCurrentView = None
        self.__angleUnderMouse = None
        self.__availableRings = None
        self.__radialUnit = None
        self.__wavelength = None
        self.__directDist = None
        self.__geometry = None
        self.__inverseGeometry = None

        markerModel = CalibrationContext.instance().getCalibrationModel().markerModel()
        self.__markerManager = MarkerManager(self.__plot2d, markerModel)

        self.__plot2d.getXAxis().sigLimitsChanged.connect(self.__axesChanged)
        self.__plot1d.sigPlotSignal.connect(self.__plot1dSignalReceived)
        self.__plot2d.sigPlotSignal.connect(self.__plot2dSignalReceived)

        self.__plotBackground = SynchronizePlotBackground(self.__plot2d)

        widget = self.__plot1d
        if hasattr(widget, "centralWidget"):
            widget.centralWidget()
        widget.installEventFilter(self)
        widget = self.__plot2d
        if hasattr(widget, "centralWidget"):
            widget.centralWidget()
        widget.installEventFilter(self)

        colormap = CalibrationContext.instance().getRawColormap()
        self.__plot2d.setDefaultColormap(colormap)

        from silx.gui.plot.utils.axis import SyncAxes
        self.__syncAxes = SyncAxes([self.__plot1d.getXAxis(), self.__plot2d.getXAxis()])

    def aboutToClose(self):
        # Avoid double free release problem. See #892
        self.__syncAxes.stop()
        self.__syncAxes = None

    def resetZoom(self):
        self.__plot1d.resetZoom()
        self.__plot2d.resetZoom()

    def hasData(self):
        return self.__result1d is not None

    def eventFilter(self, widget, event):
        if event.type() == qt.QEvent.Leave:
            self.__mouseLeave()
            return True

        if event.type() == qt.QEvent.ToolTip:
            if self.__availableRings is not None:
                pos = widget.mapFromGlobal(event.globalPos())
                coord = widget.pixelToData(pos.x(), pos.y())

                angle = coord[0]
                ringId, angle = self.__getClosestAngle(angle)

                if ringId is not None:
                    message = "%s ring" % stringutil.to_ordinal(ringId + 1)
                    qt.QToolTip.showText(event.globalPos(), message)
                else:
                    qt.QToolTip.hideText()
                    event.ignore()

                return True

        return False

    def __mouseLeave(self):
        self.__statusBar.clearValues()

        if self.__angleUnderMouse is None:
            return
        if self.__angleUnderMouse not in self.__displayedAngles:
            items = self.__ringItems.get(self.__angleUnderMouse, [])
            for item in items:
                item.setVisible(False)
        self.__angleUnderMouse = None

    def __plot1dSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)
            self.__updateStatusBar(x, None)

    def __plot2dSignalReceived(self, event):
        """Called when old style signals at emmited from the plot."""
        if event["event"] == "mouseMoved":
            x, y = event["x"], event["y"]
            self.__mouseMoved(x, y)
            self.__updateStatusBar(x, y)

    def __getClosestAngle(self, angle):
        """
        Returns the closest ring index and ring angle
        """
        # TODO: Could be done in log(n) using bisect search
        result = None
        iresult = None
        minDistance = float("inf")
        for ringId, ringAngle in self.__availableRings:
            distance = abs(angle - ringAngle)
            if distance < minDistance:
                minDistance = distance
                result = ringAngle
                iresult = ringId
        return iresult, result

    def dataToChiTth(self, data):
        """Returns chi and 2theta angles in radian from data coordinate"""
        try:
            tthRad = unitutils.tthToRad(data[0],
                                        unit=self.__radialUnit,
                                        wavelength=self.__wavelength,
                                        directDist=self.__directDist)
        except Exception:
            _logger.debug("Backtrace", exc_info=True)
            tthRad = None

        chiDeg = data[1]
        if chiDeg is not None:
            chiRad = numpy.deg2rad(chiDeg)
        else:
            chiRad = None

        return chiRad, tthRad

    def __updateStatusBar(self, x, y):
        chiRad, tthRad = self.dataToChiTth((x, y))

        if y is not None and self.__inverseGeometry is not None:
            pixelY, pixelX = self.__inverseGeometry(x, y, True)
            ax, ay = numpy.array([pixelX]), numpy.array([pixelY])
            tthFromPixel = self.__geometry.tth(ay, ax)[0]
            chiFromPixel = self.__geometry.chi(ay, ax)[0]

            if tthRad is not None:
                error = numpy.sqrt((tthRad - tthFromPixel) ** 2 + (chiRad - chiFromPixel) ** 2)
                if error > 0.05:
                    # The identified pixel is far from the requested chi/tth. Marker ignored.
                    pixelY, pixelX = None, None
        else:
            pixelY, pixelX = None, None

        self.__statusBar.setValues(pixelX, pixelY, chiRad, tthRad)

    def __mouseMoved(self, x, y):
        """Called when mouse move over the plot."""
        if self.__availableRings is None:
            return
        angle = x
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

    def __axesChanged(self, minValue, maxValue):
        axisOfCurrentView = self.__plot2d.getXAxis().getLimits()
        if self.__axisOfCurrentView == axisOfCurrentView:
            return
        self.__updateRings()

    def __getAvailableAngles(self, minTth, maxTth):
        result = []
        for ringId, angle in self.__availableRings:
            if minTth is None or maxTth is None:
                result.append(ringId, angle)
            if minTth <= angle <= maxTth:
                result.append((ringId, angle))
        return result

    def __updateRings(self):
        if self.__availableRings is None:
            return

        minTth, maxTth = self.__plot2d.getXAxis().getLimits()
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

        color = CalibrationContext.instance().getMarkerColor(ringId, mode="numpy")
        items = []

        legend = "ring-%i" % (ringId,)

        self.__plot1d.addXMarker(x=ringAngle, color=color, legend=legend)
        item = self.__plot1d._getMarker(legend)
        items.append(item)

        self.__plot2d.addXMarker(x=ringAngle, color=color, legend=legend)
        item = self.__plot2d._getMarker(legend)
        items.append(item)

        self.__ringItems[ringAngle] = items
        return items

    def __syncModeToPlot1d(self, _event):
        modeDict = self.__plot2d.getInteractiveMode()
        mode = modeDict["mode"]
        self.__plot1d.setInteractiveMode(mode)

    def getDefaultColormap(self):
        return self.__plot2d.getDefaultColormap()

    def __createPlots(self, parent):
        margin = 0.02
        plot1d = silx.gui.plot.PlotWidget(parent)
        plot1d.setGraphXLabel("Radial")
        plot1d.setGraphYLabel("Intensity")
        plot1d.setGraphGrid(False)
        plot1d.setDataMargins(margin, margin, margin, margin)
        plot2d = silx.gui.plot.PlotWidget(parent)
        plot2d.setGraphXLabel("Radial")
        plot2d.setGraphYLabel("Azimuthal")
        plot2d.sigInteractiveModeChanged.connect(self.__syncModeToPlot1d)
        plot2d.setDataMargins(margin, margin, margin, margin)

        handle = plot2d.getWidgetHandle()
        handle.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        handle.customContextMenuRequested.connect(self.__plot2dContextMenu)

        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot2d)
        plot2d.addToolBar(toolBar)

        toolBar = tools.ImageToolBar(parent=self, plot=plot2d)
        colormapDialog = CalibrationContext.instance().getColormapDialog()
        toolBar.getColormapAction().setColorDialog(colormapDialog)
        previousResetZoomAction = toolBar.getResetZoomAction()
        resetZoomAction = qt.QAction(toolBar)
        resetZoomAction.triggered.connect(self.resetZoom)
        resetZoomAction.setIcon(previousResetZoomAction.icon())
        resetZoomAction.setText(previousResetZoomAction.text())
        resetZoomAction.setToolTip(previousResetZoomAction.toolTip())
        toolBar.insertAction(previousResetZoomAction, resetZoomAction)
        previousResetZoomAction.setVisible(False)
        self.__resetZoomAction = resetZoomAction
        plot2d.addToolBar(toolBar)

        ownToolBar = qt.QToolBar(plot2d)
        from silx.gui.plot import actions
        logAction = actions.control.YAxisLogarithmicAction(parent=ownToolBar, plot=plot1d)
        logAction.setToolTip("Logarithmic y-axis intensity when checked")
        ownToolBar.addAction(logAction)
        plot2d.addToolBar(ownToolBar)

        action = qt.QAction(ownToolBar)
        action.setIcon(silx.gui.icons.getQIcon("document-save"))
        action.triggered.connect(self.__saveAsCsv)
        action.setToolTip("Save 1D integration as CSV file")
        self.__saveResult1dAction = action
        ownToolBar.addAction(action)

        return plot1d, plot2d

    def __plot2dContextMenu(self, pos):
        from silx.gui.plot.actions.control import ZoomBackAction
        zoomBackAction = ZoomBackAction(plot=self.__plot2d, parent=self.__plot2d)

        menu = qt.QMenu(self)

        menu.addAction(zoomBackAction)
        menu.addSeparator()
        menu.addAction(self.__markerManager.createMarkPixelAction(menu, pos))
        menu.addAction(self.__markerManager.createMarkGeometryAction(menu, pos))
        action = self.__markerManager.createRemoveClosestMaskerAction(menu, pos)
        if action is not None:
            menu.addAction(action)

        handle = self.__plot2d.getWidgetHandle()
        menu.exec_(handle.mapToGlobal(pos))

    def __clearRings(self):
        """Remove of ring item cached on the plots"""
        for items in self.__ringItems.values():
            for item in items:
                self.__plot1d.removeMarker(item.getLegend())
                self.__plot2d.removeMarker(item.getLegend())
        self.__ringItems = {}
        self.__availableRings = []

    def clear(self):
        self.__clearRings()
        try:
            self.__plot1d.remove("result1d", "histogram")
        except Exception:
            pass
        try:
            self.__plot2d.removeImage("integrated_mask")
        except Exception:
            pass
        try:
            self.__plot2d.removeImage("integrated_data")
        except Exception:
            pass

    def setIntegrationProcess(self, integrationProcess):
        """
        :param :class:`~pyFAI.gui.tasks.IntegrationTask.IntegrationProcess` integrationProcess:
            Result of the integration process
        """
        self.__clearRings()

        self.__availableRings = integrationProcess.rings()
        self.__updateRings()

        # FIXME set axes units
        result1d = integrationProcess.result1d()
        self.__plot1d.addHistogram(
            legend="result1d",
            align="center",
            edges=result1d.radial,
            color="blue",
            histogram=result1d.intensity,
            resetzoom=False)

        self.__setResult(result1d)

        def compute_location(result):
            # Assume that axes are linear
            if result.intensity.shape[1] > 1:
                scaleX = (result.radial[-1] - result.radial[0]) / (result.intensity.shape[1] - 1)
            else:
                scaleX = 1.0
            if result.intensity.shape[0] > 1:
                scaleY = (result.azimuthal[-1] - result.azimuthal[0]) / (result.intensity.shape[0] - 1)
            else:
                scaleY = 1.0
            halfPixel = 0.5 * scaleX, 0.5 * scaleY
            origin = (result.radial[0] - halfPixel[0], result.azimuthal[0] - halfPixel[1])
            return origin, scaleX, scaleY

        resultMask2d = integrationProcess.resultMask2d()
        isMaskDisplayed = resultMask2d is not None

        result2d = integrationProcess.result2d()
        result2d_intensity = result2d.intensity

        # Mask pixels with no data
        result2d_intensity[result2d.count == 0] = float("NaN")

        if isMaskDisplayed:
            maskedColor = CalibrationContext.instance().getMaskedColor()
            transparent = (0.0, 0.0, 0.0, 0.0)
            resultMask2d_rgba = imageutils.maskArrayToRgba(resultMask2d.count != 0,
                                                           falseColor=transparent,
                                                           trueColor=maskedColor)
            origin, scaleX, scaleY = compute_location(resultMask2d)
            self.__plot2d.addImage(
                legend="integrated_mask",
                data=resultMask2d_rgba,
                origin=origin,
                scale=(scaleX, scaleY),
                resetzoom=False)
        else:
            try:
                self.__plot2d.removeImage("integrated_mask")
            except Exception:
                pass

        colormap = self.getDefaultColormap()
        origin, scaleX, scaleY = compute_location(result2d)
        self.__plot2d.addImage(
            legend="integrated_data",
            data=result2d_intensity,
            origin=origin,
            scale=(scaleX, scaleY),
            colormap=colormap,
            resetzoom=False)

        self.__radialUnit = integrationProcess.radialUnit()
        self.__wavelength = integrationProcess.wavelength()
        self.__directDist = integrationProcess.directDist()
        self.__geometry = integrationProcess.geometry()
        self.__inverseGeometry = InvertGeometry(
            self.__geometry.array_from_unit(typ="center", unit=self.__radialUnit, scale=True),
            numpy.rad2deg(self.__geometry.chiArray()))

        self.__markerManager.updateProjection(self.__geometry,
                                              self.__radialUnit,
                                              self.__wavelength,
                                              self.__directDist)

        resetZoomPolicy = integrationProcess.resetZoomPolicy()
        if resetZoomPolicy is None:
            # Default behaviour
            self.resetZoom()
        elif resetZoomPolicy is False:
            pass
        else:
            raise ValueError("Reset zoom policy not implemented")

    def __setResult(self, result1d):
        self.__result1d = result1d
        self.__saveResult1dAction.setEnabled(result1d is not None)

    def __saveAsCsv(self):
        if self.__result1d is None:
            return
        dialog = createSaveDialog(self, "Save 1D integration as CSV file", csv=True)
        result = dialog.exec_()
        if not result:
            return
        filename = dialog.selectedFiles()[0]
        # TODO: it would be good to store the units
        silx.io.save1D(filename,
                       x=self.__result1d.radial,
                       y=self.__result1d.intensity,
                       xlabel="radial",
                       ylabels=["intensity"],
                       filetype="csv",
                       autoheader=True)

    def setProcessing(self):
        self.__setResult(None)
        self.__processing1d = ProcessingWidget.createProcessingWidgetOverlay(self.__plot1d)
        self.__processing2d = ProcessingWidget.createProcessingWidgetOverlay(self.__plot2d)

    def unsetProcessing(self):
        if self.__processing1d is not None:
            self.__processing1d.deleteLater()
            self.__processing1d = None
        if self.__processing2d is not None:
            self.__processing2d.deleteLater()
            self.__processing2d = None


class IntegrationTask(AbstractCalibrationTask):

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-result.ui"), self)
        icon = silx.gui.icons.getQIcon("pyfai:gui/icons/task-cake")
        self.setWindowIcon(icon)

        self.initNextStep()

        self._methodLabel.setLabelTemplate("{split}")
        self._warning.setVisible(False)

        self.__integrationUpToDate = True
        self.__integrationResetZoomPolicy = None
        method = method_registry.Method(666, "bbox", "csr", "cython", None)
        self.__setMethod(method)

        positiveValidator = validators.IntegerAndEmptyValidator(self)
        positiveValidator.setBottom(1)

        self._radialPoints.setValidator(positiveValidator)
        self._azimuthalPoints.setValidator(positiveValidator)
        self._radialUnit.setUnits(pyFAI.units.RADIAL_UNITS.values())
        self.__polarizationModel = None
        self._polarizationFactorCheck.clicked[bool].connect(self.__polarizationFactorChecked)
        self.widgetShow.connect(self.__widgetShow)
        self._displayMask.clicked[bool].connect(self.__displayMaskChecked)

        self._integrateButton.beforeExecuting.connect(self.__integrate)
        self._integrateButton.setDisabledWhenWaiting(True)
        self._integrateButton.finished.connect(self.__integratingFinished)

        self._customMethodButton.clicked.connect(self.__customIntegrationMethod)

        self._savePoniButton.clicked.connect(self.__saveAsPoni)

    def __customIntegrationMethod(self):
        dialog = IntegrationMethodDialog(self)
        dialog.selectMethod(self.__method)
        result = dialog.exec_()
        if result:
            method = dialog.selectedMethod()
            self.__setMethod(method)

    def __setMethod(self, method):
        self.__method = method
        self._methodLabel.setMethod(method)
        self.__invalidateIntegrationNoReset()

    def aboutToClose(self):
        self._plot.aboutToClose()

    def __polarizationFactorChecked(self, checked):
        self.__polarizationModel.setEnabled(checked)
        self._polarizationFactor.setEnabled(checked)

    def __polarizationModelChanged(self):
        old = self._polarizationFactorCheck.blockSignals(True)
        isEnabled = self.__polarizationModel.isEnabled()
        self._polarizationFactorCheck.setChecked(isEnabled)
        self._polarizationFactor.setEnabled(isEnabled)
        self._polarizationFactorCheck.blockSignals(old)

    def __displayMaskChecked(self):
        self.__invalidateIntegrationNoReset()

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

    def __invalidateIntegrationNoReset(self):
        if self._plot.hasData():
            self.__integrationResetZoomPolicy = False
            self.__invalidateIntegration()
        else:
            # If there is not yet data, it is not needed to constrain the
            # current range
            pass

    def __widgetShow(self):
        if not self.__integrationUpToDate:
            self._integrateButton.executeCallable()

    def __integrate(self):
        self.__integrationProcess = IntegrationProcess(self.model())
        self.__integrationProcess.setMethod(self.__method)

        if self.__integrationResetZoomPolicy is not None:
            self.__integrationProcess.setResetZoomPolicy(self.__integrationResetZoomPolicy)
            self.__integrationResetZoomPolicy = None

        self.__integrationProcess.setDisplayMask(self._displayMask.isChecked())

        if not self.__integrationProcess.isValid():
            self.__integrationProcess = None
            return
        self.__updateGUIWhileIntegrating()
        self._integrateButton.setCallable(self.__integrationProcess.run)
        self.__integrationUpToDate = True

    def __integratingFinished(self):
        self._plot.unsetProcessing()
        qt.QApplication.restoreOverrideCursor()

        self.__updateGUIWithIntegrationResult(self.__integrationProcess)
        self.__integrationProcess = None
        if not self.__integrationUpToDate:
            # Maybe it was invalidated while priocessing
            self._integrateButton.executeCallable()

    def __updateGUIWhileIntegrating(self):
        self._plot.setProcessing()
        qt.QApplication.setOverrideCursor(qt.Qt.WaitCursor)

    def __updateGUIWithIntegrationResult(self, integrationProcess):
        error = integrationProcess.errorMessage()
        if isinstance(error, Exception):
            self._plot.clear()
            MessageBox.exception(self, "Internal error while integrating", error, None)
            self._warning.setVisible(False)
            return

        if error is not None:
            self._warning.setText(error)
            self._warning.setVisible(True)
        else:
            self._warning.setVisible(False)
        self._plot.setIntegrationProcess(integrationProcess)

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
        self._radialPoints.setModel(integrationSettings.nPointsRadial())
        self._azimuthalPoints.setModel(integrationSettings.nPointsAzimuthal())
        # connect model
        self.__polarizationModel.changed.connect(self.__polarizationModelChanged)
        experimentSettings.detectorModel().changed.connect(self.__invalidateIntegrationNoReset)
        experimentSettings.mask().changed.connect(self.__invalidateIntegrationNoReset)
        experimentSettings.polarizationFactor().changed.connect(self.__invalidateIntegrationNoReset)
        model.fittedGeometry().changed.connect(self.__invalidateIntegration)
        model.fittedGeometry().changed.connect(self.__fittedGeometryChanged)
        integrationSettings.radialUnit().changed.connect(self.__invalidateIntegration)
        integrationSettings.nPointsRadial().changed.connect(self.__invalidateIntegrationNoReset)
        integrationSettings.nPointsAzimuthal().changed.connect(self.__invalidateIntegrationNoReset)

    def __fittedGeometryChanged(self):
        # File have to be saved again
        poniFile = self.model().experimentSettingsModel().poniFile()
        with poniFile.lockContext():
            poniFile.setSynchronized(False)

    def __saveAsPoni(self):
        # FIXME test the validity of the geometry before opening the dialog
        dialog = createSaveDialog(self, "Save as PONI file", poni=True)
        # Disable the warning as the data is append to the file
        dialog.setOption(qt.QFileDialog.DontConfirmOverwrite, True)
        poniFile = self.model().experimentSettingsModel().poniFile()
        previousPoniFile = poniFile.value()
        if previousPoniFile is not None:
            dialog.selectFile(previousPoniFile)

        result = dialog.exec_()
        if not result:
            return
        filename = dialog.selectedFiles()[0]
        nameFilter = dialog.selectedNameFilter()
        isPoniFilter = ".poni" in nameFilter
        if isPoniFilter and not filename.endswith(".poni"):
            filename = filename + ".poni"
        with poniFile.lockContext():
            poniFile.setValue(filename)

        pyfaiGeometry = pyFAI.geometry.Geometry()

        geometry = self.model().fittedGeometry()
        pyfaiGeometry.dist = geometry.distance().value()
        pyfaiGeometry.poni1 = geometry.poni1().value()
        pyfaiGeometry.poni2 = geometry.poni2().value()
        pyfaiGeometry.rot1 = geometry.rotation1().value()
        pyfaiGeometry.rot2 = geometry.rotation2().value()
        pyfaiGeometry.rot3 = geometry.rotation3().value()
        pyfaiGeometry.wavelength = geometry.wavelength().value()

        experimentSettingsModel = self.model().experimentSettingsModel()
        detector = experimentSettingsModel.detector()
        pyfaiGeometry.detector = detector

        try:
            writer = ponifile.PoniFile(pyfaiGeometry)
            with open(filename, "wt") as fd:
                writer.write(fd)
            with poniFile.lockContext():
                poniFile.setValue(filename)
                poniFile.setSynchronized(True)
        except Exception as e:
            MessageBox.exception(self, "Error while saving poni file", e, _logger)
