# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2021 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "13/05/2025"

import logging
import os.path

from silx.gui import qt
import silx.gui.plot
import silx.gui.icons
from silx.gui.plot.tools import PositionInfo

import pyFAI.utils
from .AbstractCalibrationTask import AbstractCalibrationTask
from ..CalibrationContext import CalibrationContext
from ..helper.SynchronizeRawView import SynchronizeRawView
from ..helper.MarkerManager import MarkerManager
from ..helper.SynchronizeMaskToolColor import SynchronizeMaskToolColor
from ..helper.SynchronizePlotBackground import SynchronizePlotBackground

_logger = logging.getLogger(__name__)


class _MaskToolsWidget(silx.gui.plot.MaskToolsWidget.MaskToolsWidget):
    """Inherite the silx mask to be able to save and restore internally
    imported/exported masks to the application model."""

    sigUserMaskChanged = qt.Signal()
    """Emitted when the user changes the mask.

    sigMaskChanged on silx 0.9 to not provide that. This signal is used with a
    filter.
    """

    def __init__(self, parent=None, plot=None):
        silx.gui.plot.MaskToolsWidget.MaskToolsWidget.__init__(self, parent=parent, plot=plot)
        self.__syncColor = SynchronizeMaskToolColor(self)
        self.sigMaskChanged.connect(self.__emitUserMaskChanged)

    def __emitUserMaskChanged(self):
        self.sigUserMaskChanged.emit()

    def __extractDirectory(self, filename):
        if filename is not None and filename != "":
            if os.path.exists(filename):
                if os.path.isdir(filename):
                    return filename
                else:
                    return os.path.dirname(filename)
        return None

    @property
    def maskFileDir(self):
        """The directory from which to load/save mask from/to files."""
        model = CalibrationContext.instance().getCalibrationModel()
        experimentSettings = model.experimentSettingsModel()

        # Reach from the previous mask
        previousFile = experimentSettings.mask().filename()
        directory = self.__extractDirectory(previousFile)
        if directory is None:
            previousFile = experimentSettings.image().filename()
            directory = self.__extractDirectory(previousFile)
        if directory is None:
            directory = os.getcwd()
        return directory

    @maskFileDir.setter
    def maskFileDir(self, maskFileDir):
        # We dont need to store it
        pass

    def save(self, filename, kind):
        try:
            result = silx.gui.plot.MaskToolsWidget.MaskToolsWidget.save(self, filename, kind)
            self.__maskFilenameUpdated(filename)
        finally:
            pass
        return result

    def load(self, filename):
        """Override the fuction importing a new mask."""
        try:
            result = silx.gui.plot.MaskToolsWidget.MaskToolsWidget.load(self, filename)
            self.__maskFilenameUpdated(filename)
            self.__emitUserMaskChanged()
        finally:
            pass
        return result

    def __maskFilenameUpdated(self, filename):
        model = CalibrationContext.instance().getCalibrationModel()
        experimentSettings = model.experimentSettingsModel()
        with experimentSettings.mask().lockContext() as mask:
            mask.setFilename(filename)
            mask.setSynchronized(True)

    def setSelectionMask(self, mask, copy=True):
        self.sigMaskChanged.disconnect(self.__emitUserMaskChanged)
        result = super(_MaskToolsWidget, self).setSelectionMask(mask, copy=copy)
        self.sigMaskChanged.connect(self.__emitUserMaskChanged)
        return result

    def showEvent(self, event):
        self.sigMaskChanged.disconnect(self.__emitUserMaskChanged)
        result = silx.gui.plot.MaskToolsWidget.MaskToolsWidget.showEvent(self, event)
        self.sigMaskChanged.connect(self.__emitUserMaskChanged)
        return result


class MaskTask(AbstractCalibrationTask):

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-mask.ui"), self)
        icon = silx.gui.icons.getQIcon("pyfai:gui/icons/task-mask")
        self.setWindowIcon(icon)

        self.initNextStep()

        self.__plot = None
        self.__plot = self.__createPlot(self._imageHolder)
        self.__plot.setObjectName("plot-mask")

        self.__plotBackground = SynchronizePlotBackground(self.__plot)

        markerModel = CalibrationContext.instance().getCalibrationModel().markerModel()
        self.__markerManager = MarkerManager(self.__plot, markerModel, pixelBasedPlot=True)

        handle = self.__plot.getWidgetHandle()
        handle.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        handle.customContextMenuRequested.connect(self.__plotContextMenu)

        self.__maskPanel = _MaskToolsWidget(parent=self._toolHolder, plot=self.__plot)
        self.__maskPanel.setDirection(qt.QBoxLayout.TopToBottom)
        self.__maskPanel.setMultipleMasks("single")
        layout = self.__maskPanel.layout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout = qt.QVBoxLayout(self._toolHolder)
        layout.addWidget(self.__maskPanel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._toolHolder.setLayout(layout)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        self.__maskPanel.sigUserMaskChanged.connect(self.__maskFromPlotChanged)
        self.widgetShow.connect(self.__widgetShow)
        self.widgetHide.connect(self.__widgetHide)

        self.__plotMaskChanged = False
        self.__modelMaskChanged = False

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

        super()._initGui()

    def __plotContextMenu(self, pos):
        plot = self.__plot
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

    def __createPlot(self, parent):
        plot = silx.gui.plot.PlotWidget(parent=parent)
        plot.setKeepDataAspectRatio(True)
        plot.setDataMargins(0.1, 0.1, 0.1, 0.1)
        plot.setGraphXLabel("Y")
        plot.setGraphYLabel("X")
        plot.setAxesDisplayed(False)

        colormap = CalibrationContext.instance().getRawColormap()
        plot.setDefaultColormap(colormap)

        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        colormapDialog = CalibrationContext.instance().getColormapDialog()
        toolBar.getColormapAction().setColormapDialog(colormapDialog)
        plot.addToolBar(toolBar)

        statusBar = self.__createPlotStatusBar(plot)
        plot.setStatusBar(statusBar)
        return plot

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

    def __getImageValue(self, x, y):
        """Get value of top most image at position (x, y).

        :param float x: X position in plot coordinates
        :param float y: Y position in plot coordinates
        :return: The value at that point or 'n/a'
        """
        value = 'n/a'

        if self.__plot is None:
            # It could happen at the creation of the plot
            # the creation of PositionInfo
            return value
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
        settings.mask().changed.connect(self.__maskFromModelChanged)
        self.__maskFromModelChanged()
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
        # Really make sure to be synchronized
        # We can't trust events from libs
        self.__modelMaskChanged = True
        self.__updateWidgetFromModel()

    def __widgetHide(self):
        self.__updateModelFromWidget()

    def __maskFromPlotChanged(self):
        _logger.debug("MaskTask.__maskFromPlotChanged")
        self.__plotMaskChanged = True

    def __maskFromModelChanged(self):
        _logger.debug("MaskTask.__maskFromModelChanged")
        self.__modelMaskChanged = True
        if self.isVisible():
            self.__updateWidgetFromModel()

    def __updateWidgetFromModel(self):
        """Update the widget using the mask from the model, only if needed"""
        _logger.debug("MaskTask.updateWidgetFromModel")
        if not self.__modelMaskChanged:
            return

        mask = self.model().experimentSettingsModel().mask().value()
        # FIXME if mask is none, the mask should be cleaned up
        if mask is not None:
            self.__maskPanel.setSelectionMask(mask)
        # Everything is synchronized now
        self.__plotMaskChanged = False
        self.__modelMaskChanged = False

    def __updateModelFromWidget(self):
        """Update the model using the mask stored on the widget, only if needed"""
        _logger.debug("MaskTask.__updateModelFromWidget")
        if not self.__plotMaskChanged:
            return

        mask = self.__maskPanel.getSelectionMask()
        maskModel = self.model().experimentSettingsModel().mask()
        maskModel.changed.disconnect(self.__maskFromModelChanged)
        maskModel.setValue(mask)
        maskModel.changed.connect(self.__maskFromModelChanged)
        # Everything is synchronized now
        self.__plotMaskChanged = False
        self.__modelMaskChanged = False
