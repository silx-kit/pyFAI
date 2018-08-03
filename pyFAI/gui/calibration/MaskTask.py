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
__date__ = "01/08/2018"

import logging
from silx.gui import qt
from silx.gui.colors import Colormap
import silx.gui.plot
from silx.gui.plot.tools import PositionInfo

import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask

_logger = logging.getLogger(__name__)


class MaskTask(AbstractCalibrationTask):

    def __init__(self):
        super(MaskTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-mask.ui"), self)
        self.initNextStep()

        self.__plot = self.__createPlot(self._imageHolder)
        self.__maskPanel = silx.gui.plot.MaskToolsWidget.MaskToolsWidget(parent=self._toolHolder, plot=self.__plot)
        self.__maskPanel.setDirection(qt.QBoxLayout.TopToBottom)
        self.__maskPanel.setMultipleMasks("single")

        layout = qt.QVBoxLayout(self._toolHolder)
        layout.addWidget(self.__maskPanel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._toolHolder.setLayout(layout)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        # FIXME ask for a stable API
        self.__maskPanel._mask.sigChanged.connect(self.__maskFromPlotChanged)
        self.widgetShow.connect(self.__widgetShow)
        self.widgetHide.connect(self.__widgetHide)

        self.__plotMaskChanged = False
        self.__modelMaskChanged = False

    def __createPlot(self, parent):
        plot = silx.gui.plot.PlotWidget(parent=parent)
        plot.setKeepDataAspectRatio(True)
        plot.setDataMargins(0.1, 0.1, 0.1, 0.1)
        plot.setGraphXLabel("Y")
        plot.setGraphYLabel("X")
        plot.setAxesDisplayed(False)

        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)

        statusBar = self.__createPlotStatusBar(plot)
        plot.setStatusBar(statusBar)

        colormap = Colormap("inferno", normalization=Colormap.LOGARITHM)
        plot.setDefaultColormap(colormap)
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
        settings = model.experimentSettingsModel()
        settings.image().changed.connect(self.__imageUpdated)
        settings.mask().changed.connect(self.__maskFromModelChanged)
        self.__maskFromModelChanged()
        self.__imageUpdated()

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        if image is not None:
            self.__plot.addImage(image, legend="image")
            self.__plot.setGraphXLimits(0, image.shape[0])
            self.__plot.setGraphYLimits(0, image.shape[1])
            self.__plot.resetZoom()
        else:
            self.__plot.removeImage("image")

    def __widgetShow(self):
        self.__updateWidgetFromModel()

    def __widgetHide(self):
        self.__updateModelFromWidget()

    def __maskFromPlotChanged(self):
        self.__plotMaskChanged = True

    def __maskFromModelChanged(self):
        self.__modelMaskChanged = True
        if self.isVisible():
            self.__updateWidgetFromModel()

    def __updateWidgetFromModel(self):
        """Update the widget using the mask from the model, only if needed"""
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
