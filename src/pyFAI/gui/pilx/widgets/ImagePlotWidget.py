#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Loïc Huder (loic.huder@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Tool to visualize diffraction maps."""
from __future__ import annotations

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/04/2024"
__status__ = "development"


from silx.gui import qt
from silx.gui.plot import PlotWidget
from silx.gui.plot.actions.control import (
    ColormapAction,
    KeepAspectRatioAction,
    ResetZoomAction,
)
from silx.gui.plot.actions.io import SaveAction
from silx.gui.plot.ColorBar import ColorBarWidget
from silx.gui.plot.tools import PositionInfo


class ImageToolbar(qt.QToolBar):
    def __init__(self, plot):
        super().__init__(plot)
        self.addAction(ResetZoomAction(plot, self))
        self.addAction(ColormapAction(plot, self))
        self.addAction(KeepAspectRatioAction(plot, self))


class ImagePlotWidget(PlotWidget):
    plotClicked = qt.Signal(float, float)
    pinContextEntrySelected = qt.Signal(float, float)
    setBackgroundClicked = qt.Signal(float, float)

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self._initColorbar()
        self._initStatusBar()
        self._initCentralWidget()
        self._toolbar = self._initToolbar()
        self._toolbar.addSeparator()
        self._toolbar.addAction(SaveAction(self, self._toolbar))
        self.addToolBar(self._toolbar)
        self.sigPlotSignal.connect(self.emitMouseClickSignal)

    def _initToolbar(self):
        return ImageToolbar(self)

    def _initColorbar(self):
        self._colorBarWidget = ColorBarWidget(
            plot=self,
            parent=self,
        )
        # Make ColorBarWidget background white by changing its palette
        self._colorBarWidget.setAutoFillBackground(True)
        palette = self._colorBarWidget.palette()
        palette.setColor(qt.QPalette.Window, qt.Qt.GlobalColor.white)
        self._colorBarWidget.setPalette(palette)

    def _initCentralWidget(self):
        gridLayout = qt.QGridLayout()
        gridLayout.setSpacing(0)
        gridLayout.setContentsMargins(0, 0, 0, 0)
        gridLayout.addWidget(self.getWidgetHandle(), 0, 0)
        gridLayout.addWidget(self._colorBarWidget, 0, 1)
        gridLayout.addWidget(self._statusBar, 1, 0, 1, -1)

        gridLayout.setRowStretch(0, 1)
        gridLayout.setColumnStretch(0, 1)
        centralWidget = qt.QWidget(self)
        centralWidget.setLayout(gridLayout)
        self.setCentralWidget(centralWidget)

    def _initStatusBar(self):
        converters = (
            ("X", lambda x, y: x),
            ("Y", lambda x, y: y),
            ("Data", self._dataConverter),
        )
        self._statusBar = PositionInfo(plot=self, converters=converters)

    def _dataConverter(self, x, y):
        raise NotImplementedError()

    def getColorBarWidget(self):
        """Public method needed for ColorBarAction"""
        return self._colorBarWidget

    def emitMouseClickSignal(self, signal_data):
        if signal_data["event"] != "mouseClicked":
            return

        self.plotClicked.emit(signal_data["x"], signal_data["y"])

    def emitMouseDoubleClickSignal(self, signal_data):
        if signal_data["event"] != "mouseDoubleClicked":
            return

        self.plotDoubleClicked.emit(signal_data["x"], signal_data["y"])
