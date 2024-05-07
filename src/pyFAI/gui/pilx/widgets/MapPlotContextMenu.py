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

__authors__ = ["Loïc Huder", "E. Gutierrez-Fernandez"]
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/04/2024"
__status__ = "development"

from silx.gui import qt
from silx.gui.plot.actions.control import ZoomBackAction
from silx.gui.plot.actions import PlotAction
from silx.gui import icons

class MapPlotContextMenu(qt.QMenu):
    def __init__(self, plot=None, pos=None):
        super().__init__(parent=plot)
        self._plot = plot
        self._pos = pos
        self._build()

    def _build(self):
        action_zoom_back = ZoomBackAction(
            plot=self._plot,
            parent=self._plot,
        )
        action_multi_selection = MultiCurveAction(
            plot=self._plot,
            parent=self,
        )
        action_set_background_curve = SetBackgroundCurveAction(
            plot=self._plot,
            parent=self,
        )
        self.addAction(action_zoom_back)
        self.addSeparator()
        self.addAction(action_multi_selection)
        self.addAction(action_set_background_curve)
        self.addSeparator()

    def _exec(self, pos):
        self._pos = pos
        plotArea = self._plot.getWidgetHandle()
        globalPosition = plotArea.mapToGlobal(pos)
        self.exec(globalPosition)


class MultiCurveAction(PlotAction):
    def __init__(self, plot, parent=None):
        super().__init__(
            plot=plot,
            icon=icons.getQIcon("stats-whole-items"),
            text="Show/Hide curve on graph",
            parent=parent,
            triggered=self._actionTriggered,
            checkable=False,
        )
        self._parent = parent

    def _actionTriggered(self):
        plotArea = self.plot.getWidgetHandle()
        globalPosition = plotArea.mapToParent(self._parent._pos)
        x, y = self.plot.pixelToData(globalPosition.x(), globalPosition.y())
        self.plot.pinContextEntrySelected.emit(x,y)

class SetBackgroundCurveAction(PlotAction):
    def __init__(self, plot, parent=None):
        super().__init__(
            plot=plot,
            icon=icons.getQIcon("math-substract"),
            text="Set/Unset curve as background",
            parent=parent,
            triggered=self._actionTriggered,
            checkable=False,
        )
        self._parent = parent

    def _actionTriggered(self):
        plotArea = self.plot.getWidgetHandle()
        globalPosition = plotArea.mapToParent(self._parent._pos)
        x, y = self.plot.pixelToData(globalPosition.x(), globalPosition.y())
        self.plot.setBackgroundClicked.emit(x,y)
