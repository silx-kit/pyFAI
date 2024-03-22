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

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/03/2024"
__status__ = "development"

from silx.gui.plot import PlotWidget
from silx.gui.plot.actions import PlotAction


from ..HorizontalRangeROI import HorizontalRangeROI
from ..models import ROI_COLOR


class RoiModeAction(PlotAction):
    def __init__(self, plot: PlotWidget, parent=None):
        super(RoiModeAction, self).__init__(
            plot,
            icon=HorizontalRangeROI.ICON,
            text="ROI mode",
            tooltip="Draw a ROI",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        # Listen to mode change
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        # Init the state
        self._modeChanged(None)

    def _modeChanged(self, source):
        modeDict = self.plot.getInteractiveMode()
        old = self.blockSignals(True)
        self.setChecked(modeDict["mode"] == "select-draw")
        self.blockSignals(old)

    def _actionTriggered(self, checked=False):
        plot = self.plot
        if plot is not None:
            plot.setInteractiveMode("select-draw", shape="rectangle", color=ROI_COLOR)
