#!/usr/bin/env python
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
__date__ = "18/06/2025"
__status__ = "development"

from silx.gui import qt
from silx.gui.plot import items
from silx.gui.plot.items.roi import HorizontalRangeROI as SilxHorizontalRangeROI


class HorizontalRangeROI(SilxHorizontalRangeROI):
    """A range ROI with committed changes and Ctrl-drag symmetry."""

    sigRangeCommitted = qt.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drag_center = None
        for marker in (self._markerMin, self._markerMax):
            marker.sigDragStarted.connect(self._rememberDragCenter)
            marker.sigDragFinished.connect(self._clearDragCenter)
        self.sigEditingFinished.connect(self.sigRangeCommitted)

    def setRange(self, vmin: float, vmax: float):
        super().setRange(vmin, vmax)
        self.sigRangeCommitted.emit()

    def _rememberDragCenter(self):
        self._drag_center = self.getCenter()

    def _clearDragCenter(self):
        self._drag_center = None

    def _minPositionChanged(self, event):
        symmetric = qt.QApplication.keyboardModifiers() & qt.Qt.KeyboardModifier.ControlModifier
        if event is items.ItemChangedType.POSITION and symmetric and self._drag_center is not None:
            minimum = min(self.sender().getXPosition(), self._drag_center)
            self._updatePos(minimum, 2 * self._drag_center - minimum, force=True)
        else:
            super()._minPositionChanged(event)

    def _maxPositionChanged(self, event):
        symmetric = qt.QApplication.keyboardModifiers() & qt.Qt.KeyboardModifier.ControlModifier
        if event is items.ItemChangedType.POSITION and symmetric and self._drag_center is not None:
            maximum = max(self.sender().getXPosition(), self._drag_center)
            self._updatePos(2 * self._drag_center - maximum, maximum, force=True)
        else:
            super()._maxPositionChanged(event)
