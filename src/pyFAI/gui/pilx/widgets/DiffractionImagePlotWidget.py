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
from __future__ import annotations

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "06/01/2026"
__status__ = "development"

import numpy
from silx.gui import qt
from silx.gui.plot.actions import PlotAction
from silx.gui.plot.backends.BackendMatplotlib import BackendMatplotlibQt
from silx.gui.plot.items import ImageData

from ...utils.colorutils import DEFAULT_COLORMAP
from ..models import ROI_COLOR, ImageIndices
from .ImagePlotWidget import ImagePlotWidget

_LEGEND = "IMAGE"


class DetectorMatplotlibBackend(BackendMatplotlibQt):
    def setLimits(self, xmin, xmax, ymin, ymax, y2min=None, y2max=None):
        # PlotWidget has already enforced aspect using the actual plot area.
        # The silx backend repeats it using the full canvas, which can expand
        # the limits on every pan event when the two rectangles differ.
        keep_aspect = self.isKeepDataAspectRatio()
        self.setKeepDataAspectRatio(False)
        try:
            super().setLimits(xmin, xmax, ymin, ymax, y2min, y2max)
        finally:
            self.setKeepDataAspectRatio(keep_aspect)


class DetectorRoiModeAction(PlotAction):
    def __init__(self, plot, parent=None):
        super().__init__(
            plot,
            icon="shape-circle",
            text="2θ ROI mode",
            tooltip="Select the 2θ ROI from the detector image",
            triggered=self._actionTriggered,
            checkable=True,
            parent=parent,
        )
        self.plot.sigInteractiveModeChanged.connect(self._modeChanged)
        self._modeChanged(None)

    def _modeChanged(self, source):
        old = self.blockSignals(True)
        self.setChecked(self.plot.getInteractiveMode()["mode"] == "select")
        self.blockSignals(old)

    def _actionTriggered(self, checked=False):
        if checked:
            self.plot.setInteractiveMode("select", source=self)
        else:
            self.plot.resetInteractiveMode()


class DiffractionImagePlotWidget(ImagePlotWidget):

    def __init__(self, parent=None, backend=None):
        if backend is None or backend in ("matplotlib", "mpl"):
            backend = DetectorMatplotlibBackend
        super().__init__(parent, backend)
        self.setAxesMargins(left=0.10, top=0.16, right=0.03, bottom=0.10)
        self._roi_mode_action = DetectorRoiModeAction(self, self._toolbar)
        self._toolbar.insertAction(
            self._toolbar.display_separator, self._roi_mode_action
        )
        self._roi_mode_action.trigger()
        image_item = self.addImage([[]], legend=_LEGEND, colormap=DEFAULT_COLORMAP)
        if not isinstance(image_item, ImageData):
            raise RuntimeError("addImage should return a ImageData instance")
        self._image_item = image_item
        self._first_plot = True
        self._reset_zoom_when_shown = False

    def emitMouseClickSignal(self, signal_data):
        if (
            self.getInteractiveMode()["mode"] != "select"
            or signal_data.get("button") != "left"
        ):
            return
        super().emitMouseClickSignal(signal_data)

    def _dataConverter(self, x, y):
        image = self._image_item.getData(copy=False)
        indices = self.getImageIndices(x, y)
        if indices is None:
            return

        return image[indices.row, indices.col]

    def setImageData(self,
                     image: numpy.ndarray,
                     title: str=""):
        self._image_item.setData(image)
        if self._first_plot:
            if self.isVisible():
                qt.QTimer.singleShot(0, self.resetZoom)
            else:
                self._reset_zoom_when_shown = True
            self._first_plot = False
        self.setGraphTitle(title)
        backend = self.getBackend()
        if hasattr(backend, "ax"):
            backend.ax.title.set_fontsize(11)

    def showEvent(self, event):
        super().showEvent(event)
        if self._reset_zoom_when_shown:
            self._reset_zoom_when_shown = False
            qt.QTimer.singleShot(0, self.resetZoom)

    def getImageIndices(self, x_data: float, y_data: float) -> ImageIndices | None:
        tmp = self.dataToPixel(x_data, y_data)
        if tmp:
            pixel_x, pixel_y = tmp
            picking_result = self._image_item.pick(pixel_x, pixel_y)
        else:
            picking_result = None
        if picking_result is None:
            return
        # Image dims are first rows then cols
        row_indices_array, col_indices_array = picking_result.getIndices(copy=False)
        return ImageIndices(row=row_indices_array[0], col=col_indices_array[0])

    def addContour(
        self, contour: numpy.ndarray, legend: str, linestyle: str | None=None
    ):
        self.addCurve(
            contour[:, 1],
            contour[:, 0],
            legend=legend,
            linestyle=linestyle,
            color=ROI_COLOR,
            resetzoom=False,
            selectable=False,
        )
