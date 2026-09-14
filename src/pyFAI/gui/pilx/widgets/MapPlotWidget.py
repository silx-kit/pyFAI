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
__date__ = "31/01/2025"
__status__ = "development"

import os.path

import h5py
import numpy
import silx.io
from silx.gui import qt
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.plot.actions.control import ColormapAction
from silx.gui.plot.items import Scatter
from silx.io.url import DataUrl

from ..models import ImageIndices
from ..utils import (
    get_dataset,
    get_dataset_name,
    guess_axis_path,
)
from .ClearPointsAction import ClearPointsAction
from .ImagePlotWidget import ImagePlotWidget
from .MapPlotContextMenu import MapPlotContextMenu
from .OpenAxisDatasetAction import OpenAxisDatasetAction

_LEGEND = "MAP"
_RGB_LEGEND = "RGB_MAP"


class MapColormapDialog(ColormapDialog):
    def __init__(self, parent=None):
        self._colormap_name_locked = False
        super().__init__(parent)

    def setColormapNameLocked(self, locked):
        self._colormap_name_locked = bool(locked)
        colormap = self.getColormap()
        editable = colormap is not None and colormap.isEditable()
        self._comboBoxColormap.setEnabled(editable and not locked)

    def _applyColormap(self):
        super()._applyColormap()
        if self._colormap_name_locked:
            self._comboBoxColormap.setEnabled(False)


class MapColormapAction(ColormapAction):
    """Use the standard colormap dialog to scale the active RGB channel."""

    @staticmethod
    def _createDialog(parent):
        dialog = MapColormapDialog(parent=parent)
        dialog.setModal(False)
        return dialog

    def _updateColormap(self):
        if self._dialog is None:
            return
        plot = self.plot
        if plot._rgb_raw_data is None:
            self._dialog.setColormapNameLocked(False)
            super()._updateColormap()
            return

        channel = plot._rgb_channel
        index = "RGB".index(channel)
        channel_name = {"R": "Red", "G": "Green", "B": "Blue"}[channel]
        self._dialog.setWindowTitle(f"{channel_name} channel scaling")
        self._dialog.setColormap(plot._rgb_colormaps[channel])
        plot._rgb_dialog_data = plot._rgb_raw_data[:, :, index]
        self._dialog.setData(plot._rgb_dialog_data)
        self._dialog.setColormapNameLocked(True)


class MapPlotWidget(ImagePlotWidget):
    clearPointsSignal = qt.Signal()

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self._rgb_raw_data = None
        self._rgb_dialog_data = None
        self._rgb_data = None
        self._rgb_colormaps = {}
        self._rgb_channel = "R"
        previous_colormap_action = self._toolbar.colormap_action
        self._toolbar.colormap_action = MapColormapAction(self, self._toolbar)
        self._toolbar.insertAction(
            previous_colormap_action, self._toolbar.colormap_action
        )
        self._toolbar.removeAction(previous_colormap_action)
        previous_colormap_action.deleteLater()

        self.axis_dataset_action = self._initAxisDatasetAction()
        self.clear_points_action = self._initclearPointsAction()
        self._toolbar.addAction(self.axis_dataset_action)
        self._toolbar.addAction(self.clear_points_action)

        self.addScatter([], [], [], legend=_LEGEND)
        scatter_item = self.getScatter(_LEGEND)
        if not isinstance(scatter_item, Scatter):
            raise RuntimeError(f"`scatter_item` is {type(scatter_item)}, not a `silx.gui.plot.items.scatter.Scatter` instance")
        self._scatter_item = scatter_item
        self._scatter_item.setVisualization(scatter_item.Visualization.REGULAR_GRID)
        self._first_plot = True
        self._map_shape = None

        self._build_context_menu()

    def _build_context_menu(self):
        plotArea = self.getWidgetHandle()
        plotArea.setContextMenuPolicy(qt.Qt.CustomContextMenu)
        plotArea.customContextMenuRequested.connect(self._contextMenu)

    def _contextMenu(self, pos):
        menu = MapPlotContextMenu(plot=self)
        menu._exec(pos=pos)

    def _initAxisDatasetAction(self):
        action = OpenAxisDatasetAction(self._toolbar)
        action.datasetOpened.connect(self.changeAxes)
        return action

    def _initclearPointsAction(self):
        action = ClearPointsAction(self._toolbar)
        action.clearPoints.connect(self.clearPoints)
        return action

    def _dataConverter(self, x, y):
        index = self.getScatterIndex(x, y)
        if index is None:
            return
        if self._rgb_raw_data is not None:
            row, col = numpy.unravel_index(index, self._map_shape)
            return tuple(self._rgb_raw_data[row, col])
        value_data = self._scatter_item.getValueData(copy=False)
        return value_data[index]

    def findCenterOfNearestPixel(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        index = self.getScatterIndex(x, y)

        if index is None:
            return 0, 0

        x_data: numpy.ndarray = self._scatter_item.getXData(copy=False)
        y_data: numpy.ndarray = self._scatter_item.getYData(copy=False)

        return (x_data[index], y_data[index])

    def getMapPointCoordinates(
        self, indices: ImageIndices
    ) -> tuple[float, float] | None:
        """Return this plot's coordinates for a map point."""
        if self._map_shape is None:
            return None
        rows, cols = self._map_shape
        if not (0 <= indices.row < rows and 0 <= indices.col < cols):
            return None

        index = indices.row * cols + indices.col
        x_data = self._scatter_item.getXData(copy=False)
        y_data = self._scatter_item.getYData(copy=False)
        return x_data[index], y_data[index]

    def setAxes(self, Xname, Xvalues, Yname, Yvalues):
        """Changes the label (name) and numerical values for axis of the map"""
        self.setGraphXLabel(Xname)
        self.setGraphYLabel(Yname)
        z = self._scatter_item.getValueData(copy=False)
        Xvalues = numpy.atleast_1d(Xvalues)
        Yvalues = numpy.atleast_1d(Yvalues)
        if  z.size != Xvalues.size * Yvalues.size:
            raise RuntimeError("size of the Xvalues*Yvalues does not match scatter_item size ")
        x = numpy.outer(numpy.ones(Yvalues.size), Xvalues).ravel()
        y = numpy.outer(Yvalues, numpy.ones(Xvalues.size)).ravel()
        self._scatter_item.setData(x, y, z)
        self.resetZoom()

    def changeAxes(self, axis_data_url: DataUrl):
        with silx.io.open(axis_data_url.file_path()) as h5:
            if not isinstance(h5, h5py.Group):
                return
            axis0_path: str | None = axis_data_url.data_path()
            if axis0_path is None:
                return
            axis1_path = guess_axis_path(axis0_path, h5)
            if axis1_path is None:
                return

            axis0_dataset = get_dataset(h5, axis0_path)
            axis0 = axis0_dataset[()]
            axis0_name = get_dataset_name(axis0_dataset)
            axis1_dataset = get_dataset(h5, axis1_path)
            axis1 = axis1_dataset[()]
            axis1_name = get_dataset_name(axis1_dataset)

        z = self._scatter_item.getValueData(copy=False)
        self._scatter_item.setData(axis0, axis1, z)
        self.setGraphXLabel(axis0_name)
        self.setGraphYLabel(axis1_name)
        self.resetZoom()

    def clearPoints(self):
        self.clearPointsSignal.emit()

    def setScatterData(self,
                       image: numpy.ndarray,
                       x: numpy.ndarray | None=None,
                       y: numpy.ndarray | None=None,
                       xlabel: str | None="X",
                       ylabel: str | None="Y"):
        self.setGraphXLabel(xlabel)
        self.setGraphYLabel(ylabel)

        z = image.flatten()
        rows, cols = image.shape[:2]
        self._map_shape = rows, cols
        if (x is not None)  and (y is not None):
            if x.size != cols:
                raise RuntimeError(f"size of x({x.size}) does not march the number of columns of the image ({cols})")
            if y.size != rows:
                raise RuntimeError(f"size of x({y.size}) does not march the number of columns of the image ({rows})")

        if self._first_plot:
            if (x is None) or (y is None):
                x = numpy.arange(cols)
                y = numpy.arange(rows)
            x2 = numpy.outer(numpy.ones(rows), x).ravel()
            y2 = numpy.outer(y, numpy.ones(cols)).ravel()

            self._scatter_item.setData(x2, y2, z)
            dx = 0.5 / (cols - 1) if cols > 1 else 0.0
            dy = 0.5 / (rows - 1) if rows > 1 else 0.0
            self.setDataMargins(dx, dx, dy, dy)
            self.resetZoom()
            self._first_plot = False
        else:
            if (x is not None) and (y is not None):
                x2 = numpy.outer(numpy.ones(rows), x).ravel()
                y2 = numpy.outer(y, numpy.ones(cols)).ravel()
            else:
                x2 = self._scatter_item.getXData(copy=False)
                y2 = self._scatter_item.getYData(copy=False)
            self._scatter_item.setData(x2, y2, z)

    def setRgbData(
        self,
        image: numpy.ndarray,
        x: numpy.ndarray | None = None,
        y: numpy.ndarray | None = None,
        xlabel: str = "X",
        ylabel: str = "Y",
    ):
        """Display an RGB image while retaining the scatter grid for picking."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("RGB source maps must have shape (rows, columns, 3)")
        rows, cols = image.shape[:2]
        if x is None:
            x = numpy.arange(cols, dtype=float)
        if y is None:
            y = numpy.arange(rows, dtype=float)
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        if x.size != cols or y.size != rows:
            raise RuntimeError("RGB map dimensions do not match its map axes")

        self._rgb_raw_data = numpy.asarray(image, dtype=float)
        if not self._rgb_colormaps:
            for channel, name in zip("RGB", ("red", "green", "blue")):
                colormap = Colormap(
                    name=name,
                    normalization=Colormap.LINEAR,
                    autoscaleMode=Colormap.PERCENTILE,
                )
                colormap.setAutoscalePercentiles((1.0, 99.0))
                colormap.sigChanged.connect(self._updateRgbImage)
                self._rgb_colormaps[channel] = colormap

        self.setScatterData(
            numpy.zeros((rows, cols), dtype=float), x, y, xlabel, ylabel
        )
        self._scatter_item.setAlpha(0.0)
        dx = (x[-1] - x[0]) / (x.size - 1) if x.size > 1 else 1.0
        dy = (y[-1] - y[0]) / (y.size - 1) if y.size > 1 else 1.0
        self.addImage(
            numpy.zeros((rows, cols, 3), dtype=numpy.uint8),
            legend=_RGB_LEGEND,
            origin=(x[0] - 0.5 * dx, y[0] - 0.5 * dy),
            scale=(dx, dy),
            resetzoom=False,
        )
        self._updateRgbImage()
        self._colorBarWidget.hide()
        self.axis_dataset_action.setEnabled(False)
        self._toolbar.colormap_action.setText("RGB channel scaling")
        channel_name = {"R": "red", "G": "green", "B": "blue"}[
            self._rgb_channel
        ]
        self._toolbar.colormap_action.setToolTip(
            f"Scale the {channel_name} channel"
        )
        self._toolbar.colormap_action._updateColormap()

    def setRgbChannel(self, channel):
        if channel not in "RGB":
            raise ValueError(f"Unsupported RGB channel: {channel}")
        self._rgb_channel = channel
        channel_name = {"R": "red", "G": "green", "B": "blue"}[channel]
        self._toolbar.colormap_action.setToolTip(f"Scale the {channel_name} channel")
        self._toolbar.colormap_action._updateColormap()

    def _updateRgbImage(self):
        if self._rgb_raw_data is None:
            return
        rgb = numpy.empty(self._rgb_raw_data.shape, dtype=numpy.uint8)
        for index, channel in enumerate("RGB"):
            rgba = self._rgb_colormaps[channel].applyToData(
                self._rgb_raw_data[:, :, index]
            )
            rgb[:, :, index] = rgba[:, :, index]
        self._rgb_data = rgb
        image = self.getImage(_RGB_LEGEND)
        if image is not None:
            image.setData(rgb)

    def getImageIndices(self, x_data: float, y_data: float) -> ImageIndices | None:
        pixels = self.dataToPixel(x_data, y_data)
        if pixels is None:
            return

        pixel_x, pixel_y = pixels
        # Use the base class `pick` to retrieve row and col indices instead of the scatter index
        picking_result = super(Scatter, self._scatter_item).pick(pixel_x, pixel_y)
        if picking_result is None:
            return
        # Image dims are first rows then cols
        row_indices_array, col_indices_array = picking_result.getIndices(copy=False)
        return ImageIndices(row=row_indices_array[0], col=col_indices_array[0])

    def getScatterIndex(self, x_data: float, y_data: float) -> int | None:
        pixels = self.dataToPixel(x_data, y_data)
        if pixels is None:
            return

        pixel_x, pixel_y = pixels

        picking_result = self._scatter_item.pick(pixel_x, pixel_y)
        if picking_result is None:
            return
        index_array = picking_result.getIndices(copy=False)
        return index_array[0]

    def onFileChange(self, new_file_name: str):
        self.axis_dataset_action.setFileDirectory(os.path.dirname(new_file_name))
