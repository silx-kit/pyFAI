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
__date__ = "27/01/2025"
__status__ = "development"

from typing import Optional
import numpy
import os.path
import h5py
import silx.io
from silx.io.url import DataUrl
from silx.gui.plot.items import Scatter
from silx.gui import qt

from ..models import ImageIndices

from .ImagePlotWidget import ImagePlotWidget
from .MapPlotContextMenu import MapPlotContextMenu
from .OpenAxisDatasetAction import OpenAxisDatasetAction
from .ClearPointsAction import ClearPointsAction
from ..utils import (
    get_dataset,
    get_dataset_name,
    guess_axis_path,
)

_LEGEND = "MAP"


class MapPlotWidget(ImagePlotWidget):
    clearPointsSignal = qt.Signal()

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        self.axis_dataset_action = self._initAxisDatasetAction()
        self.clear_points_action = self._initclearPointsAction()
        self._toolbar.addAction(self.axis_dataset_action)
        self._toolbar.addAction(self.clear_points_action)

        self.addScatter([], [], [], legend=_LEGEND)
        scatter_item = self.getScatter(_LEGEND)
        assert isinstance(scatter_item, Scatter)
        self._scatter_item = scatter_item
        self._scatter_item.setVisualization(scatter_item.Visualization.REGULAR_GRID)
        self._first_plot = True

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
        value_data = self._scatter_item.getValueData(copy=False)
        index = self.getScatterIndex(x, y)
        if index is None:
            return

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

    def setAxes(self, Xname, Xvalues, Yname, Yvalues):
        """Changes the label (name) and numerical values for axis of the map"""
        self.setGraphXLabel(Xname)
        self.setGraphYLabel(Yname)
        z = self._scatter_item.getValueData(copy=False)
        Xvalues = numpy.atleast_1d(Xvalues)
        Yvalues = numpy.atleast_1d(Yvalues)
        assert z.size == Xvalues.size * Yvalues.size
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
                       x: Optional[numpy.ndarray]=None,
                       y: Optional[numpy.ndarray]=None,
                       xlabel: Optional[str]="X",
                       ylabel: Optional[str]="Y"):
        self.setGraphXLabel(xlabel)
        self.setGraphYLabel(ylabel)

        z = image.flatten()
        rows, cols = image.shape[:2]
        if (x is not None)  and (y is not None):
            assert x.size == cols
            assert y.size == rows

        if self._first_plot:
            if (x is None) or (y is None):
                x = numpy.arange(cols)
                y = numpy.arange(rows)
            x2 = numpy.outer(numpy.ones(rows), x).ravel()
            y2 = numpy.outer(y, numpy.ones(cols)).ravel()

            self._scatter_item.setData(x2, y2, z)
            dx = 0.5 * ((x[1:] - x[:-1]).mean()) / cols
            dy = 0.5 * ((y[1:] - y[:-1]).mean()) / rows
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
