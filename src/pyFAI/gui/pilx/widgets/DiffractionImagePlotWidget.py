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
__date__ = "17/09/2024"
__status__ = "development"

import numpy
from silx.gui.plot.items import ImageData

from .ImagePlotWidget import ImagePlotWidget
from ..models import ROI_COLOR, ImageIndices
from ...utils.colorutils import DEFAULT_COLORMAP

_LEGEND = "IMAGE"


class DiffractionImagePlotWidget(ImagePlotWidget):

    def __init__(self, parent=None, backend=None):
        super().__init__(parent, backend)
        image_item = self.addImage([[]], legend=_LEGEND, colormap=DEFAULT_COLORMAP)
        assert isinstance(image_item, ImageData)
        self._image_item = image_item
        self._first_plot = True

    def _dataConverter(self, x, y):
        image = self._image_item.getData(copy=False)
        indices = self.getImageIndices(x, y)
        if indices is None:
            return

        return image[indices.row, indices.col]

    def setImageData(self, image: numpy.ndarray):
        self._image_item.setData(image)
        if self._first_plot:
            self.resetZoom()
            self._first_plot = False

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
