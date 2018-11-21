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
__date__ = "21/11/2018"

from .DataModel import DataModel
from silx.gui.plot.items.axis import XAxis, YAxis


class PlotViewModel(DataModel):

    def __getAxisRangeInPixel(self, axis):
        """Returns the size of the axis in pixel"""
        bounds = axis._getPlot().getPlotBoundsInPixels()
        # bounds: left, top, width, height
        if isinstance(axis, XAxis):
            return bounds[2]
        elif isinstance(axis, YAxis):
            return bounds[3]
        else:
            assert(False)

    def __getAxisInfo(self, axis):
        limits = axis.getLimits()
        valueRange = limits[1] - limits[0]
        middle = (limits[0] + limits[1]) * 0.5
        pixelRange = self.__getAxisRangeInPixel(axis)
        pixelSize = valueRange / pixelRange
        return middle, pixelSize

    def __getLimitsFromMiddle(self, axis, pos, pixelSize):
        """Returns the limits to apply to this axis to move the `pos` into the
        center of this axis.
        :param Axis axis:
        :param float pos: Position in the center of the computed limits
        :param Union[None,float] pixelSize: Pixel size to apply to compute the
            limits. If `None` the current pixel size is applyed.
        """
        pixelRange = self.__getAxisRangeInPixel(axis)
        a = pos - pixelRange * 0.5 * pixelSize
        b = pos + pixelRange * 0.5 * pixelSize
        if a > b:
            return b, a
        return a, b

    def __setAxisInfo(self, axis, info):
        middle, pixelSize = info
        limits = self.__getLimitsFromMiddle(axis, middle, pixelSize)
        axis.setLimits(*limits)

    def setFromPlot(self, plot):
        xAxis = plot.getXAxis()
        yAxis = plot.getYAxis()
        value = self.__getAxisInfo(xAxis), self.__getAxisInfo(yAxis)
        self.setValue(value)

    def synchronizePlot(self, plot):
        value = self.value()
        xInfo, yInfo = value
        xAxis = plot.getXAxis()
        yAxis = plot.getYAxis()
        self.__setAxisInfo(yAxis, yInfo)
        self.__setAxisInfo(xAxis, xInfo)
