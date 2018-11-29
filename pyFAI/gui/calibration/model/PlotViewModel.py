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

import numpy
from .DataModel import DataModel


class PlotViewModel(DataModel):
    """This model allow to store and restitute a plot view.

    Stored data can be applyed to another plot in order to synchronize
    location of the data coords.
    """

    def __getRangeInPixel(self, plot):
        """Returns the size of the axis in pixel"""
        bounds = plot.getPlotBoundsInPixels()
        # bounds: left, top, width, height
        return numpy.array([bounds[2], bounds[3]])

    def __getMinMaxPixelCoord(self, plot):
        """Returns the size of the axis in pixel"""
        bounds = plot.getPlotBoundsInPixels()
        # bounds: left, top, width, height
        coord1 = numpy.array([bounds[0], bounds[1]])
        coord2 = coord1 + numpy.array([bounds[2], bounds[3]])
        return coord1, coord2

    def __getAxisDataRange(self, axis):
        limits = axis.getLimits()
        dataRange = limits[1] - limits[0]
        dataRange = abs(dataRange)
        return dataRange

    def __getDataRange(self, plot):
        return numpy.array([self.__getAxisDataRange(plot.getXAxis()),
                            self.__getAxisDataRange(plot.getYAxis())])

    def __getPixelSize(self, plot):
        dataRange = self.__getDataRange(plot)
        pixelRange = self.__getRangeInPixel(plot)
        return dataRange / pixelRange

    def setFromPlot(self, plot):
        pixelSize = self.__getPixelSize(plot)
        dataCoordAtPixelCoordZero = numpy.array(plot.pixelToData(x=0, y=0))
        isYaxisInverted = plot.getYAxis().isInverted()
        isKeepAspectRatio = plot.isKeepDataAspectRatio()
        interactionMode = plot.getInteractiveMode()["mode"]
        if interactionMode not in set(['pan', 'zoom']):
            interactionMode = 'pan'
        plotConfig = isYaxisInverted, isKeepAspectRatio, interactionMode
        value = dataCoordAtPixelCoordZero, pixelSize, plotConfig
        self.setValue(value)

    def __setViewLocation(self, plot, coord1, coord2):
        xLimits = coord1[0], coord2[0]
        yLimits = coord1[1], coord2[1]
        if xLimits[0] > xLimits[1]:
            xLimits = xLimits[1], xLimits[0]
        if yLimits[0] > yLimits[1]:
            yLimits = yLimits[1], yLimits[0]
        xAxis = plot.getXAxis()
        yAxis = plot.getYAxis()
        xAxis.setLimits(*xLimits)
        yAxis.setLimits(*yLimits)

    def synchronizePlotConfig(self, plot):
        value = self.value()
        plotConfig = value[2]
        isYaxisInverted, isKeepAspectRatio, interactionMode = plotConfig
        plot.setKeepDataAspectRatio(isKeepAspectRatio)
        plot.getYAxis().setInverted(isYaxisInverted)
        plot.setInteractiveMode(interactionMode)

    def synchronizePlotView(self, plot):
        value = self.value()
        dataCoordAtPixelCoordZero, pixelSize, plotConfig = value
        isYaxisInverted = plotConfig[0]
        if not isYaxisInverted:
            # Coord of pixel and data are switched sometimes
            pixelSize = numpy.array([pixelSize[0], -pixelSize[1]])

        # Pixel coords
        coord1, coord2 = self.__getMinMaxPixelCoord(plot)
        # Data coords
        coord1 = dataCoordAtPixelCoordZero + coord1 * pixelSize
        coord2 = dataCoordAtPixelCoordZero + coord2 * pixelSize
        self.__setViewLocation(plot, coord1, coord2)
