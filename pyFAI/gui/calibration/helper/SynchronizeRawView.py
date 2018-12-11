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

import weakref
from silx.gui import qt


class SynchronizeRawView(object):
    """Synchronize the raw view of plots altogether.

    The active plot of each tasks synchonizing a model containing information
    to reach the loacation of the view.

    When the task is shown, the view of the visible plot is synchronized only
    if the shared view was changed.
    """

    def __init__(self):
        self.__plot = None
        self.__model = None

    def registerModel(self, model):
        assert(self.__model is None)
        self.__model = model
        self.__model.changed.connect(self.__rawPlotViewChanged)

    def registerTask(self, task):
        task.widgetShow.connect(self.__widgetShown)

    def __widgetShown(self):
        if self.__synchronizePlot:
            self.__synchronizePlot = False
            self.__synchronizePlotConfig()
            # Uses a timer to fix matplotlib issue at the very first redisplay
            # When using keep aspect ratio
            qt.QTimer.singleShot(1, self.__synchronizePlotView)

    def registerPlot(self, plot):
        assert(self.__plot is None)
        self.__plot = weakref.ref(plot)
        self.__synchronizePlot = False
        if hasattr(plot, "sigVisibilityChanged"):
            # At least silx 0.10
            plot.sigVisibilityChanged.connect(self.__plotVisibilityChanged)
        else:
            plot.getXAxis().sigLimitsChanged.connect(self.__plotViewChanged)
            plot.getYAxis().sigLimitsChanged.connect(self.__plotViewChanged)

    def __rawPlotViewChanged(self):
        self.__synchronizePlot = True

    def __plotViewChanged(self):
        if self.__model is None:
            return
        model = self.__model
        plot = self.__plot()
        if plot is not None:
            model.setFromPlot(plot)

    def __plotVisibilityChanged(self, isVisible):
        if not isVisible:
            # Save the state when it is hidden
            model = self.__model
            plot = self.__plot()
            model.setFromPlot(plot)

    def __synchronizePlotConfig(self):
        if self.__model is None:
            return
        model = self.__model
        plot = self.__plot()
        if plot is not None:
            model.synchronizePlotConfig(plot)

    def __synchronizePlotView(self):
        if self.__model is None:
            return
        model = self.__model
        plot = self.__plot()
        if plot is not None:
            model.synchronizePlotView(plot)
