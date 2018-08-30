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
__date__ = "14/08/2018"

import weakref


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
        task.widgetShow.connect(self.__synchronizePlot)

    def registerPlot(self, plot):
        assert(self.__plot is None)
        self.__plot = weakref.ref(plot)
        self.__synchronizePlotView = False
        plot.getXAxis().sigLimitsChanged.connect(self.__plotViewChanged)
        plot.getYAxis().sigLimitsChanged.connect(self.__plotViewChanged)

    def __rawPlotViewChanged(self):
        self.__synchronizePlotView = True

    def __plotViewChanged(self):
        if self.__model is None:
            return
        view = self.__model
        plot = self.__plot()
        if plot is not None:
            view.setFromPlot(plot)

    def __synchronizePlot(self):
        if self.__model is None:
            return
        if self.__synchronizePlotView:
            self.__synchronizePlotView = False
            view = self.__model
            plot = self.__plot()
            if plot is not None:
                view.synchronizePlot(plot)
