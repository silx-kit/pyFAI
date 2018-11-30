# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Utils to update the plot background according to the changes of the
application styles"""

__authors__ = ["V. Valls"]
__license__ = "MIT"


import weakref
from ..CalibrationContext import CalibrationContext


class SynchronizePlotBackground(object):

    def __init__(self, plot=None):
        self.__registerBackgroundColor()
        self.__plot = weakref.ref(plot)
        self.__previousColor = None
        self.__updateApplicationStyle()

    def getPlot(self):
        return self.__plot()

    def __registerBackgroundColor(self):
        context = CalibrationContext.instance()
        context.sigStyleChanged.connect(self.__updateApplicationStyle)

    def __updateApplicationStyle(self):
        context = CalibrationContext.instance()
        color = context.getBackgroundColor()
        if color != self.__previousColor:
            self.__previousColor = color
            self.__updateBackgroundColor(color)

    def __updateBackgroundColor(self, color):
        plot = self.getPlot()
        if hasattr(plot, "setDataBackgroundColor"):
            # Silx 0.10
            plot.setDataBackgroundColor(color)
        else:
            # Older silx
            pass
