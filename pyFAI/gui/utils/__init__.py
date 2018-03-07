#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2016-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""
Module providing gui util tools
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/03/2018"
__status__ = "production"


from .. import matplotlib
from silx.gui import qt

main_loop = False


def update_fig(fig=None):
    """
    Update a matplotlib figure with a Qt4 backend

    :param fig: pylab figure
    """
    if fig and "canvas" in dir(fig) and fig.canvas:
        fig.canvas.draw()
        if "Qt4" in matplotlib.pylab.get_backend():
            event = qt.QResizeEvent(fig.canvas.size(), fig.canvas.size())
            qt.qApp.postEvent(fig.canvas, event)
            if not main_loop:
                qt.QCoreApplication.processEvents()


class Event(object):
    "Dummy class for dummy things"
    def __init__(self, width, height):
        self.width = width
        self.height = height


def maximize_fig(fig=None):
    """
    Try to set the figure fullscreen
    """
    if fig and "canvas" in dir(fig) and fig.canvas:
        if "Qt4" in matplotlib.pylab.get_backend():
            fig.canvas.setWindowState(qt.Qt.WindowMaximized)
        else:
            mng = matplotlib.pylab.get_current_fig_manager()
            # attempt to maximize the figure ... lost hopes.
            win_shape = (1920, 1080)
            event = Event(*win_shape)
            try:
                mng.resize(event)
            except TypeError:
                mng.resize(*win_shape)
    update_fig(fig)
