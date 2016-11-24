#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Module providing gui util tools
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/10/2016"
__status__ = "production"


from . import matplotlib
from . import qt

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
