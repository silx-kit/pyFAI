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
gui_utils

Module to handle matplotlib and the Qt backend

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "27/06/2014"
__status__ = "production"

import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib.backends import backend_qt4 as backend
import pylab

from PyQt4 import QtGui, QtCore

main_loop = False

def update_fig(fig=None):
    """
    Update a matplotlib figure with a Qt4 backend

    @param fig: pylab figure
    """
    if fig and "canvas" in dir(fig) and fig.canvas:
        fig.canvas.draw()
        if "Qt4" in pylab.get_backend():
            QtGui.qApp.postEvent(fig.canvas,
                                 QtGui.QResizeEvent(fig.canvas.size(),
                                                    fig.canvas.size()))
            if not main_loop:
                QtCore.QCoreApplication.processEvents()

def maximize_fig(fig=None):
    """
    Try to set the figure fullscreen
    """
    if fig and "canvas" in dir(fig) and fig.canvas:
        if "Qt4" in pylab.get_backend():
            fig.canvas.setWindowState(QtCore.Qt.WindowMaximized)
        else:
            mng = pylab.get_current_fig_manager()
            # attempt to maximize the figure ... lost hopes.
            win_shape = (1920, 1080)
            event = Event(*win_shape)
            try:
                mng.resize(event)
            except TypeError:
                 mng.resize(*win_shape)
    update_fig(fig)
