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
__date__ = "03/06/2016"
__status__ = "production"

import sys
import matplotlib

import matplotlib.cm
has_Qt = True
if ('PySide' in sys.modules):
    from PySide import QtGui, QtCore, QtUiTools, QtWebKit
    from PySide.QtCore import SIGNAL, Signal

    from .third_party.pyside_dynamic import loadUi as _loadUi


    #we need to handle uic !!!
    """
    loadUi(uifile, baseinstance=None, package='') -> widget

Load a Qt Designer .ui file and return an instance of the user interface.

uifile is a file name or file-like object containing the .ui file.
baseinstance is an optional instance of the Qt base class.  If specified
then the user interface is created in it.  Otherwise a new instance of the
base class is automatically created.
package is the optional package which is used as the base for any relative
imports of custom widgets.

    """
    class uic(object):
        @staticmethod
        def loadUi(uifile, baseinstance=None, package=''):
            """Load a Qt Designer .ui file and return an instance of the user interface.

            uifile is a file name or file-like object containing the .ui file.
            baseinstance is an optional instance of the Qt base class.  If specified
            then the user interface is created in it.  Otherwise a new instance of the
            base class is automatically created.
            package is the optional package which is used as the base for any relative
            imports of custom widgets.

            Totally untested !
            """
            return _loadUi(uifile, baseinstance,
                           customWidgets={'Line': QtGui.QFrame})

    sys.modules["PySide.uic"] = uic
    matplotlib.rcParams['backend.qt4'] = 'PySide'
    Qt_version = QtCore.__version_info__
else:
    try:
        from PyQt4 import QtGui, QtCore, uic, QtWebKit
        from PyQt4.QtCore import SIGNAL, pyqtSignal as Signal
    except ImportError:
        has_Qt = False
    else:
        from PyQt4.QtCore import QT_VERSION_STR
        from PyQt4.Qt import PYQT_VERSION_STR
        from sip import SIP_VERSION_STR
        Qt_version = tuple(int(i) for i in QT_VERSION_STR.split(".")[:3])
        SIP_version = tuple(int(i) for i in SIP_VERSION_STR.split(".")[:3])
        PyQt_version = tuple(int(i) for i in PYQT_VERSION_STR.split(".")[:3])

if has_Qt:
    matplotlib.rcParams['backend'] = 'Qt4Agg'
    import matplotlib.pyplot
    matplotlib.pyplot.switch_backend('Qt4Agg')
    # Dear reader, I apologize for something that ugly !
    # Any cleaner version would be appreciated

    # matplotlib.use('Qt4Agg')
    from matplotlib.backends import backend_qt4 as backend
    from matplotlib import pyplot
    from matplotlib import pylab
else:
    from matplotlib import pyplot
    from matplotlib import pylab
    from matplotlib.backends import backend
    QtGui = QtCore = QtUiTools = QtWebKit = loadUi = None
    SIGNAL = Signal = None

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
