# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides convenient functions about Qt events.
"""

from __future__ import division


__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/01/2019"


from silx.gui import qt
from silx.gui.utils import concurrent

import logging
import types

_logger = logging.getLogger(__name__)


class SimulatedSignal:
    """
    Simulated signal to create signal at runtime.

    .. codeauthor:: Abstract Factory <hello@pipi.io>
    .. note:: Code from http://abstractfactory.io/blog/dynamic-signals-in-pyqt/
    """
    def __init__(self):
        self.__subscribers = []

    def subs(self):
        return self.__subscribers

    def emit(self, *args, **kwargs):
        for subs in self.__subscribers:
            subs(*args, **kwargs)

    def connect(self, func):
        self.__subscribers.append(func)

    def disconnect(self, func):
        try:
            self.__subscribers.remove(func)
        except ValueError:
            _logger.error('Warning: function %s not removed from signal %s', func, self)


def createCloseSignal(widget):
    """
    Create a Qt close signal to the widget as sigClosed attribute

    :type widget: Qt.QWidget
    """

    if hasattr(widget, "sigClosed"):
        if isinstance(widget.sigClosed, SimulatedSignal):
            return
        raise Exception("Attribute sigClose already exists and is not a Qt signal")

    def closeEvent(self, event):
        widget.sigClosed.emit()
        self._createCloseSignal_oldCloseEvent(event)

    widget.sigClosed = SimulatedSignal()
    widget._createCloseSignal_oldCloseEvent = widget.closeEvent
    widget.closeEvent = types.MethodType(closeEvent, widget)


def createShowSignal(widget):
    """
    Create a Qt shown signal to the widget as sigShown attribute

    :type widget: Qt.QWidget
    """

    if hasattr(widget, "sigShown"):
        if isinstance(widget.sigShown, SimulatedSignal):
            return
        raise Exception("Attribute sigClose already exists and is not a Qt signal")

    def showEvent(self, event):
        widget.sigShown.emit()
        self._createShowSignal_oldShowEvent(event)

    widget.sigShown = SimulatedSignal()
    widget._createShowSignal_oldShowEvent = widget.showEvent
    widget.showEvent = types.MethodType(showEvent, widget)


def createHideSignal(widget):
    """
    Create a Qt hidden signal to the widget as sigHidden attribute

    :type widget: Qt.QWidget
    """

    if hasattr(widget, "sigHidden"):
        if isinstance(widget.sigHidden, SimulatedSignal):
            return
        raise Exception("Attribute sigClose already exists and is not a Qt signal")

    def hideEvent(self, event):
        widget.sigHidden.emit()
        self._createHideSignal_oldHideEvent(event)

    widget.sigHidden = SimulatedSignal()
    widget._createHideSignal_oldHideEvent = widget.hideEvent
    widget.hideEvent = types.MethodType(hideEvent, widget)


class QtProxifier(qt.QObject):
    """Provide a safe Qt object from an unsafe object."""

    _callRequested = qt.Signal(str, tuple, dict)

    def __init__(self, target):
        qt.QObject.__init__(self)
        self.__target = target
        self._callRequested.connect(self.__callRequested)

    def _target(self):
        return self.__target

    def __getattr__(self, name):
        """Convert a call request to a Qt signal"""
        def createSignal(*args, **kwargs):
            method = getattr(self.__target, name)
            result = concurrent.submitToQtMainThread(method, *args, **kwargs)
            return result
        return createSignal
