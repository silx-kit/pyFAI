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
__date__ = "08/08/2018"


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
