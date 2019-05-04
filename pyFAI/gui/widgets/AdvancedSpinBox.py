# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2019 European Synchrotron Radiation Facility
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
__date__ = "09/04/2019"

import logging

from silx.gui import qt


_logger = logging.getLogger(__name__)


class AdvancedSpinBox(qt.QSpinBox):
    """
    A spin box with a little more custom behaviour.
    """

    def __init__(self, parent=None):
        super(AdvancedSpinBox, self).__init__(parent)
        self.__mouseWheelEnabled = True
        self.installEventFilter(self)

    def eventFilter(self, widget, event):
        """
        :param qt.QWidget widget: The widget receiving this event
        :param qt.QEvent event: Event received by the widget
        """

        if not self.__mouseWheelEnabled and event.type() == qt.QEvent.Wheel:
            # Inhibit mouse wheel event
            event.ignore()
            return True
        return False

    def mouseWheelEnabled(self):
        """
        True if the mouse wheel is used to changed the value contained by the
        spin box.

        :rtype: bool
        """
        return self.__mouseWheelEnabled

    def setMouseWheelEnabled(self, isWheelEnabled):
        """
        Change the behaviour of the mouse wheel with the value.

        :param bool isWheelEnabled: If `True` the mouse wheel can be used to
            edit the value contained into the spin box.
        """
        self.__mouseWheelEnabled = isWheelEnabled
