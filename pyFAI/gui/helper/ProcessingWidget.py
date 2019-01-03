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
__date__ = "03/01/2019"

from silx.gui import qt
from silx.gui.widgets.WaitingPushButton import WaitingPushButton


def createProcessingWidgetOverlay(parent):
    """Create a widget overlay to show that the application is processing data
    to update the plot.

    :param qt.QWidget widget: Widget containing the overlay
    :rtype: qt.QWidget
    """
    if hasattr(parent, "centralWidget"):
        parent = parent.centralWidget()
    button = WaitingPushButton(parent)
    button.setWaiting(True)
    button.setText("Processing...")
    button.setDown(True)
    position = parent.size()
    size = button.sizeHint()
    position = (position - size) / 2
    rect = qt.QRect(qt.QPoint(position.width(), position.height()), size)
    button.setGeometry(rect)
    button.setVisible(True)
    return button
