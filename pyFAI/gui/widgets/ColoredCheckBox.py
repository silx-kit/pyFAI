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
__date__ = "05/04/2019"

import logging

from silx.gui import qt


_logger = logging.getLogger(__name__)


class ColoredCheckBox(qt.QCheckBox):
    """Check box with an explict API to change the background color of the
    indicator.
    """

    def __init__(self, parent=None):
        super(ColoredCheckBox, self).__init__(parent=parent)
        self.__color = None

    def paintEvent(self, event):
        painter = qt.QPainter(self)
        style = qt.QApplication.style()

        palette = qt.QPalette(self.palette())
        option = qt.QStyleOptionButton()
        if self.__color is not None:
            palette.setBrush(qt.QPalette.Normal, qt.QPalette.Base, self.__color)
            palette.setBrush(qt.QPalette.Disabled, qt.QPalette.Base, self.__color)
            palette.setBrush(qt.QPalette.Inactive, qt.QPalette.Base, self.__color)
        self.initStyleOption(option)
        option.palette = palette

        painter.save()
        style.drawPrimitive(qt.QStyle.PE_IndicatorCheckBox, option, painter, self)
        painter.restore()

    def boxColor(self):
        return self.__color

    def setBoxColor(self, color):
        self.__color = color
