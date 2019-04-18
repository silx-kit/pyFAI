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
__date__ = "18/04/2019"

import logging

from silx.gui import qt


_logger = logging.getLogger(__name__)


class AdvancedComboBox(qt.QComboBox):
    """Combobox which provide a way to custom the displayed """

    def __init__(self, parent=None):
        qt.QComboBox.__init__(self, parent=parent)
        self.__displayedDataCallback = None

    def initStyleOption(self, option):
        """
        :param qt.QStyleOptionComboBox option: Option to initialize
        """
        qt.QComboBox.initStyleOption(self, option)
        dataCallback = self.__displayedDataCallback
        if dataCallback is not None:
            index = self.currentIndex()
            text = dataCallback(self, index, role=qt.Qt.DisplayRole)
            if text is None:
                text = ""
            icon = dataCallback(self, index, role=qt.Qt.DecorationRole)
            if icon is None:
                icon = qt.QIcon()
            option.currentText = text
            option.currentIcon = icon

    def paintEvent(self, event):
        """
        :param qt.QPaintEvent event: Qt event
        """
        # Inherite paint event to be able to custom initStyleOption.
        # It's a protected function that can't be hinerited by Python.
        painter = qt.QStylePainter(self)
        painter.setPen(self.palette().color(qt.QPalette.Text))
        # draw the combobox frame, focusrect and selected etc.
        opt = qt.QStyleOptionComboBox()
        self.initStyleOption(opt)
        painter.drawComplexControl(qt.QStyle.CC_ComboBox, opt)
        # draw the icon and text
        painter.drawControl(qt.QStyle.CE_ComboBoxLabel, opt)

    def setDisplayedDataCallback(self, callback):
        """
        Set a callback to custom the displayed text and icon of the displayed
        selected item.

        This was designed to be used with `setModel`, in case the cell to
        display is not part of the model provided.

        Only `qt.Qt.DisplayRole` and `qt.Qt.DecorationRole` are supported.

        .. code-block::

            def displayedData(widget, row, role=qt.Qt.DisplayRole):
                if role == qt.Qt.DisplayRole:
                    model = widget.model()
                    index0 = model.index(row, 0)
                    index1 = model.index(row, 1)
                    text = index0.data() + " " + index1.data()
                    return text
                elif role == qt.Qt.DecorationRole:
                    return None
                return None
            comboBox = AdvancedComboBox()
            comboBox.setModel(model)
            comboBox.setDisplayedDataCallback(displayedData)

        :param Callable(int,QVariant) callback: Callback a-la Qt abstract
            model `data`. Called with the row index and the role to update the
            text and the icon of the displayed item.
        """
        self.__displayedDataCallback = callback
