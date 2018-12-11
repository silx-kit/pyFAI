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
__date__ = "03/12/2018"

import logging
from silx.gui import qt

_logger = logging.getLogger(__name__)


class FileEdit(qt.QLineEdit):
    """
    QLineEdit connected to a DataModel containing a file or nothing.

    It allows to edit a float value which can be nonified (by the use of an
    empty string).
    """

    def __init__(self, parent=None):
        super(FileEdit, self).__init__(parent)
        self.__model = None
        self.__applyedWhenFocusOut = True

        self.editingFinished.connect(self.__editingFinished)
        self.returnPressed.connect(self.__returnPressed)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def model(self):
        return self.__model

    def keyPressEvent(self, event):
        if event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            self.__returnPressed()
            event.accept()
        elif event.key() == qt.Qt.Key_Escape:
            self.__cancelText()
            event.accept()
        else:
            return super(FileEdit, self).keyPressEvent(event)

    def __modelChanged(self):
        self.__cancelText()

    def __editingFinished(self):
        if self.__applyedWhenFocusOut:
            self.__applyText()
        else:
            self.__cancelText()

    def __returnPressed(self):
        self.__applyText()

    def __applyText(self):
        text = self.text()

        if text.strip() == "":
            value = None
        else:
            value = text

        try:
            self.__model.setValue(value)
        except ValueError as e:
            _logger.debug(e, exc_info=True)
            self.__cancelText()

    def __cancelText(self):
        """Reset the edited value to the original one"""
        value = self.__model.value()
        if value is None:
            text = ""
        else:
            text = value
        old = self.blockSignals(True)
        self.setText(text)
        self.blockSignals(old)

    def isApplyedWhenFocusOut(self):
        return self.__applyedWhenFocusOut

    def setApplyedWhenFocusOut(self, isApplyed):
        self.__applyedWhenFocusOut = isApplyed

    applyedWhenFocusOut = qt.Property(bool, isApplyedWhenFocusOut, setApplyedWhenFocusOut)
    """Apply the current edited value to the widget when it lose the
    focus. By default the previous value is displayed.
    """
