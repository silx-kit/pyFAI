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
__date__ = "01/03/2019"

import logging

from silx.gui import qt

from ..model.DataModel import DataModel
from ..model.ImageModel import ImageFromFilenameModel
from ..model.ImageModel import ImageFilenameModel
import pyFAI.io.image


_logger = logging.getLogger(__name__)


class FileEdit(qt.QLineEdit):
    """
    QLineEdit connected to a DataModel containing a file or nothing.

    It allows to edit a float value which can be nonified (by the use of an
    empty string).
    """

    sigValueAccepted = qt.Signal()
    """Emitted when a file was accepted.

    In case the value is still the same, no signal is sent from the DataModel,
    but this signal is emitted."""

    def __init__(self, parent=None):
        super(FileEdit, self).__init__(parent)
        self.__model = None
        self.__applyedWhenFocusOut = True
        self.__previousText = None
        self.__wasModified = False

        self.editingFinished.connect(self.__editingFinished)
        self.returnPressed.connect(self.__returnPressed)

    def event(self, event):
        if event.type() == 207:
            if self.__previousText != self.text():
                # TODO: This tries to capture Linux copy-paste using middle mouse
                # button. But this event do not match exactly what it is intented.
                # None of the available events capture this special copy-paste.
                self.__wasModified = True
        return qt.QLineEdit.event(self, event)

    def focusInEvent(self, event):
        self.__previousText = self.text()
        self.__wasModified = False
        super(FileEdit, self).focusInEvent(event)

    def dragEnterEvent(self, event):
        if self.__model is not None:
            if event.mimeData().hasFormat("text/uri-list"):
                event.acceptProposedAction()
            elif event.mimeData().hasFormat("application/x-silx-uri"):
                event.acceptProposedAction()

    def dropEvent(self, event):
        mimeData = event.mimeData()
        if mimeData.hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            path = byteString.data().decode("utf-8")
        elif mimeData.hasUrls():
            urls = mimeData.urls()
            if len(urls) > 1:
                qt.QMessageBox.critical(self, "Drop cancelled", "A single file is expected")
                return
            path = urls[0].toLocalFile()
        else:
            qt.QMessageBox.critical(self, "Drop cancelled", "A file is expected")
            return

        old = self.blockSignals(True)
        self.setText(path)
        self.blockSignals(old)
        self.__applyFilename(path)

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
            result = super(FileEdit, self).keyPressEvent(event)
            if event.isAccepted():
                self.__wasModified = True
            return result

    def __modelChanged(self):
        self.__cancelText()

    def __editingFinished(self):
        if not self.__wasModified:
            self.__cancelText()
        elif self.__applyedWhenFocusOut:
            self.__applyText()
        else:
            self.__cancelText()

    def __returnPressed(self):
        self.__applyText()

    def __applyText(self):
        text = self.text()
        if text == self.__previousText:
            self.sigValueAccepted.emit()
            return

        if text.strip() == "":
            value = None
        else:
            value = text

        try:
            self.__applyFilename(value)
            # Avoid sending further signals
            self.__previousText = text
            self.sigValueAccepted.emit()
        except Exception as e:
            _logger.debug(e, exc_info=True)
            self.__cancelText()

    def __cancelText(self):
        """Reset the edited value to the original one"""
        value = self.__getFilename()
        if value is None:
            text = ""
        else:
            text = value
        old = self.blockSignals(True)
        self.setText(text)
        # Avoid sending further signals
        self.__previousText = text
        self.blockSignals(old)

    def isApplyedWhenFocusOut(self):
        return self.__applyedWhenFocusOut

    def setApplyedWhenFocusOut(self, isApplyed):
        self.__applyedWhenFocusOut = isApplyed

    applyedWhenFocusOut = qt.Property(bool, isApplyedWhenFocusOut, setApplyedWhenFocusOut)
    """Apply the current edited value to the widget when it lose the
    focus. By default the previous value is displayed.
    """

    def __getFilename(self):
        model = self.__model
        if isinstance(model, (ImageFromFilenameModel, ImageFilenameModel)):
            return self.__model.filename()
        elif isinstance(model, DataModel):
            return self.__model.value()
        else:
            assert(False)

    def __applyFilename(self, filename):
        model = self.__model
        errorInfo = None
        try:
            # Avoid reentrant signal from model
            model.changed.disconnect(self.__modelChanged)
            # Avoid focuslost if a message is displayed
            self.editingFinished.disconnect(self.__editingFinished)

            if isinstance(model, ImageFromFilenameModel):
                if filename is not None:
                    try:
                        data = pyFAI.io.image.read_image_data(filename)
                    except Exception as e:
                        message = "Filename '%s' not supported.<br />%s" % (filename, str(e))
                        title = "Loading image error"
                        errorInfo = title, message
                        _logger.error("Error while loading %s" % filename)
                        _logger.debug("Backtrace", exc_info=True)
                        data = None
                else:
                    data, filename = None, None
                with model.lockContext():
                    model.setValue(data)
                    model.setFilename(filename)
                    model.setSynchronized(True)
                if errorInfo is not None:
                    title, message = errorInfo
                    qt.QMessageBox.critical(self, title, message)

            elif isinstance(model, (DataModel, ImageFilenameModel)):
                previous = model.value()
                try:
                    model.setValue(filename)
                except Exception as e:
                    if model.value() is not previous:
                        model.setValue(previous)
                    message = "Filename '%s' not supported.<br />%s" % (filename, str(e))
                    qt.QMessageBox.critical(self, "Unsupported filename", message)
            else:
                assert(False)
        finally:
            model.changed.connect(self.__modelChanged)
            self.editingFinished.connect(self.__editingFinished)
