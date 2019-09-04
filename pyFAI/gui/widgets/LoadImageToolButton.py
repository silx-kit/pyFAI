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
__date__ = "28/02/2019"

import fabio
import os
import logging

_logger = logging.getLogger(__name__)

from silx.gui import qt

from ..model.ImageModel import ImageFilenameModel
from ..model.ImageModel import ImageFromFilenameModel
from ..ApplicationContext import ApplicationContext
from ..utils.FilterBuilder import FilterBuilder


class _LoadImageFromFileDialogAction(qt.QAction):
    """Action loading an image using the default `QFileDialog`."""

    def __init__(self, parent):
        qt.QAction.__init__(self, parent)
        self.triggered.connect(self.__execute)
        self.setText("Use file dialog")

    def __execute(self):
        previousFile = self.parent().model().filename()
        if previousFile is None:
            previousFile = os.getcwd()

        context = ApplicationContext.instance()
        dialog = context.createFileDialog(self.parent(),
                                          previousFile=previousFile)

        builder = FilterBuilder()
        builder.addImageFormat("EDF image files", "edf edf.gz")
        builder.addImageFormat("TIFF image files", "tif tiff tif.gz tiff.gz")
        builder.addImageFormat("NumPy binary files", "npy")
        builder.addImageFormat("CBF files", "cbf")
        builder.addImageFormat("MarCCD image files", "mccd")
        builder.addImageFormat("Fit2D mask files", "msk")
        dialog.setNameFilters(builder.getFilters())

        dialog.setWindowTitle(self.parent().dialogTitle())
        dialog.setModal(True)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)

        result = dialog.exec_()
        if result:
            filename = dialog.selectedFiles()[0]
            if self.parent()._isDataSupported():
                try:
                    with fabio.open(filename) as image:
                        data = image.data
                except Exception as e:
                    message = "Filename '%s' not supported.<br />%s", (filename, str(e))
                    qt.QMessageBox.critical(self, "Loading image error", message)
                    _logger.error("Error while loading '%s'" % filename)
                    _logger.debug("Backtrace", exc_info=True)
                    return
            else:
                data = None

            self.parent()._setValue(filename=filename, data=data)


class _LoadImageFromImageDialogAction(qt.QAction):
    """Action loading an image using the silx ImageFileDialog."""

    def __init__(self, parent):
        qt.QAction.__init__(self, parent)
        self.triggered.connect(self.__execute)
        self.setText("Use image dialog")

    def __execute(self):
        previousFile = self.parent().model().filename()

        context = ApplicationContext.instance()
        dialog = context.createImageFileDialog(self.parent(),
                                               previousFile=previousFile)
        dialog.setWindowTitle(self.parent().dialogTitle())
        dialog.setModal(True)

        result = dialog.exec_()
        if result:
            url = dialog.selectedUrl()
            data = dialog.selectedImage()
            self.parent()._setValue(filename=url, data=data)


class LoadImageToolButton(qt.QToolButton):
    """ToolButton updating a DataModel using the selection from a file dialog.
    """

    def __init__(self, parent=None):
        super(LoadImageToolButton, self).__init__(parent)
        self.__model = None
        self.__isEnabled = True
        self.__dialogTitle = "Select an image"

        loadFileAction = _LoadImageFromFileDialogAction(self)
        loadImageAction = _LoadImageFromImageDialogAction(self)

        self.addAction(loadImageAction)
        self.addAction(loadFileAction)
        self.setDefaultAction(loadImageAction)

    def setEnabled(self, enabled):
        if self.__isEnabled == enabled:
            return
        self.__isEnabled = enabled
        self.__updateEnabledState()

    def isEnabled(self):
        return self.__isEnabled

    def __updateEnabledState(self):
        enabledState = self.__model is not None and self.__isEnabled
        qt.QToolButton.setEnabled(self, enabledState)

    def _setValue(self, filename, data=None):
        """Update the model with this new parameters.

        :param str filename: A filename
        :param data numpy.ndarray: The associated data
        """
        if self.__model is None:
            return
        model = self.__model
        with model.lockContext():
            if isinstance(model, ImageFilenameModel):
                model.setFilename(filename)
            elif isinstance(model, ImageFromFilenameModel):
                model.setFilename(filename)
                model.setValue(data)
                model.setSynchronized(True)
            else:
                model.setValue(filename)

    def _isDataSupported(self):
        """Returns true if the model supports the image data.

        :rtype: bool
        """
        if isinstance(self.__model, ImageFromFilenameModel):
            return True
        return False

    def filename(self):
        if self.__model is None:
            return None
        if isinstance(self.__model, ImageFromFilenameModel):
            return self.__model.filename()
        return self.__model.value()

    def setModel(self, model):
        self.__model = model
        self.__updateEnabledState()

    def model(self):
        return self.__model

    def dialogTitle(self):
        """Returns the specified dialog title

        :rtype: str
        """
        return self.__dialogTitle

    def setDialogTitle(self, title):
        """Specify a title for the dialog

        :param str title: Title of the dialog
        """
        return self.__dialogTitle
