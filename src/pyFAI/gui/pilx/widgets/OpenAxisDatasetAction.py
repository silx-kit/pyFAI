#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Loïc Huder (loic.huder@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Tool to visualize diffraction maps."""

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/03/2024"
__status__ = "development"

from silx.gui import qt
import silx.gui.icons
from silx.gui.dialog.DataFileDialog import DataFileDialog
from silx.io.url import DataUrl


class OpenAxisDatasetAction(qt.QAction):
    datasetOpened = qt.Signal(DataUrl)

    def __init__(self, parent=None):
        super().__init__(
            icon=silx.gui.icons.getQIcon("axis"),
            text="Open axis dataset",
            parent=parent,
        )
        self._file_directory = None
        self.setToolTip("Change axis values to motor positions")
        self.triggered[bool].connect(self._onTrigger)

    def _onTrigger(self):
        dialog = DataFileDialog(self.parentWidget())
        #       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ this line triggers an exception later on ... TODO: investigate
        dialog.setWindowTitle("Open the dataset containing X values")
        dialog.setFilterMode(DataFileDialog.FilterMode.ExistingDataset)
        if self._file_directory is not None:
            dialog.setDirectory(self._file_directory)

        result = dialog.exec()
        if not result:
            return

        self.datasetOpened.emit(dialog.selectedDataUrl())

    def setFileDirectory(self, file_directory: str):
        self._file_directory = file_directory
