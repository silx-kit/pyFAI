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
from silx.io.url import DataUrl

_logger = logging.getLogger(__name__)


class MonitorNameEdit(qt.QLineEdit):
    """
    QLineEdit specialied to select a monitor name from HDF5 file.
    """

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-silx-uri"):
            event.acceptProposedAction()

    def dropEvent(self, event):
        mimeData = event.mimeData()
        if mimeData.hasFormat("application/x-silx-uri"):
            byteString = event.mimeData().data("application/x-silx-uri")
            path = byteString.data().decode("utf-8")
            url = DataUrl(path)
            self.setText(url.data_path())
        else:
            qt.QMessageBox.critical(self, "Drop cancelled", "A HDF5 path is expected")
            return
