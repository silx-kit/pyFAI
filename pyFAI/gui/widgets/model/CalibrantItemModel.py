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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "25/06/2023"

import os.path
from silx.gui import qt
from silx.gui import icons
import pyFAI.calibrant
from pyFAI.calibrant import Calibrant


class CalibrantItemModel(qt.QStandardItemModel):

    CALIBRANT_ROLE = qt.Qt.UserRole

    def __init__(self, parent=None):
        qt.QStandardItemModel.__init__(self, parent=parent)

        calibrants = pyFAI.calibrant.CALIBRANT_FACTORY.items()
        calibrants = sorted(calibrants, key=lambda x: x[0].lower())
        for calibrantName, calibrant in calibrants:
            item = self.createStandardItem(calibrant, calibrantName)
            self.appendRow(item)

    def createStandardItem(self, calibrant, calibrantName=None) -> qt.QStandardItem:
        if calibrantName is None:
            name = os.path.splitext(os.path.basename(calibrant.filename))[0]
            calibrantName = name
        item = qt.QStandardItem()
        item.setText(calibrantName)
        item.setToolTip(calibrant.filename)
        item.setData(calibrant, role=self.CALIBRANT_ROLE)
        if calibrant.filename is None or calibrant.filename.startswith("pyfai:"):
            icon = icons.getQIcon("pyfai:gui/icons/calibrant")
        else:
            icon = icons.getQIcon("pyfai:gui/icons/calibrant-custom")

        item.setIcon(icon)
        item.setEditable(False)
        return item

    def indexFromCalibrant(self, calibrant: Calibrant):
        for row in range(self.rowCount()):
            index = self.index(row, 0)
            calibrantObj = self.data(index, role=self.CALIBRANT_ROLE)
            if calibrant.filename == calibrantObj.filename:
                return index
        return qt.QModelIndex()

    def appendCalibrant(self, calibrant) -> qt.QModelIndex:
        item = self.createStandardItem(calibrant)
        self.appendRow(item)
