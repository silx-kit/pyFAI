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
__date__ = "16/10/2020"

from silx.gui import qt
from .AllDetectorItemModel import AllDetectorItemModel


class DetectorFilterProxyModel(qt.QSortFilterProxyModel):

    def __init__(self, parent):
        super(DetectorFilterProxyModel, self).__init__(parent)
        self.__manufacturerFilter = None

    def setManufacturerFilter(self, manufacturer):
        if self.__manufacturerFilter == manufacturer:
            return
        self.__manufacturerFilter = manufacturer
        self.invalidateFilter()

    def filterAcceptsRow(self, sourceRow, sourceParent):
        if self.__manufacturerFilter == "*":
            return True
        sourceModel = self.sourceModel()
        index = sourceModel.index(sourceRow, 0, sourceParent)
        manufacturer = index.data(AllDetectorItemModel.MANUFACTURER_ROLE)
        return manufacturer == self.__manufacturerFilter

    def indexFromDetector(self, detector, manufacturer):
        sourceModel = self.sourceModel()
        index = sourceModel.indexFromDetector(detector, manufacturer)
        index = self.mapFromSource(index)
        return index
