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
__date__ = "09/08/2018"

from silx.gui import qt
import pyFAI.detectors


class AllDetectorModel(qt.QStandardItemModel):

    CLASS_ROLE = qt.Qt.UserRole

    def __init__(self, parent):
        qt.QStandardItemModel.__init__(self, parent)

        detectorClasses = set(pyFAI.detectors.ALL_DETECTORS.values())

        def getClassModel(detectorClass):
            modelName = None
            if hasattr(detectorClass, "aliases"):
                if len(detectorClass.aliases) > 0:
                    modelName = detectorClass.aliases[0]
            if modelName is None:
                modelName = detectorClass.__name__
            return modelName

        items = [(getClassModel(c), c) for c in detectorClasses]
        items = sorted(items)
        for detectorName, detector in items:
            if detector is pyFAI.detectors.Detector:
                continue
            item = qt.QStandardItem(detectorName)
            item.setData(detector, role=self.CLASS_ROLE)
            self.appendRow(item)

    def indexFromDetector(self, detector):
        for row in range(self.rowCount()):
            index = self.index(row, 0)
            detectorClass = self.data(index, role=self.CLASS_ROLE)
            if detectorClass == detector:
                return index
        return qt.QModelIndex()


class DetectorFilter(qt.QSortFilterProxyModel):

    def __init__(self, parent):
        super(DetectorFilter, self).__init__(parent)
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
        detectorClass = index.data(AllDetectorModel.CLASS_ROLE)
        return detectorClass.MANUFACTURER == self.__manufacturerFilter

    def indexFromDetector(self, detector):
        sourceModel = self.sourceModel()
        index = sourceModel.indexFromDetector(detector)
        index = self.mapFromSource(index)
        return index
