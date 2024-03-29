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
import pyFAI.detectors


class AllDetectorItemModel(qt.QStandardItemModel):

    CLASS_ROLE = qt.Qt.UserRole
    MODEL_ROLE = qt.Qt.UserRole + 1
    MANUFACTURER_ROLE = qt.Qt.UserRole + 2

    def __init__(self, parent):
        qt.QStandardItemModel.__init__(self, parent)

        detectorClasses = set(pyFAI.detectors.ALL_DETECTORS.values())

        def getNameAndManufacturer(detectorClass):
            modelName = None
            result = []

            if hasattr(detectorClass, "MANUFACTURER"):
                manufacturer = detectorClass.MANUFACTURER
            else:
                manufacturer = None

            if isinstance(manufacturer, list):
                for index, m in enumerate(manufacturer):
                    if m is None:
                        continue
                    modelName = detectorClass.aliases[index]
                    result.append((modelName, m, detectorClass))
            else:
                if hasattr(detectorClass, "aliases"):
                    if len(detectorClass.aliases) > 0:
                        modelName = detectorClass.aliases[0]
                if modelName is None:
                    modelName = detectorClass.__name__
                result.append((modelName, manufacturer, detectorClass))
            return result

        def sortingKey(item):
            modelName, manufacturerName, _detector = item
            if modelName:
                modelName = modelName.lower()
            if manufacturerName:
                manufacturerName = manufacturerName.lower()
            return modelName, manufacturerName

        items = []
        for c in detectorClasses:
            items.extend(getNameAndManufacturer(c))
        items = sorted(items, key=sortingKey)
        for modelName, manufacturerName, detector in items:
            if detector is pyFAI.detectors.Detector:
                continue
            item = qt.QStandardItem(modelName)
            item.setData(detector, role=self.CLASS_ROLE)
            item.setData(modelName, role=self.MODEL_ROLE)
            item.setData(manufacturerName, role=self.MANUFACTURER_ROLE)
            item2 = qt.QStandardItem(manufacturerName)
            item2.setData(detector, role=self.CLASS_ROLE)
            item2.setData(modelName, role=self.MODEL_ROLE)
            item2.setData(manufacturerName, role=self.MANUFACTURER_ROLE)
            self.appendRow([item, item2])

    def indexFromDetector(self, detector, manufacturer):
        for row in range(self.rowCount()):
            index = self.index(row, 0)
            manufacturerName = self.data(index, role=self.MANUFACTURER_ROLE)
            if manufacturerName != manufacturer:
                continue
            detectorClass = self.data(index, role=self.CLASS_ROLE)
            if detectorClass != detector:
                continue
            return index
        return qt.QModelIndex()
