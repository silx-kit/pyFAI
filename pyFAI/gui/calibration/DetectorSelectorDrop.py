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

import logging

from silx.gui import qt

import pyFAI.utils
import pyFAI.detectors
from .DetectorModel import AllDetectorModel
from .DetectorModel import DetectorFilter

_logger = logging.getLogger(__name__)


class DetectorSelectorDrop(qt.QWidget):

    _ManufacturerRole = qt.Qt.UserRole

    def __init__(self, parent):
        super(DetectorSelectorDrop, self).__init__(parent)
        qt.loadUi(pyFAI.utils.get_ui_file("detector-selection-drop.ui"), self)

        self.__detector = None

        model = self.__createManufacturerModel()
        self._manufacturerList.setModel(model)
        selection = self._manufacturerList.selectionModel()
        selection.selectionChanged.connect(self.__manufacturerChanged)

        model = AllDetectorModel(self)
        modelFilter = DetectorFilter(self)
        modelFilter.setSourceModel(model)
        self._modelList.setModel(modelFilter)

    def setDetector(self, detector):
        if self.__detector == detector:
            return
        self.__detector = detector
        if self.__detector is None:
            self.__selectNoDetector()
        else:
            self.__selectRegistreredDetector(detector)

    def detector(self):
        classDetector = self.currentDetectorClass()
        if classDetector is None:
            return None
        return classDetector()

    def __createManufacturerModel(self):
        manufacturers = set([])
        for detector in pyFAI.detectors.ALL_DETECTORS.values():
            manufacturers.add(detector.MANUFACTURER)

        hasOther = None in manufacturers
        manufacturers.remove(None)
        manufacturers = sorted(list(manufacturers))

        model = qt.QStandardItemModel()

        item = qt.QStandardItem("All")
        item.setData("*", role=self._ManufacturerRole)
        model.appendRow(item)

        # TODO rework this thing with a delegate
        separator = qt.QStandardItem("                  ")
        separator.setSelectable(False)
        separator.setEnabled(False)
        stricked = separator.font()
        stricked.setStrikeOut(True)
        separator.setFont(stricked)
        model.appendRow(separator)

        for manufacturer in manufacturers:
            item = qt.QStandardItem(manufacturer)
            item.setData(manufacturer, role=self._ManufacturerRole)
            model.appendRow(item)

        if hasOther:
            item = qt.QStandardItem("Other")
            item.setData(None, role=self._ManufacturerRole)
            model.appendRow(item)

        return model

    def __selectNoDetector(self):
        self.__setManufacturer("*")

    def __selectRegistreredDetector(self, detector):
        self.__setManufacturer(detector.MANUFACTURER)
        model = self._modelList.model()
        index = model.indexFromDetector(detector.__class__)
        selection = self._modelList.selectionModel()
        selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
        self._modelList.scrollTo(index)

    def __setManufacturer(self, manufacturer):
        model = self._manufacturerList.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            storedManufacturer = model.data(index, role=self._ManufacturerRole)
            if manufacturer == storedManufacturer:
                selection = self._manufacturerList.selectionModel()
                selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
                self._manufacturerList.scrollTo(index)
                return

    def currentManufacturer(self):
        indexes = self._manufacturerList.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._manufacturerList.model()
        return model.data(index, role=self._ManufacturerRole)

    def currentDetectorClass(self):
        indexes = self._modelList.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._modelList.model()
        return model.data(index, role=AllDetectorModel.CLASS_ROLE)

    def __manufacturerChanged(self, selected, deselected):
        manufacturer = self.currentManufacturer()
        model = self._modelList.model()
        model.setManufacturerFilter(manufacturer)
