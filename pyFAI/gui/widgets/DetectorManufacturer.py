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
__date__ = "24/08/2018"

from silx.gui import qt
import pyFAI.detectors
from ..calibration.model.DetectorModel import DetectorModel


class DetectorManufacturer(qt.QComboBox):

    def __init__(self, parent=None):
        super(DetectorManufacturer, self).__init__(parent)

        # feed the widget with default manufacturers
        manufacturers = set([])
        for detector in pyFAI.detectors.ALL_DETECTORS.values():
            manufacturers.add(detector.MANUFACTURER)

        hasOther = None in manufacturers
        manufacturers.remove(None)
        manufacturers = sorted(list(manufacturers))

        for manufacturer in manufacturers:
            self.addItem(manufacturer, manufacturer)
        if hasOther:
            self.addItem("Others", None)
        self.insertItem(0, "Any", "*")
        self.insertSeparator(1)

        self.__model = None
        self.setModel(DetectorModel())
        self.setCurrentIndex(0)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def currentManufacturer(self):
        index = self.currentIndex()
        if index == -1:
            return "*"
        manufacturer = self.itemData(index)
        return manufacturer

    def findManufacturer(self, manufacturer):
        """Returns the first index containing the requested manufacturer.
        Else return -1"""
        for index in range(self.count()):
            item = self.itemData(index)
            if item == "*":
                continue
            if item is manufacturer:
                return index
        # TODO we are supposed to return other group
        # or create a new manufacturer
        return -1

    def __modelChanged(self):
        value = self.__model.detector()
        if value is None:
            self.setCurrentIndex(-1)
        else:
            manufacturer = value.MANUFACTURER
            index = self.currentIndex()
            item = self.itemData(index)
            if item != manufacturer:
                # findData is not working
                index = self.findManufacturer(manufacturer)
                self.setCurrentIndex(index)

    def model(self):
        return self.__model
