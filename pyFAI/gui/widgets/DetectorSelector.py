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
from ..calibration.model.DetectorModel import DetectorModel
from .DetectorModel import AllDetectorModel
from .DetectorModel import DetectorFilter


class DetectorSelector(qt.QComboBox):

    def __init__(self, parent=None):
        super(DetectorSelector, self).__init__(parent)

        # feed the widget with default detectors
        model = AllDetectorModel(self)
        self.__filter = DetectorFilter(self)
        self.__filter.setSourceModel(model)

        super(DetectorSelector, self).setModel(self.__filter)

        self.__model = None
        self.setModel(DetectorModel())
        self.currentIndexChanged[int].connect(self.__currentIndexChanged)

    def setManufacturerFilter(self, manufacturer):
        self.__filter.setManufacturerFilter(manufacturer)

    def __currentIndexChanged(self, index):
        model = self.model()
        if model is None:
            return
        detectorClass = self.itemData(index)
        if detectorClass is not None:
            detector = detectorClass()
        else:
            detector = None
        old = self.blockSignals(True)
        model.setDetector(detector)
        self.blockSignals(old)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def findDetectorClass(self, detectorClass):
        """Returns the first index containing the requested detector.
        Else return -1"""
        for index in range(self.count()):
            item = self.itemData(index)
            if item is detectorClass:
                return index
        return -1

    def __modelChanged(self):
        value = self.__model.detector()
        if value is None:
            self.setCurrentIndex(-1)
        else:
            detectorClass = value.__class__
            index = self.currentIndex()
            item = self.itemData(index)
            if item != detectorClass:
                # findData is not working
                index = self.findDetectorClass(detectorClass)
                self.setCurrentIndex(index)

    def model(self):
        return self.__model
