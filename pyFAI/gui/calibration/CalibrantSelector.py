# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "17/03/2017"

from pyFAI.gui import qt
import pyFAI.calibrant
from .model.CalibrantModel import CalibrantModel


class CalibrantSelector(qt.QComboBox):

    def __init__(self, parent=None):
        super(CalibrantSelector, self).__init__(parent)

        # feed the widget with default calibrants
        items = pyFAI.calibrant.CALIBRANT_FACTORY.items()
        items = sorted(items)
        for calibrantName, calibrant in items:
            self.addItem(calibrantName, calibrant)

        self.__model = None
        self.setModel(CalibrantModel())
        self.currentIndexChanged[int].connect(self.__currentIndexChanged)

    def __currentIndexChanged(self, index):
        model = self.model()
        if model is None:
            return
        item = self.itemData(index)
        old = self.blockSignals(True)
        model.setCalibrant(item)
        self.blockSignals(old)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def findCalibrant(self, calibrant):
        """Returns the first index containing the requested calibrant.
        Else return -1"""
        for index in range(self.count()):
            item = self.itemData(index)
            if item == calibrant:
                return index
        return -1

    def __modelChanged(self):
        value = self.__model.calibrant()
        if value is None:
            self.setCurrentIndex(-1)
        else:
            index = self.currentIndex()
            item = self.itemData(index)
            if item != value:
                # findData is not working
                index = self.findCalibrant(value)
                self.setCurrentIndex(index)

    def model(self):
        return self.__model
