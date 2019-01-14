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
__date__ = "03/01/2019"

from silx.gui import qt

from ..model.DataModel import DataModel
from ...utils import stringutil


class UnitSelector(qt.QComboBox):

    def __init__(self, parent=None):
        super(UnitSelector, self).__init__(parent)

        self.__model = None
        self.setModel(DataModel())
        self.currentIndexChanged[int].connect(self.__currentIndexChanged)
        self.__shortName = False

    def setShortNameDisplay(self, shortName):
        if self.__shortName == shortName:
            return
        self.__shortName = shortName
        units = self.units()
        self.setUnits(units)

    def units(self):
        units = []
        for index in range(self.count()):
            unit = self.itemData(index)
            units.append(unit)
        return units

    def setUnits(self, units):
        previousUnit = self.__model.value()
        old = self.blockSignals(True)
        # clean up
        self.clear()

        units = sorted(list(units), key=lambda u: u.label)

        for unit in units:
            if self.__shortName:
                name = stringutil.latex_to_unicode(unit.short_name)
                symbol = unit.unit_symbol
                if symbol == "?":
                    label = name
                else:
                    symbol = stringutil.latex_to_unicode(unit.unit_symbol)
                    label = "%s (%s)" % (name, symbol)
            else:
                label = stringutil.latex_to_unicode(unit.label)
            self.addItem(label, unit)
        # try to find the previous unit in the new list
        if previousUnit is None:
            currentIndex = self.currentIndex()
            index = -1
        else:
            currentIndex = self.currentIndex()
            index = self.findUnit(previousUnit)
        if index == -1:
            self.blockSignals(old)
            if currentIndex != index:
                # the previous selected unit is not anymore available
                self.setCurrentIndex(index)
        else:
            # the previous index is found
            # we dont have to emit signals
            self.setCurrentIndex(index)
            self.blockSignals(old)

    def __currentIndexChanged(self, index):
        model = self.model()
        if model is None:
            return
        unit = self.itemData(index)
        old = self.blockSignals(True)
        model.setValue(unit)
        self.blockSignals(old)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def findUnit(self, unit):
        """Returns the first index containing the requested detector.
        Else return -1"""
        for index in range(self.count()):
            item = self.itemData(index)
            if item is unit:
                return index
        return -1

    def __modelChanged(self):
        value = self.__model.value()
        if value is None:
            self.setCurrentIndex(-1)
        else:
            unit = value
            index = self.currentIndex()
            item = self.itemData(index)
            if item is not unit:
                # findData is not working
                index = self.findUnit(unit)
                self.setCurrentIndex(index)

    def model(self):
        return self.__model
