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

__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__license__ = "MIT"
__date__ = "03/12/2018"

import functools

from silx.gui import qt

from ..utils import units


class UnitLabel(qt.QLabel):
    """QLabel displaying an unit.
    """

    def __init__(self, parent):
        super(UnitLabel, self).__init__(parent)
        self.__unit = None
        self.__model = None
        self.__isUnitEditable = False

    def setUnit(self, unit):
        """
        Set the displayed unit.

        :param pyFAI.gui.calibration.units.Unit unit: An unit
        """
        if self.__unit is unit:
            return
        self.__unit = unit
        if unit is None:
            self.setText("")
            self.setToolTip("No unit")
        else:
            self.setText(unit.symbol)
            self.setToolTip(u"%s (%s)" % (unit.fullname, unit.symbol))

    def getUnit(self):
        """
        :rtype: pyFAI.gui.calibration.units.Unit
        """
        return self.__unit

    def setUnitModel(self, model):
        """
        Set the model containing an unit.

        :param pyFAI.gui.calibration.model.DataUnit.DataUnit: Model containing
            the unit.
        """
        if self.__model is not None:
            self.__model.changed.disconnect(self.__unitChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__unitChanged)
        self.__unitChanged()

    def __unitChanged(self):
        if self.__model is None:
            self.setUnit(None)
        else:
            self.setUnit(self.__model.value())

    def getUnitModel(self):
        """
        :rtype: pyFAI.gui.calibration.model.DataUnit.DataUnit
        """
        return self.__model

    def __popupUnitSelection(self, pos):
        """Display a popup list to allow to select a new unit"""
        if self.__unit is None:
            return

        unitList = units.Unit.get_units(self.__unit.dimensionality)
        if len(unitList) <= 1:
            return

        menu = qt.QMenu(self)
        menu.addSection("Unit for %s" % self.__unit.dimensionality.fullname.lower())

        for unit in unitList:
            action = qt.QAction(menu)
            text = u"%s: %s" % (unit.fullname, unit.symbol)
            if unit is self.__unit:
                text += " (current)"
            action.setText(text)
            action.triggered.connect(functools.partial(self.__unitSelected, unit))
            menu.addAction(action)

        menu.popup(pos)

    def __unitSelected(self, unit):
        model = self.getUnitModel()
        if model is not None:
            model.setValue(unit)
        else:
            self.setUnit(unit)

    def mouseReleaseEvent(self, event):
        if event.button() == qt.Qt.LeftButton and self.__isUnitEditable:
            pos = event.pos()
            pos = self.mapToGlobal(pos)
            self.__popupUnitSelection(pos)
            return
        super(UnitLabel, self).mouseReleaseEvent(event)

    def setUnitEditable(self, isUnitEditable):
        self.__isUnitEditable = isUnitEditable

    def isUnitEditable(self):
        return self.__isUnitEditable
