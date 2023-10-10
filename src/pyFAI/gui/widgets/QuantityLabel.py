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
__date__ = "03/01/2019"

import numpy
import numbers
import logging
import functools

from silx.gui import qt

from ..utils import units


_logger = logging.getLogger(__name__)


class QuantityLabel(qt.QLabel):

    def __init__(self, parent=None):
        qt.QLabel.__init__(self, parent)
        self.__prefix = ""
        self.__formatter = "{value}"
        self.__floatFormatter = None
        self.__value = None
        self.__internalUnit = None
        self.__displayedUnit = None
        self.__isUnitEditable = False
        self.__displayedUnitModel = None
        self.__preferedSize = None
        self.__elasticSize = False

    def setInternalUnit(self, unit):
        self.__internalUnit = unit
        self.__updateText()

    def setDisplayedUnitModel(self, model):
        """
        Set the model containing an unit.

        :param pyFAI.gui.model.DataUnit.DataUnit: Model containing
            the unit.
        """
        if self.__displayedUnitModel is not None:
            self.__displayedUnitModel.changed.disconnect(self.__displayedUnitChanged)
        self.__displayedUnitModel = model
        if self.__displayedUnitModel is not None:
            self.__displayedUnitModel.changed.connect(self.__displayedUnitChanged)
        self.__displayedUnitChanged()

    def getDisplayedUnitModel(self):
        return self.__displayedUnitModel

    def __displayedUnitChanged(self):
        model = self.__displayedUnitModel
        if model is None:
            self.setDisplayedUnit(None)
        else:
            self.setDisplayedUnit(model.value())

    def usedUnit(self):
        """Returns the unit used to display the quantity"""
        if self.__displayedUnit is not None:
            return self.__displayedUnit
        else:
            return self.__internalUnit

    def setDisplayedUnit(self, unit):
        self.__displayedUnit = unit
        self.__updateText()

    def setPrefix(self, prefix):
        self.__prefix = prefix
        self.__updateText()

    def setFormatter(self, formatter):
        self.__formatter = formatter
        self.__updateText()

    def setFloatFormatter(self, formatter):
        """Set a specific formatter for float.

        If this formatter is None (default value) or the value is not a
        floatting point, the default formatter is used.
        """
        self.__floatFormatter = formatter
        self.__updateText()

    def setElasticSize(self, useElasticSize):
        if self.__elasticSize == useElasticSize:
            return
        self.__elasticSize = useElasticSize
        if self.__elasticSize is None:
            self.__preferedSize = None
        self.updateGeometry()

    def sizeHint(self):
        if self.__preferedSize is not None:
            return self.__preferedSize
        return qt.QLabel.sizeHint(self)

    def setValue(self, value):
        self.__value = value
        self.__updateText()

    def __updateText(self):
        if self.__value is None:
            text = "na"
        else:
            value = self.__value
            if numpy.isscalar(value) and numpy.isnan(value):
                text = "nan"
            else:
                currentUnit = self.usedUnit()
                if currentUnit is not None:
                    symbol = currentUnit.symbol
                else:
                    symbol = None

                try:
                    value = units.convert(value, self.__internalUnit, currentUnit)
                    if isinstance(value, numbers.Real) and self.__floatFormatter is not None:
                        formatter = self.__floatFormatter
                    else:
                        formatter = self.__formatter
                    text = formatter.format(value=value)
                    if symbol is not None:
                        text = text + " " + symbol
                except Exception as e:
                    _logger.error("Error while formating value: %s", e.args[0])
                    _logger.debug("Backtrace", exc_info=True)
                    text = "error"

        text = self.__prefix + text
        self.setText(text)
        if self.__elasticSize:
            size = self.size()
            if self.__preferedSize is None or self.__preferedSize.width() < size.width():
                self.__preferedSize = size

    def __popupUnitSelection(self, pos):
        """Display a popup list to allow to select a new unit"""
        currentUnit = self.usedUnit()
        if currentUnit is None:
            return

        unitList = units.Unit.get_units(currentUnit.dimensionality)
        if len(unitList) <= 1:
            return

        menu = qt.QMenu(self)
        menu.addSection("Unit for %s" % currentUnit.dimensionality.fullname.lower())

        for unit in unitList:
            action = qt.QAction(menu)
            text = "%s: %s" % (unit.fullname, unit.symbol)
            if unit is currentUnit:
                text += " (current)"
            action.setText(text)
            action.triggered.connect(functools.partial(self.__unitSelected, unit))
            menu.addAction(action)

        menu.popup(pos)

    def __unitSelected(self, unit):
        model = self.getDisplayedUnitModel()
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
        super(QuantityLabel, self).mouseReleaseEvent(event)

    def setUnitEditable(self, isUnitEditable):
        self.__isUnitEditable = isUnitEditable

    def isUnitEditable(self):
        return self.__isUnitEditable
