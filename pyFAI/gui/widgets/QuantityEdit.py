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
__date__ = "15/05/2019"

import logging
from silx.gui import qt
from ..utils import validators
from ..utils import units

_logger = logging.getLogger(__name__)


class QuantityEdit(qt.QLineEdit):
    """
    QLineEdit connected to a DataModel.

    It allows to edit a float value which can be nonified (by the use of an
    empty string).
    """

    sigValueAccepted = qt.Signal()
    """Emitted when a quantity was accepted.

    In case the value is still the same, no signal is sent from the DataModel,
    but this signal is emitted."""

    def __init__(self, parent=None):
        super(QuantityEdit, self).__init__(parent)
        validator = validators.AdvancedDoubleValidator(self)
        validator.setAllowEmpty(True)
        self.setValidator(validator)

        self.__wasModified = False
        self.__model = None
        self.__applyedWhenFocusOut = True
        self.__modelUnit = None
        self.__displayedUnit = None
        self.__displayedUnitModel = None
        self.__previousText = None

        self.editingFinished.connect(self.__editingFinished)
        self.returnPressed.connect(self.__returnPressed)

    def event(self, event):
        if event.type() == 207:
            if self.__previousText != self.text():
                # TODO: This tries to capture Linux copy-paste using middle mouse
                # button. But this event do not match exactly what it is intented.
                # None of the available events capture this special copy-paste.
                self.__wasModified = True
        return qt.QLineEdit.event(self, event)

    def focusInEvent(self, event):
        self.__previousText = self.text()
        self.__wasModified = False
        super(QuantityEdit, self).focusInEvent(event)

    def setModel(self, model):
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def model(self):
        return self.__model

    def setDisplayedUnit(self, displayedUnit):
        """
        Set the displayed unit.

        :param pyFAI.gui.units.Unit displayedUnit: An unit
        """
        if self.__displayedUnit is displayedUnit:
            return
        self.__displayedUnit = displayedUnit
        self.__cancelText()

    def getDisplayedUnit(self):
        """
        :rtype: pyFAI.gui.units.Unit
        """
        return self.__displayedUnit

    def setDisplayedUnitModel(self, displayedUnitModel):
        """
        Set the displayed unit.

        :param pyFAI.gui.units.Unit displayedUnitModel: A model containing a unit
        """
        if self.__displayedUnitModel is not None:
            self.__displayedUnitModel.changed.disconnect(self.__displayedUnitModelChanged)
        self.__displayedUnitModel = displayedUnitModel
        if self.__displayedUnitModel is not None:
            self.__displayedUnitModel.changed.connect(self.__displayedUnitModelChanged)
        self.__displayedUnitModelChanged()

    def __displayedUnitModelChanged(self):
        model = self.__displayedUnitModel
        if model is None:
            self.setDisplayedUnit(None)
        else:
            self.setDisplayedUnit(model.value())

    def getDisplayedUnitModel(self):
        """
        :rtype: pyFAI.gui.model.DataModel.DataModel
        """
        return self.__displayedUnitModel

    def setModelUnit(self, modelUnit):
        """
        Set the unit of the stroed data in the model.

        :param pyFAI.gui.units.Unit unit: An unit
        """
        if self.__modelUnit is modelUnit:
            return
        self.__modelUnit = modelUnit
        self.__cancelText()

    def getModelUnit(self):
        """
        :rtype: pyFAI.gui.units.Unit
        """
        return self.__modelUnit

    def keyPressEvent(self, event):
        if event.key() in (qt.Qt.Key_Return, qt.Qt.Key_Enter):
            self.__returnPressed()
            event.accept()
        elif event.key() == qt.Qt.Key_Escape:
            self.__cancelText()
            event.accept()
        else:
            result = super(QuantityEdit, self).keyPressEvent(event)
            if event.isAccepted():
                self.__wasModified = True
            return result

    def __modelChanged(self):
        self.__cancelText()

    def __editingFinished(self):
        if not self.__wasModified:
            self.__cancelText()
        elif self.__applyedWhenFocusOut:
            self.__applyText()
        else:
            self.__cancelText()

    def __returnPressed(self):
        self.__applyText()

    def __getUnits(self):
        """Returns internal unit and displayed unit"""
        return self.__modelUnit, self.__displayedUnit

    def __convertToModel(self, value):
        internalUnit, displayedUnit = self.__getUnits()
        if internalUnit is displayedUnit:
            return value
        return units.convert(value, displayedUnit, internalUnit)

    def __convertToDisplay(self, value):
        internalUnit, displayedUnit = self.__getUnits()
        if internalUnit is displayedUnit:
            return value
        if internalUnit is None:
            return value
        if displayedUnit is None:
            return value
        return units.convert(value, internalUnit, displayedUnit)

    def __applyText(self):
        text = self.text()
        if text == self.__previousText:
            self.sigValueAccepted.emit()
            return
        validator = self.validator()
        value, validated = validator.toValue(text)
        try:
            if validated:
                value = self.__convertToModel(value)
                self.__model.setValue(value)
                # Avoid sending further signals
                self.__previousText = text
                self.sigValueAccepted.emit()
            else:
                self.__cancelText()
        except ValueError as e:
            _logger.debug(e, exc_info=True)
            self.__cancelText()

    def __cancelText(self):
        """Reset the edited value to the original one"""
        if self.__model is None:
            text = ""
        else:
            value = self.__model.value()
            value = self.__convertToDisplay(value)
            validator = self.validator()
            text = validator.toText(value)
        old = self.blockSignals(True)
        self.setText(text)
        # For very big numbers, make the begining visible
        self.setCursorPosition(0)
        # Avoid sending further signals
        self.__previousText = text
        self.blockSignals(old)

    def isApplyedWhenFocusOut(self):
        return self.__applyedWhenFocusOut

    def setApplyedWhenFocusOut(self, isApplyed):
        self.__applyedWhenFocusOut = isApplyed

    applyedWhenFocusOut = qt.Property(bool, isApplyedWhenFocusOut, setApplyedWhenFocusOut)
    """Apply the current edited value to the widget when it lose the
    focus. By default the previous value is displayed.
    """
