# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2019 European Synchrotron Radiation Facility
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

from silx.gui import icons

import pyFAI.utils
from ..utils import units
from silx.gui import qt
from ..widgets.UnitLabel import UnitLabel
from ..widgets.QuantityEdit import QuantityEdit
from ..model.DataModel import DataModel
from ..utils import eventutils
from ..utils import validators


class ConstraintsPopup(qt.QFrame):

    def __init__(self, parent=None):
        super(ConstraintsPopup, self).__init__(parent=parent)
        qt.loadUi(pyFAI.utils.get_ui_file("constraint-drop.ui"), self)
        validator = validators.AdvancedDoubleValidator(self)
        validator.setAllowEmpty(True)
        self.__useDefaultMin = False
        self.__useDefaultMax = False
        self.__min = DataModel(self)
        self.__max = DataModel(self)
        self._minEdit.setValidator(validator)
        self._minEdit.setModel(self.__min)
        self._maxEdit.setValidator(validator)
        self._maxEdit.setModel(self.__max)
        self.__defaultConstraints = None
        self._resetMin.clicked.connect(self.__resetMin)
        self._resetMax.clicked.connect(self.__resetMax)
        self._slider.sigValueChanged.connect(self.__sliderValuesChanged)

    def __resetMin(self):
        range_ = self.__defaultConstraints.range()
        if range_ is None:
            value = None
        else:
            value = range_[0]
        self.__min.setValue(value)
        if value > self.__max.value():
            self.__max.setValue(value)
        self.__useDefaultMin = True
        self.__updateData()

    def __resetMax(self):
        range_ = self.__defaultConstraints.range()
        if range_ is None:
            value = None
        else:
            value = range_[1]
        self.__max.setValue(value)
        if self.__min.value() > value:
            self.__min.setValue(value)
        self.__useDefaultMax = True
        self.__updateData()

    def setLabel(self, text):
        self._quantity.setText(text)

    def setUnits(self, internalUnit, displayedUnit):
        if isinstance(internalUnit, DataModel):
            internalUnit = internalUnit.value()
        if isinstance(displayedUnit, units.Unit):
            model = DataModel()
            model.setValue(displayedUnit)
            displayedUnit = model

        # TODO Not the best way to do it
        # It would be better to swap the widgets
        if internalUnit.direction != displayedUnit.value().direction:
            self._leftSign.setText(u"≥")
            self._rightSign.setText(u"≥")
        else:
            self._leftSign.setText(u"≤")
            self._rightSign.setText(u"≤")

        self._minEdit.setModelUnit(internalUnit)
        self._minEdit.setDisplayedUnitModel(displayedUnit)
        self._minEdit.sigValueAccepted.connect(self.__validateMinConstraint)
        self._maxEdit.setModelUnit(internalUnit)
        self._maxEdit.setDisplayedUnitModel(displayedUnit)
        self._maxEdit.sigValueAccepted.connect(self.__validateMaxConstraint)
        self._unit.setUnit(displayedUnit.value())

    def __validateMinConstraint(self):
        self.__useDefaultMin = False
        if self.__min.value() > self.__max.value():
            self.__min.setValue(self.__max.value())
        self.__updateData()

    def __validateMaxConstraint(self):
        self.__useDefaultMax = False
        if self.__min.value() > self.__max.value():
            self.__max.setValue(self.__min.value())
        self.__updateData()

    def __sliderValuesChanged(self, first, second):
        self.__min.setValue(first)
        self.__max.setValue(second)
        if self.__defaultConstraints is not None:
            vRange = self.__defaultConstraints.range()
            self.__useDefaultMin = first == vRange[0]
            self.__useDefaultMax = second == vRange[1]
        self.__updateData()

    def labelCenter(self):
        pos = self._quantity.rect().center()
        pos = self._quantity.mapToParent(pos)
        return pos

    def fromConstaints(self, constraint):
        """Update the widget using a constraint model"""
        range_ = constraint.range()
        if range_ is None:
            minValue, maxValue = None, None
        else:
            minValue, maxValue = range_
        self.__min.setValue(minValue)
        self.__useDefaultMin = minValue is None
        self.__max.setValue(maxValue)
        self.__useDefaultMax = maxValue is None
        self.__updateData()

    def setMinFocus(self):
        self._minEdit.setFocus()

    def setMaxFocus(self):
        self._maxEdit.setFocus()

    def toConstraint(self, constraint):
        """UUpdate a constrain tmodel using the content of this widget"""
        minValue = self.__min.value()
        maxValue = self.__max.value()
        if self.__useDefaultMin:
            minValue = None
        if self.__useDefaultMax:
            maxValue = None
        constraint.setRangeConstraint(minValue, maxValue)

    def setDefaultConstraints(self, model):
        self.__defaultConstraints = model
        self.__updateData()

        vRange = model.range()
        if vRange is not None:
            self._slider.setRange(vRange[0], vRange[1])

    _DEFAULT_CONSTRAINT_STYLE = ".QuantityEdit { color: #BBBBBB; qproperty-toolTip: 'Default constraint'}"
    _CUSTOM_CONSTRAINT_STYLE = ".QuantityEdit { color: #000000; qproperty-toolTip: 'Custom constraint'}"

    def __updateData(self):

        # Update values
        if self.__defaultConstraints is None:
            return
        if self.__useDefaultMin:
            value = self.__defaultConstraints.range()
            if value is not None:
                value = value[0]
            self.__min.setValue(value)
            minStyle = self._DEFAULT_CONSTRAINT_STYLE
            self._resetMin.setEnabled(False)
        else:
            minStyle = self._CUSTOM_CONSTRAINT_STYLE
            self._resetMin.setEnabled(True)
        if self.__useDefaultMax:
            value = self.__defaultConstraints.range()
            if value is not None:
                value = value[1]
            self.__max.setValue(value)
            maxStyle = self._DEFAULT_CONSTRAINT_STYLE
            self._resetMax.setEnabled(False)
        else:
            maxStyle = self._CUSTOM_CONSTRAINT_STYLE
            self._resetMax.setEnabled(True)
        self._minEdit.setStyleSheet(minStyle)
        self._maxEdit.setStyleSheet(maxStyle)

        # Update slider
        vMin, vMax = self.__min.value(), self.__max.value()
        # TODO: The slider should use the displayedUnit, and not the internal unit
        # Here the slider for the energy is not linear
        if vMin is not None and vMax is not None:
            old = self._slider.blockSignals(True)
            if self.__defaultConstraints is not None:
                vRange = self._slider.getRange()
                if vRange is not None:
                    updated = False
                    vRange = list(vRange)
                    if vMin < vRange[0]:
                        vRange[0] = vMin
                        updated = True
                    if vMax > vRange[1]:
                        vRange[1] = vMax
                        updated = True
                    if updated:
                        self._slider.setRange(vRange[0], vRange[1])
            self._slider.setValues(vMin, vMax)
            self._slider.blockSignals(old)


class FitParamView(qt.QObject):

    sigValueAccepted = qt.Signal()
    """Emitted when a quantity was accepted."""

    _iconVariableFixed = None
    _iconVariableConstrained = None
    _iconVariableConstrainedOut = None
    _iconConstraintMin = None
    _iconConstraintNoMin = None
    _iconConstraintMax = None
    _iconConstraintNoMax = None

    def __init__(self, parent, label, internalUnit, displayedUnit=None):
        qt.QObject.__init__(self, parent=parent)
        self.__label = label
        self.__labelWidget = qt.QLabel(parent)
        self.__labelWidget.setText("%s:" % label)
        self.__quantity = QuantityEdit(parent)
        self.__quantity.setAlignment(qt.Qt.AlignRight)
        self.__quantity.sigValueAccepted.connect(self.__fireValueAccepted)
        self.__unit = UnitLabel(parent)
        self.__unit.setUnitEditable(True)
        self.__min = qt.QToolButton(parent)
        self.__min.setFixedWidth(12)
        self.__min.setAutoRaise(True)
        self.__min.clicked.connect(self.__dropContraintsOnMin)
        self.__max = qt.QToolButton(parent)
        self.__max.setAutoRaise(True)
        self.__max.setFixedWidth(12)
        self.__max.clicked.connect(self.__dropContraintsOnMax)
        self.__defaultConstraintsModel = None

        self.__subLayout = qt.QHBoxLayout()
        self.__subLayout.setSpacing(0)
        self.__subLayout.setContentsMargins(0, 0, 0, 0)
        self.__subLayout.addWidget(self.__min)
        self.__subLayout.addWidget(self.__quantity)
        self.__subLayout.addWidget(self.__max)

        if displayedUnit is None:
            displayedUnit = internalUnit

        self.__quantity.setModelUnit(internalUnit)

        if isinstance(displayedUnit, units.Unit):
            model = DataModel()
            model.setValue(displayedUnit)
            displayedUnit = model
        elif isinstance(displayedUnit, DataModel):
            pass
        else:
            raise TypeError("Unsupported type %s" % type(displayedUnit))
        self.__units = internalUnit, displayedUnit
        self.__unit.setUnitModel(displayedUnit)
        self.__quantity.setDisplayedUnitModel(displayedUnit)

        self.__constraints = qt.QToolButton(parent)
        self.__constraints.setAutoRaise(True)
        self.__constraints.clicked.connect(self.__constraintsClicked)
        self.__model = None
        self.__constraintsModel = None

        if self._iconVariableFixed is None:
            self._iconVariableFixed = icons.getQIcon("pyfai:gui/icons/variable-fixed")
        if self._iconVariableConstrained is None:
            self._iconVariableConstrained = icons.getQIcon("pyfai:gui/icons/variable-constrained")
        if self._iconVariableConstrainedOut is None:
            self._iconVariableConstrainedOut = icons.getQIcon("pyfai:gui/icons/variable-constrained-out")
        if self._iconConstraintMin is None:
            self._iconConstraintMin = icons.getQIcon("pyfai:gui/icons/constraint-min")
        if self._iconConstraintNoMin is None:
            self._iconConstraintNoMin = icons.getQIcon("pyfai:gui/icons/constraint-no-min")
        if self._iconConstraintMax is None:
            self._iconConstraintMax = icons.getQIcon("pyfai:gui/icons/constraint-max")
        if self._iconConstraintNoMax is None:
            self._iconConstraintNoMax = icons.getQIcon("pyfai:gui/icons/constraint-no-max")

    def __fireValueAccepted(self):
        self.sigValueAccepted.emit()

    def __createDropConstraint(self):
        popup = ConstraintsPopup(self.__quantity)
        popup.setWindowFlags(qt.Qt.Popup)
        popup.setAttribute(qt.Qt.WA_DeleteOnClose)
        eventutils.createCloseSignal(popup)
        popup.sigClosed.connect(self.__constraintsPopupClosed)
        self.__constraintPopup = popup

        popup.setLabel(self.__label)
        popup.fromConstaints(self.__constraintsModel)
        popup.setDefaultConstraints(self.__defaultConstraintsModel)
        popup.setUnits(*self.__units)

        popup.updateGeometry()
        # force the update of the geometry
        popup.show()

        popupParent = self.__quantity
        pos = popupParent.mapToGlobal(popupParent.rect().center())
        pos = pos - popup.labelCenter()

        # Make sure the popup is fully inside the screen
        # FIXME: It have to be tested with multi screen
        wid = self.__quantity.winId()
        window = qt.QWindow.fromWinId(wid)
        screen = window.screen()
        screen = screen.virtualGeometry()
        rect = popup.rect()
        rect.moveTopLeft(pos)
        if not screen.contains(rect):
            pos -= qt.QPoint(rect.right() - screen.right(), 0)

        popup.move(pos)
        popup.show()
        return popup

    def __dropContraintsOnMin(self):
        popup = self.__createDropConstraint()
        popup.setMinFocus()

    def __dropContraintsOnMax(self):
        popup = self.__createDropConstraint()
        popup.setMaxFocus()

    def __constraintsPopupClosed(self):
        self.__constraintPopup.toConstraint(self.__constraintsModel)
        self.__constraintPopup = None

    def model(self):
        return self.__model

    def setModel(self, model):
        self.__quantity.setModel(model)
        self.__model = model

    def setConstraintsModel(self, model):
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.disconnect(self.__constraintsModelChanged)
        self.__constraintsModel = model
        if self.__constraintsModel is not None:
            self.__constraintsModel.changed.connect(self.__constraintsModelChanged)
            self.__constraintsModelChanged()
        self.__updateConstraintsLookAndFeel()

    def setDefaultConstraintsModel(self, model):
        self.__defaultConstraintsModel = model

    def __constraintsModelChanged(self):
        self.__updateConstraintsLookAndFeel()

    def __updateConstraintsLookAndFeel(self):
        constraint = self.__constraintsModel
        if constraint.isFixed():
            icon = self._iconVariableFixed
            minIcon = self._iconConstraintNoMin
            maxIcon = self._iconConstraintNoMax
        else:
            icon = self._iconVariableConstrained
            if constraint.isRangeConstrained():
                minValue, maxValue = constraint.range()
            else:
                minValue, maxValue = None, None
            if minValue is not None:
                minIcon = self._iconConstraintMin
            else:
                minIcon = self._iconConstraintNoMin
            if maxValue is not None:
                maxIcon = self._iconConstraintMax
            else:
                maxIcon = self._iconConstraintNoMax

        self.__constraints.setIcon(icon)
        self.__min.setIcon(minIcon)
        self.__max.setIcon(maxIcon)

    def __constraintsClicked(self):
        constraint = self.__constraintsModel
        # FIXME implement popup with range
        constraint.setFixed(not constraint.isFixed())

    def widgets(self):
        return [self.__labelWidget, self.__subLayout, self.__unit, self.__constraints]
