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
__date__ = "20/08/2018"

from silx.gui import qt


class UnitLabel(qt.QLabel):
    """QLabel displaying an unit.
    """

    def __init__(self, parent):
        super(qt.QLabel, self).__init__(parent)
        self.__unit = None
        self.__model = None

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
            self.setToolTip("%s (%s)" % (unit.fullname, unit.symbol))

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
