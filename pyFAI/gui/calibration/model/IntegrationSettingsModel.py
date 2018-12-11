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
__date__ = "27/08/2018"

from pyFAI import units
from .AbstractModel import AbstractModel
from .DataModel import DataModel


class IntegrationSettingsModel(AbstractModel):

    def __init__(self, parent=None):
        super(IntegrationSettingsModel, self).__init__(parent)
        self.__radialUnit = DataModel()
        self.__radialUnit.setValue(units.TTH_RAD)
        self.__radialUnit.changed.connect(self.wasChanged)
        self.__nPointsRadial = DataModel()
        self.__nPointsRadial.changed.connect(self.wasChanged)
        self.__nPointsAzimuthal = DataModel()
        self.__nPointsAzimuthal.changed.connect(self.wasChanged)

    def isValid(self):
        if self.__radialUnit.value() is None:
            return False
        return True

    def radialUnit(self):
        return self.__radialUnit

    def nPointsRadial(self):
        return self.__nPointsRadial

    def nPointsAzimuthal(self):
        return self.__nPointsAzimuthal
