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
__date__ = "21/02/2019"

from .AbstractModel import AbstractModel
from .DataModel import DataModel


class Fit2dGeometryModel(AbstractModel):

    def __init__(self, parent=None):
        super(Fit2dGeometryModel, self).__init__(parent)
        self.__distance = DataModel()
        self.__centerX = DataModel()
        self.__centerY = DataModel()
        self.__tilt = DataModel()
        self.__tiltPlan = DataModel()

        self.__distance.changed.connect(self.wasChanged)
        self.__centerX.changed.connect(self.wasChanged)
        self.__centerY.changed.connect(self.wasChanged)
        self.__tilt.changed.connect(self.wasChanged)
        self.__tiltPlan.changed.connect(self.wasChanged)

    def __eq__(self, other):
        if not isinstance(other, Fit2dGeometryModel):
            return False
        if self.__distance.value() != other.distance().value():
            return False
        if self.__centerX.value() != other.centerX().value():
            return False
        if self.__centerY.value() != other.centerY().value():
            return False
        if self.__tilt.value() != other.tilt().value():
            return False
        if self.__tiltPlan.value() != other.tiltPlan().value():
            return False
        return True

    def isValid(self, checkWaveLength=True):
        """Check if all the modele have a meaning.

        :param bool checkWaveLength: If true (default) the wavelength is
            checked
        """
        if not self.__distance.isValid():
            return False
        if not self.__centerX.isValid():
            return False
        if not self.__centerY.isValid():
            return False
        if not self.__tilt.isValid():
            return False
        if not self.__tiltPlan.isValid():
            return False
        return True

    def distance(self):
        return self.__distance

    def centerX(self):
        return self.__centerX

    def centerY(self):
        return self.__centerY

    def tilt(self):
        return self.__tilt

    def tiltPlan(self):
        return self.__tiltPlan

    def setFrom(self, geometry):
        self.lockSignals()
        self.distance().setValue(geometry.distance().value())
        self.centerX().setValue(geometry.centerX().value())
        self.centerY().setValue(geometry.centerY().value())
        self.tilt().setValue(geometry.tilt().value())
        self.tiltPlan().setValue(geometry.tiltPlan().value())
        self.unlockSignals()
