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
__date__ = "10/05/2019"

from .AbstractModel import AbstractModel
from .ConstraintModel import ConstraintModel


class GeometryConstraintsModel(AbstractModel):

    def __init__(self, parent=None):
        super(GeometryConstraintsModel, self).__init__(parent)
        self.__distance = ConstraintModel(self)
        self.__wavelength = ConstraintModel(self)
        self.__poni1 = ConstraintModel(self)
        self.__poni2 = ConstraintModel(self)
        self.__rotation1 = ConstraintModel(self)
        self.__rotation2 = ConstraintModel(self)
        self.__rotation3 = ConstraintModel(self)

        self.__distance.changed.connect(self.wasChanged)
        self.__wavelength.changed.connect(self.wasChanged)
        self.__poni1.changed.connect(self.wasChanged)
        self.__poni2.changed.connect(self.wasChanged)
        self.__rotation1.changed.connect(self.wasChanged)
        self.__rotation2.changed.connect(self.wasChanged)
        self.__rotation3.changed.connect(self.wasChanged)

    def isValid(self):
        if not self.__distance.isValid():
            return False
        if not self.__wavelength.isValid():
            return False
        if not self.__poni1.isValid():
            return False
        if not self.__poni2.isValid():
            return False
        if not self.__rotation1.isValid():
            return False
        if not self.__rotation2.isValid():
            return False
        if not self.__rotation3.isValid():
            return False
        return True

    def distance(self):
        return self.__distance

    def wavelength(self):
        return self.__wavelength

    def poni1(self):
        return self.__poni1

    def poni2(self):
        return self.__poni2

    def rotation1(self):
        return self.__rotation1

    def rotation2(self):
        return self.__rotation2

    def rotation3(self):
        return self.__rotation3

    def copy(self, parent=None):
        """
        Copy this model to a new model

        :param qt.QObject parent: Parent of the copyed model
        :rtype: GeometryConstraintsModel
        """
        model = GeometryConstraintsModel(parent=parent)
        model.distance().set(self.__distance)
        model.wavelength().set(self.__wavelength)
        model.poni1().set(self.__poni1)
        model.poni2().set(self.__poni2)
        model.rotation1().set(self.__rotation1)
        model.rotation2().set(self.__rotation2)
        model.rotation3().set(self.__rotation3)
        return model

    def set(self, other):
        """Set this geometry constraints with the other informations.

        :param GeometryConstraintsModel other:
        """
        self.lockSignals()
        self.distance().set(other.distance())
        self.wavelength().set(other.wavelength())
        self.poni1().set(other.poni1())
        self.poni2().set(other.poni2())
        self.rotation1().set(other.rotation1())
        self.rotation2().set(other.rotation2())
        self.rotation3().set(other.rotation3())
        self.unlockSignals()

    def fillDefault(self, other):
        """Fill unset values of this model with the other model

        :param GeometryConstraintsModel other:
        """
        self.lockSignals()
        self.distance().fillDefault(other.distance())
        self.wavelength().fillDefault(other.wavelength())
        self.poni1().fillDefault(other.poni1())
        self.poni2().fillDefault(other.poni2())
        self.rotation1().fillDefault(other.rotation1())
        self.rotation2().fillDefault(other.rotation2())
        self.rotation3().fillDefault(other.rotation3())
        self.unlockSignals()

    def __str__(self):
        template = "GeometryConstraintsModel(d:%s, w:%s, p1:%s, p2:%s, r1:%s, r2:%s, r3:%s)"
        data = self.distance(), self.wavelength(), self.poni1(), self.poni2(), self.rotation1(), self.rotation2(), self.rotation3()
        return template % data
