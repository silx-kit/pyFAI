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
__date__ = "27/11/2018"

from .AbstractModel import AbstractModel


class PeakModel(AbstractModel):

    def __init__(self, parent=None):
        super(PeakModel, self).__init__(parent)
        self.__name = None
        self.__color = None
        self.__coords = []
        self.__ringNumber = None

    def __len__(self):
        return len(self.__coords)

    def isValid(self):
        return self.__name is not None and self.__ringNumber is not None

    def name(self):
        return self.__name

    def setName(self, name):
        self.__name = name
        self.wasChanged()

    def color(self):
        return self.__color

    def setColor(self, color):
        self.__color = color
        self.wasChanged()

    def coords(self):
        return self.__coords

    def setCoords(self, coords):
        self.__coords = coords
        self.wasChanged()

    def ringNumber(self):
        return self.__ringNumber

    def setRingNumber(self, ringNumber):
        assert(ringNumber >= 1)
        self.__ringNumber = ringNumber
        self.wasChanged()

    def copy(self, parent=None):
        peakModel = PeakModel(parent)
        peakModel.setName(self.name())
        peakModel.setColor(self.color())
        peakModel.setCoords(self.coords())
        peakModel.setRingNumber(self.ringNumber())
        return peakModel
