# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "03/03/2017"

from .AbstractModel import AbstractModel


class ConstraintModel(AbstractModel):

    def __init__(self, parent=None):
        super(ConstraintModel, self).__init__(parent)
        self.__fixed = None
        self.__range = None

    def isValid(self):
        return self.__fixed is True or self.__range is not None

    def setFixed(self, fixed=True):
        if self.__fixed == fixed:
            return
        self.__fixed = fixed
        # self.__range = None
        self.wasChanged()

    def setRangeConstraint(self, minValue, maxValue):
        self.__range = (minValue, maxValue)
        self.wasChanged()

    def isFixed(self):
        return self.__fixed is True

    def isRangeConstrained(self):
        if self.__fixed:
            return False
        return True

    def range(self):
        return self.__range
