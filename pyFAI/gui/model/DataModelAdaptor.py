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
__date__ = "14/02/2017"

from .AbstractModel import AbstractModel


class DataModelAdaptor(AbstractModel):

    def __init__(self, parent=None, model=None):
        if model is None:
            raise ValueError("Model expected")
        super(DataModelAdaptor, self).__init__(parent)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)

    def __modelChanged(self):
        self.wasChanged()

    def isValid(self):
        return self.__model.isValid()

    def fromModel(self, data):
        raise NotImplementedError("It have to be implemented by inheritance")

    def toModel(self, data):
        raise NotImplementedError("It have to be implemented by inheritance")

    def value(self):
        value = self.__model.value()
        value = self.fromModel(value)
        return value

    def setValue(self, value):
        value = self.toModel(value)
        self.__model.setValue(value)
