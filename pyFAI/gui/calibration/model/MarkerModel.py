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

from .AbstractModel import AbstractModel


class Marker(object):
    """Abstract marker"""

    def __init__(self, name):
        self.__name = name

    def name(self):
        return self.__name


class PixelMarker(Marker):
    """Mark a pixel at a specific location of an image"""

    def __init__(self, name, x, y):
        super(PixelMarker, self).__init__(name)
        self.__pixel = x, y

    def pixelPosition(self):
        return self.__pixel


class PhysicalMarker(Marker):
    """Mark a point at a specific location of chi/tth"""

    def __init__(self, name, chi, tth):
        super(PhysicalMarker, self).__init__(name)
        self.__physic = chi, tth
        self.__pixel = None

    def physicalPosition(self):
        return self.__physic

    def pixelPosition(self):
        return self.__pixel

    def removePixelPosition(self):
        # TODO: This should invalidate the model
        self.__pixel = None

    def setPixelPosition(self, x, y):
        # TODO: This should invalidate the model
        self.__pixel = x, y


class MarkerModel(AbstractModel):

    def __init__(self, parent=None):
        AbstractModel.__init__(self, parent=parent)
        self.__list = []

    def add(self, marker):
        self.__list.append(marker)
        self.wasChanged()

    def remove(self, marker):
        self.__list.remove(marker)
        self.wasChanged()

    def __len__(self):
        return len(self.__list)

    def __iter__(self):
        return iter(self.__list)
