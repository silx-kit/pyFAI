# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2018 European Synchrotron Radiation Facility
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


import numpy
from .DataModel import DataModel


class MaskedImageModel(DataModel):
    """Image cleaned up by setting masked pixels to NaN"""

    def __init__(self, parent=None, image=None, mask=None):
        super(MaskedImageModel, self).__init__(parent=parent)
        self.__image = image
        self.__mask = mask
        image.changed.connect(self.__invalidateValue)
        mask.changed.connect(self.__invalidateValue)
        self.__value = None

    def __computeImageData(self):
        image = self.__image.value()
        if image is None:
            return None
        mask = self.__mask.value()
        if mask is None:
            return image
        if mask.shape != image.shape:
            return image
        image = image.astype(copy=True, dtype=numpy.float32)
        image[mask != 0] = float("nan")
        return image

    def __invalidateValue(self):
        self.__value = None
        self.wasChanged()

    def value(self):
        if self.__value is None:
            self.__value = self.__computeImageData()
        return self.__value
