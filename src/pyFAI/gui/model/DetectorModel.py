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

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/12/2023"

from .AbstractModel import AbstractModel
from silx.gui import qt


class DetectorModel(AbstractModel):

    def __init__(self, parent=None):
        super(DetectorModel, self).__init__(parent)
        self.__detector = None

    def isValid(self):
        return self.__detector is not None

    def setDetector(self, detector):
        self.__detector = detector
        self.wasChanged()

    def detector(self):
        return self.__detector


class DetectorOrientationModel(qt.QAbstractItemModel):
    def __init__(self, parent=None, orientations=None):
        super(DetectorOrientationModel, self).__init__(parent)
        if orientations:
            self._orientation_list = list(orientations)
        else:
            self._orientation_list = list()

    def columnCount(self):
        return 1

    def rowCount(self, *arg, **kwarg):
        print(f"DetectorOrientationModel: arg={arg}, kwarg={kwarg}")
        return len(self._orientation_list)
