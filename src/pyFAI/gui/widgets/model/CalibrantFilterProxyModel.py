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
__date__ = "25/06/2023"

from silx.gui import qt
from silx.gui import icons
import pyFAI.calibrant
from pyFAI.calibrant import Calibrant
from .CalibrantItemModel import CalibrantItemModel


class CalibrantFilterProxyModel(qt.QSortFilterProxyModel):

    def __init__(self, parent):
        super(CalibrantFilterProxyModel, self).__init__(parent)
        self.__displayUser: bool = True
        self.__displayResource: bool = True
        self.__filenames = None

    def setFilter(self, displayResource: bool, displayUser: bool, filenames=None):
        if (self.__displayResource == displayResource and self.__displayUser == displayUser):
            return
        self.__displayResource = displayResource
        self.__displayUser = displayUser
        self.__filenames = filenames
        self.invalidateFilter()

    def filterAcceptsRow(self, sourceRow, sourceParent):
        sourceModel = self.sourceModel()
        index = sourceModel.index(sourceRow, 0, sourceParent)
        calibrant = index.data(CalibrantItemModel.CALIBRANT_ROLE)
        if self.__filenames is not None:
            return calibrant.filename in self.__filenames

        is_user = not calibrant.filename.startswith("pyfai:")
        return (self.__displayUser and is_user) or (self.__displayResource and not is_user)

    def indexFromCalibrant(self, calibrant: Calibrant):
        sourceModel = self.sourceModel()
        index = sourceModel.indexFromCalibrant(calibrant)
        index = self.mapFromSource(index)
        return index
