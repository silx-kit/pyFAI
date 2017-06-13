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
__date__ = "13/06/2017"

from .AbstractModel import AbstractModel


class PeakSelectionModel(AbstractModel):

    def __init__(self, parent=None):
        super(PeakSelectionModel, self).__init__(parent)
        self.__peaks = []

    def isValid(self):
        for p in self.__peaks:
            if not p.isValid():
                return
        return True

    def __len__(self):
        return len(self.__peaks)

    def __iter__(self):
        for p in self.__peaks:
            yield p

    def __getitem__(self, index):
        return self.__peaks[index]

    def append(self, peak):
        self.__peaks.append(peak)
        peak.changed.connect(self.wasChanged)
        self.wasChanged()

    def remove(self, peak):
        self.__peaks.remove(peak)
        peak.changed.disconnect(self.wasChanged)
        self.wasChanged()

    def clear(self):
        self.lockSignals()
        for peak in list(self.__peaks):
            self.remove(peak)
        self.unlockSignals()

    def index(self, peak):
        return self.__peaks.index(peak)
