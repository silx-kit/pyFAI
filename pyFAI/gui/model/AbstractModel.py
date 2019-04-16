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
__date__ = "28/02/2019"


import contextlib
from silx.gui import qt


class AbstractModel(qt.QObject):

    changed = qt.Signal()

    def __init__(self, parent=None):
        qt.QObject.__init__(self, parent)
        self.__isLocked = 0
        self.__wasChanged = False

    def isValid(self):
        return True

    def wasChanged(self):
        """Emit the change event in case of the model was not locked.

        :returns: True if the signal was emitted.
        """
        if self.__isLocked > 0:
            self.__wasChanged = True
            return False
        else:
            self.changed.emit()
            return True

    def isLocked(self):
        """Returns True if the events are locked.

        :rtype: bool
        """
        return self.__isLocked > 0

    @contextlib.contextmanager
    def lockContext(self):
        """Context manager to lock and unlock signals."""
        self.lockSignals()
        yield self
        self.unlockSignals()

    def lockSignals(self):
        """Lock the change events
        """
        self.__isLocked = self.__isLocked + 1

    def unlockSignals(self):
        """Unlock the change events

        :returns: False if the model is still locked, else True
        """
        assert self.__isLocked > 0
        self.__isLocked = self.__isLocked - 1
        if self.__isLocked == 0 and self.__wasChanged:
            self.__wasChanged = False
            self.wasChanged()
        return self.__isLocked == 0
