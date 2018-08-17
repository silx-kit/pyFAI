# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Context shared through all the application"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "17/08/2018"

import weakref
import logging

from silx.gui.dialog.ColormapDialog import ColormapDialog


_logger = logging.getLogger(__name__)


class CalibrationContext(object):

    _instance = None

    @classmethod
    def instance(cls):
        """
        :rtype: CalibrationContext
        """
        assert(cls._instance is not None)
        return cls._instance

    def __init__(self):
        assert(self.__class__._instance is None)
        self.__parent = None
        self.__defaultColormapDialog = None
        self.__class__._instance = self

    def setParent(self, parent):
        self.__parent = weakref.ref(parent)

    def getColormapDialog(self):
        """Returns a shared color dialog.

        :rtype: ColorDialog
        """
        if self.__defaultColormapDialog is None:
            parent = self.__parent()
            if parent is None:
                return None
            dialog = ColormapDialog(parent=parent)
            dialog.setModal(False)
            self.__defaultColormapDialog = dialog
        return self.__defaultColormapDialog
