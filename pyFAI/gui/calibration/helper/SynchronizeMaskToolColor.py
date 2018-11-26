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
"""Utils to update the plot background according to the changes of the
application styles"""

__authors__ = ["V. Valls"]
__license__ = "MIT"


import weakref
from ..CalibrationContext import CalibrationContext
from silx.gui import colors


class SynchronizeMaskToolColor(object):

    def __init__(self, maskTool=None):
        self.__register()
        self.__maskTool = weakref.ref(maskTool)
        self.__previousColor = None
        self.__updateApplicationStyle()

    def getMaskTool(self):
        return self.__maskTool()

    def __register(self):
        context = CalibrationContext.instance()
        context.sigStyleChanged.connect(self.__updateApplicationStyle)

    def __updateApplicationStyle(self):
        context = CalibrationContext.instance()
        color = context.getMaskedColor()
        if color != self.__previousColor:
            self.__previousColor = color
            self.__updateMaskColor(color)

    def __updateMaskColor(self, color):
        maskTool = self.getMaskTool()
        # TODO: Not anymore needed for silx >= 0.10
        color = colors.rgba(color)[0:3]
        maskTool.setMaskColors(color)
