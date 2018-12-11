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
__date__ = "28/11/2018"

from silx.gui import qt
from silx.gui import icons


class AbstractCalibrationTask(qt.QWidget):

    widgetShow = qt.Signal()
    widgetHide = qt.Signal()
    nextTaskRequested = qt.Signal()

    _cacheWarningIcon = None

    warningUpdated = qt.Signal()

    def __init__(self):
        super(AbstractCalibrationTask, self).__init__()
        self._initGui()
        self.__model = None
        self.installEventFilter(self)
        if hasattr(self, "_nextStep"):
            self._nextStep.setIconSize(qt.QSize(32, 32))

    def _initGui(self):
        """Inherite this method to custom the widget"""
        pass

    def initNextStep(self):
        if hasattr(self, "_nextStep"):
            self._nextStep.clicked.connect(self.nextTask)

    def _warningIcon(self):
        if self._cacheWarningIcon is None:
            icon = icons.getQIcon("pyfai:gui/icons/warning")
            pixmap = icon.pixmap(64)
            coloredIcon = qt.QIcon()
            coloredIcon.addPixmap(pixmap, qt.QIcon.Normal)
            coloredIcon.addPixmap(pixmap, qt.QIcon.Disabled)
            self._cacheWarningIcon = coloredIcon
        return self._cacheWarningIcon

    def updateNextStepStatus(self):
        if not hasattr(self, "_nextStep"):
            return
        warning = self.nextStepWarning()
        if warning is None:
            icon = qt.QIcon()
            self._nextStep.setIcon(icon)
            self._nextStep.setToolTip(None)
            self._nextStep.setEnabled(True)
        else:
            self._nextStep.setIcon(self._warningIcon())
            self._nextStep.setToolTip(warning)
            self._nextStep.setEnabled(False)
        self.warningUpdated.emit()

    def nextStepWarning(self):
        return None

    def setNextStepVisible(self, isVisible):
        if hasattr(self, "_nextStep"):
            self._nextStep.setVisible(isVisible)

    def eventFilter(self, widget, event):
        result = super(AbstractCalibrationTask, self).eventFilter(widget, event)
        if event.type() == qt.QEvent.Show:
            self.widgetShow.emit()
        elif event.type() == qt.QEvent.Hide:
            self.widgetHide.emit()
        return result

    def model(self):
        """
        Returns the calibration model

        :rtype: CalibrationModel
        """
        return self.__model

    def aboutToClose(self):
        pass

    def setModel(self, model):
        self.__model = model
        self._updateModel(model)

    def nextTask(self):
        self.nextTaskRequested.emit()
