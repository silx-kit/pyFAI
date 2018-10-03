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
__date__ = "03/10/2018"

from silx.gui import qt
import pyFAI.utils
from .model import MarkerModel


class CalibrationWindow(qt.QMainWindow):

    def __init__(self, context):
        super(CalibrationWindow, self).__init__()
        context.setParent(self)
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-main.ui"), self)
        self.__context = context
        model = context.getCalibrationModel()

        context.restoreWindowLocationSettings("main-window", self)

        self.__tasks = self.createTasks()
        for task in self.__tasks:
            task.nextTaskRequested.connect(self.nextTask)
            item = qt.QListWidgetItem(self._list)
            item.setText(task.windowTitle())
            item.setIcon(task.windowIcon())
            self._stack.addWidget(task)

            fontMetrics = qt.QFontMetrics(item.font())
            width = fontMetrics.width(item.text())
            width += self._list.iconSize().width() + 20
            item.setSizeHint(qt.QSize(width, 60))

        if len(self.__tasks) > 0:
            self._list.setCurrentRow(0)

        self._list.sizeHint = self._listSizeHint
        self.setModel(model)

    def _listSizeHint(self):
        maxWidth = 0
        for row in range(self._list.count()):
            item = self._list.item(row)
            width = item.sizeHint().width()
            if maxWidth < width:
                maxWidth = width
        return qt.QSize(maxWidth, 0)

    def closeEvent(self, event):
        for task in self.__tasks:
            task.aboutToClose()
        self.__context.saveWindowLocationSettings("main-window", self)

    def createTasks(self):
        from pyFAI.gui.calibration.ExperimentTask import ExperimentTask
        from pyFAI.gui.calibration.MaskTask import MaskTask
        from pyFAI.gui.calibration.PeakPickingTask import PeakPickingTask
        from pyFAI.gui.calibration.GeometryTask import GeometryTask
        from pyFAI.gui.calibration.IntegrationTask import IntegrationTask

        tasks = [
            ExperimentTask(),
            MaskTask(),
            PeakPickingTask(),
            GeometryTask(),
            IntegrationTask()
        ]
        return tasks

    def model(self):
        return self.__model

    def setModel(self, model):
        self.__model = model

        if len(self.__model.markerModel()) == 0:
            origin = MarkerModel.PixelMarker("Origin", 0, 0)
            self.__model.markerModel().add(origin)

        for task in self.__tasks:
            task.setModel(self.__model)

    def nextTask(self):
        index = self._list.currentRow() + 1
        if index < self._list.count():
            self._list.setCurrentRow(index)
