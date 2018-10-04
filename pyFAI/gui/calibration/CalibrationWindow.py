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
__date__ = "04/10/2018"

import functools

from silx.gui import qt
from silx.gui import icons

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

        self.__iconCache = {}
        self.__listMode = "text"

        self.__tasks = self.createTasks()
        for task in self.__tasks:
            task.nextTaskRequested.connect(self.nextTask)
            item = qt.QListWidgetItem(self._list)
            item.setText(task.windowTitle())
            item._text = task.windowTitle()
            item.setIcon(task.windowIcon())
            item._icon = task.windowIcon()
            self._stack.addWidget(task)

            fontMetrics = qt.QFontMetrics(item.font())
            width = fontMetrics.width(item.text())
            width += self._list.iconSize().width() + 20
            item.setSizeHint(qt.QSize(width, 60))

            task.warningUpdated.connect(functools.partial(self.__updateTaskState, task, item))

        if len(self.__tasks) > 0:
            self._list.setCurrentRow(0)
            # Hide the nextStep button of the last task
            task.setNextStepVisible(False)

        self._list.sizeHint = self._listSizeHint
        self._list.minimumSizeHint = self._listMinimumSizeHint
        self.setModel(model)

    def __updateTaskState(self, task, item):
        warnings = task.nextStepWarning()
        if warnings is None:
            item.setIcon(item._icon)
        else:
            warningIcon = self.__getWarningIcon(item._icon)
            item.setIcon(warningIcon)

    def __createCompoundIcon(self, backgroundIcon, foregroundIcon):
        icon = qt.QIcon()

        sizes = backgroundIcon.availableSizes()
        sizes = sorted(sizes, key=lambda s: s.height())
        sizes = filter(lambda s: s.height() < 100, sizes)
        sizes = list(sizes)
        if len(sizes) > 0:
            baseSize = sizes[-1]
        else:
            baseSize = qt.QSize(32, 32)

        modes = [qt.QIcon.Normal, qt.QIcon.Disabled]
        for mode in modes:
            pixmap = qt.QPixmap(baseSize)
            pixmap.fill(qt.Qt.transparent)
            painter = qt.QPainter(pixmap)
            painter.drawPixmap(0, 0, backgroundIcon.pixmap(baseSize, mode=mode))
            painter.drawPixmap(0, 0, foregroundIcon.pixmap(baseSize, mode=mode))
            painter.end()
            icon.addPixmap(pixmap, mode=mode)

        return icon

    def __getWarningIcon(self, baseIcon):
        iconHash = baseIcon.cacheKey()
        icon = self.__iconCache.get(iconHash, None)
        if icon is None:
            nxIcon = icons.getQIcon("pyfai:gui/icons/layer-warning")
            icon = self.__createCompoundIcon(baseIcon, nxIcon)
            self.__iconCache[iconHash] = icon
        return icon

    def _listMinimumSizeHint(self):
        return qt.QSize(self._list.iconSize().width() + 7, 10)

    def _listSizeHint(self):
        if self.__listMode == "icon":
            return self._listMinimumSizeHint()
        else:
            maxWidth = 0
            for row in range(self._list.count()):
                item = self._list.item(row)
                width = item.sizeHint().width()
                if maxWidth < width:
                    maxWidth = width
        return qt.QSize(maxWidth, 10)

    def _setListMode(self, mode):
        if self.__listMode == mode:
            return
        self.__listMode = mode
        if mode == "text":
            for row in range(self._list.count()):
                item = self._list.item(row)
                item.setText(item._text)
                item.setToolTip("")
        else:
            for row in range(self._list.count()):
                item = self._list.item(row)
                item.setText(None)
                item.setToolTip(item._text)
        self._list.adjustSize()
        self._list.updateGeometry()

    def resizeEvent(self, event):
        width = event.size().width()
        oldWidth = event.oldSize().width()
        delta = width - oldWidth
        if (delta < 0 or oldWidth == -1) and width < 1100:
            self._setListMode("icon")
        elif (delta > 0 or oldWidth == -1) and width > 1500:
            self._setListMode("text")
        return qt.QMainWindow.resizeEvent(self, event)

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
