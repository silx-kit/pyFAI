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
__date__ = "14/05/2019"

import functools

from silx.gui import qt
from silx.gui import icons

import pyFAI.utils
from .model import MarkerModel
from .utils import projecturl


class MenuItem(qt.QListWidgetItem):

    TextMode = 0
    IconMode = 1

    def __init__(self, parent):
        super(MenuItem, self).__init__(parent)
        self.__text = None
        self.__icon = qt.QIcon()
        self.__warningIcon = None
        self.__warnings = None
        self.__mode = self.TextMode

    def setText(self, text):
        self.__text = text
        self.__updateItem()

    def setIcon(self, icon):
        self.__icon = icon
        self.__updateItem()

    def setWarnings(self, warnings):
        self.__warnings = warnings
        self.__updateItem()

    def setDisplayMode(self, mode):
        self.__mode = mode
        self.__updateItem()

    def __updateItem(self):
        superSelf = super(MenuItem, self)
        if self.__mode == self.TextMode:
            superSelf.setText(self.__text)
            fontMetrics = qt.QFontMetrics(self.font())
            width = fontMetrics.width(self.text())
            width += self.listWidget().iconSize().width() + 20
            superSelf.setSizeHint(qt.QSize(width, 60))
        elif self.__mode == self.IconMode:
            superSelf.setText(None)
            width = self.listWidget().iconSize().width() + 7
            superSelf.setSizeHint(qt.QSize(width, 60))
        else:
            assert(False)

        if self.__warnings is None:
            superSelf.setIcon(self.__icon)
        else:
            if self.__warningIcon is None:
                icon = self.__createWarningIcon(self.__icon)
                self.__warningIcon = icon
            superSelf.setIcon(self.__warningIcon)

        if self.__warnings is None and self.__mode == self.TextMode:
            superSelf.setToolTip("")
        else:
            toolTip = self.__text

            toolTip = ""
            if self.__mode == self.IconMode:
                toolTip += self.__text
            if self.__warnings is not None:
                if toolTip != "":
                    toolTip += "<br/>"
                toolTip += self.__warnings
            superSelf.setToolTip(toolTip)

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

    def __createWarningIcon(self, baseIcon):
        nxIcon = icons.getQIcon("pyfai:gui/icons/layer-warning")
        icon = self.__createCompoundIcon(baseIcon, nxIcon)
        return icon


class CalibrationWindow(qt.QMainWindow):

    def __init__(self, context):
        super(CalibrationWindow, self).__init__()
        context.setParent(self)
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-main.ui"), self)
        self.__context = context
        model = context.getCalibrationModel()

        pyfaiIcon = icons.getQIcon("pyfai:gui/images/icon")
        self.setWindowIcon(pyfaiIcon)

        context.restoreWindowLocationSettings("main-window", self)

        self.__listMode = MenuItem.TextMode

        self.__tasks = self.createTasks()
        for task in self.__tasks:
            task.nextTaskRequested.connect(self.nextTask)
            item = MenuItem(self._list)
            item.setText(task.windowTitle())
            item.setIcon(task.windowIcon())
            self._stack.addWidget(task)

            task.warningUpdated.connect(functools.partial(self.__updateTaskState, task, item))

        if len(self.__tasks) > 0:
            self._list.setCurrentRow(0)
            # Hide the nextStep button of the last task
            task.setNextStepVisible(False)

        self._list.sizeHint = self._listSizeHint
        self._list.minimumSizeHint = self._listMinimumSizeHint
        self.setModel(model)

        url = projecturl.get_documentation_url("")
        if url.startswith("http"):
            self._help.setText("Online help...")
        self._helpText = self._help.text()
        self._help.clicked.connect(self.__displayHelp)

    def __displayHelp(self):
        subpath = "usage/cookbook/calib-gui/index.html"
        url = projecturl.get_documentation_url(subpath)
        qt.QDesktopServices.openUrl(qt.QUrl(url))

    def __updateTaskState(self, task, item):
        warnings = task.nextStepWarning()
        item.setWarnings(warnings)

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
        for row in range(self._list.count()):
            item = self._list.item(row)
            item.setDisplayMode(mode)
        self._list.adjustSize()
        self._list.updateGeometry()

    def __minimizeMenu(self):
        self._setListMode(MenuItem.IconMode)
        icon = icons.getQIcon("pyfai:gui/icons/menu-help")
        self._help.setIcon(icon)
        self._help.setText("")

    def __maximizeMenu(self):
        self._setListMode(MenuItem.TextMode)
        self._help.setIcon(qt.QIcon())
        self._help.setText(self._helpText)

    def resizeEvent(self, event):
        width = event.size().width()
        oldWidth = event.oldSize().width()
        delta = width - oldWidth
        if (delta < 0 or oldWidth == -1) and width < 1100:
            self.__minimizeMenu()
        elif (delta > 0 or oldWidth == -1) and width > 1500:
            self.__maximizeMenu()
        return qt.QMainWindow.resizeEvent(self, event)

    def closeEvent(self, event):
        poniFile = self.model().experimentSettingsModel().poniFile()

        if not poniFile.isSynchronized():
            button = qt.QMessageBox.question(self,
                                             "calib2",
                                             "The PONI file was not saved.\nDo you really want to close the application?",
                                             qt.QMessageBox.Cancel | qt.QMessageBox.No | qt.QMessageBox.Yes,
                                             qt.QMessageBox.Yes)
            if button != qt.QMessageBox.Yes:
                event.ignore()
                return

        event.accept()

        for task in self.__tasks:
            task.aboutToClose()
        self.__context.saveWindowLocationSettings("main-window", self)

    def createTasks(self):
        from pyFAI.gui.tasks.ExperimentTask import ExperimentTask
        from pyFAI.gui.tasks.MaskTask import MaskTask
        from pyFAI.gui.tasks.PeakPickingTask import PeakPickingTask
        from pyFAI.gui.tasks.GeometryTask import GeometryTask
        from pyFAI.gui.tasks.IntegrationTask import IntegrationTask

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
