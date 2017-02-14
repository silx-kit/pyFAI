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
__date__ = "14/02/2017"

from pyFAI.gui import qt
import pyFAI.utils
from pyFAI.gui.calibration.model.CalibrationModel import CalibrationModel
from pyFAI.gui.calibration.ExperimentTask import ExperimentTask


class CalibrationWindow(qt.QMainWindow):

    def __init__(self):
        super(CalibrationWindow, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-main.ui"), self)
        self.__model = CalibrationModel()

        self.__tasks = self.createTasks()
        for task in self.__tasks:
            task.setModel(self.__model)
            self._list.addItem(task.windowTitle())
            self._stack.addWidget(task)
        if len(self.__tasks) > 0:
            self._list.setCurrentRow(0)

    def createTasks(self):
        tasks = [
            ExperimentTask()
        ]
        return tasks

    def model(self):
        return self.__model

    def setModel(self, model):
        self.__model = model
        for task in self.__tasks:
            task.setModel(self.__model)
