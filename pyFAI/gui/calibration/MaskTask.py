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
__date__ = "15/02/2017"

import logging
from pyFAI.gui import qt
import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask

import silx.gui.plot

_logger = logging.getLogger(__name__)


class MaskTask(AbstractCalibrationTask):

    def __init__(self):
        super(MaskTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-mask.ui"), self)
        self.__dialogState = None

        self.__plot2D = silx.gui.plot.Plot2D(parent=self._imageHolder)
        self.__plot2D.setKeepDataAspectRatio(True)
        self.__plot2D.getMaskAction().setVisible(False)
        self.__plot2D.getProfileToolbar().setVisible(False)
        self.__maskPanel = silx.gui.plot.MaskToolsWidget.MaskToolsWidget(parent=self._toolHolder, plot=self.__plot2D)

        colormap = {
            'name': "inferno",
            'normalization': 'log',
            'autoscale': True,
        }
        self.__plot2D.setDefaultColormap(colormap)

        self.__maskPanel.setDirection(qt.QBoxLayout.TopToBottom)
        self.__maskPanel.setMultipleMasks("single")

        layout = qt.QVBoxLayout(self._toolHolder)
        layout.addWidget(self.__maskPanel)
        layout.setContentsMargins(0, 0, 0, 0)
        self._toolHolder.setLayout(layout)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot2D)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

    def _updateModel(self, model):
        settings = model.experimentSettingsModel()
        settings.image().changed.connect(self.__imageUpdated)
        settings.mask().changed.connect(self.__maskUpdated)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self.__plot2D.addImage(image, legend="image")

    def __maskUpdated(self):
        mask = self.model().experimentSettingsModel().mask().value()
        self.__maskPanel.setSelectionMask(mask)
