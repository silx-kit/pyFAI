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
from pyFAI.gui.calibration.model.WavelengthToEnergyAdaptor import WavelengthToEnergyAdaptor

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "14/02/2017"

import os
import fabio
import logging
from contextlib import contextmanager
from pyFAI.gui import qt
import pyFAI.utils
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask

_logger = logging.getLogger(__name__)


class ExperimentTask(AbstractCalibrationTask):

    def __init__(self):
        super(ExperimentTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-experiment.ui"), self)
        self.__currentDirectory = os.getcwd()

        self._imageLoader.clicked.connect(self.loadImage)
        self._maskLoader.clicked.connect(self.loadMask)
        self._darkLoader.clicked.connect(self.loadDark)

    def _updateModel(self, model):
        settings = model.experimentSettingsModel()

        self._calibrant.setModel(settings.calibrantModel())
        self._detector.setModel(settings.detectorModel())
        self._image.setModel(settings.imageFile())
        self._mask.setModel(settings.maskFile())
        self._dark.setModel(settings.darkFile())

        adaptor = WavelengthToEnergyAdaptor(self, settings.wavelength())
        self._wavelength.setModel(settings.wavelength())
        self._energy.setModel(adaptor)

        settings.image().changed.connect(self.__imageUpdated)

        # FIXME debug purpous
        settings.calibrantModel().changed.connect(self.printSelectedCalibrant)
        settings.detectorModel().changed.connect(self.printSelectedDetector)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self._imageSize.setVisible(image is not None)
        self._imageSizeLabel.setVisible(image is not None)
        self._imageSizeUnit.setVisible(image is not None)
        if image is not None:
            text = [str(s) for s in image.shape]
            text = u" × ".join(text)
            self._imageSize.setText(text)

    def createImageDialog(self, title, forMask=False):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        filters = [
            "EDF image files (*.edf)",
            "TIFF image files (*.tif)",
            "NumPy binary file (*.npy)",
            "CBF files (*.cbf)",
            "MarCCD image files (*.mccd)"
        ]
        if forMask:
            filters.append("Fit2D mask (*.msk)")
        filters.append("Any file (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        dialog.setDirectory(self.__currentDirectory)
        return dialog

    @contextmanager
    def getImageFromDialog(self, title, forMask=False):
        dialog = self.createImageDialog(title, forMask)
        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        try:
            with fabio.open(filename) as image:
                yield image
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def loadImage(self):
        with self.getImageFromDialog("Load calibration image") as image:
            settings = self.model().experimentSettingsModel()
            settings.imageFile().setValue(image.filename)
            settings.image().setValue(image.data)

    def loadMask(self):
        with self.getImageFromDialog("Load calibration image", forMask=True) as image:
            settings = self.model().experimentSettingsModel()
            settings.maskFile().setValue(image.filename)
            settings.mask().setValue(image.data)

    def loadDark(self):
        with self.getImageFromDialog("Load dark image") as image:
            settings = self.model().experimentSettingsModel()
            settings.darkFile().setValue(image.filename)
            settings.dark().setValue(image.data)

    def printSelectedCalibrant(self):
        print(self.model().experimentSettingsModel().calibrantModel().calibrant())

    def printSelectedDetector(self):
        print(self.model().experimentSettingsModel().detectorModel().detector())
