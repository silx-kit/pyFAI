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
__date__ = "07/08/2018"

import os
import fabio
import numpy
import logging
from contextlib import contextmanager
from collections import OrderedDict
import silx.gui.plot
from silx.gui.colors import Colormap
from silx.gui import qt
import pyFAI.utils
from pyFAI.calibrant import Calibrant
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
from pyFAI.gui.calibration.model.WavelengthToEnergyAdaptor import WavelengthToEnergyAdaptor

_logger = logging.getLogger(__name__)


class ExperimentTask(AbstractCalibrationTask):

    def __init__(self):
        super(ExperimentTask, self).__init__()
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-experiment.ui"), self)
        self.initNextStep()
        self.__dialogState = None

        self._imageLoader.clicked.connect(self.loadImage)
        self._maskLoader.clicked.connect(self.loadMask)
        self._darkLoader.clicked.connect(self.loadDark)
        self._splineLoader.clicked.connect(self.loadSpline)

        self.__plot = self.__createPlot(parent=self._imageHolder)
        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        self._calibrant.setFileLoadable(True)
        self._calibrant.sigLoadFileRequested.connect(self.loadCalibrant)

    def __createPlot(self, parent):
        plot = silx.gui.plot.PlotWidget(parent=parent)
        plot.setKeepDataAspectRatio(True)
        plot.setDataMargins(0.1, 0.1, 0.1, 0.1)
        plot.setGraphXLabel("Y")
        plot.setGraphYLabel("X")

        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)

        colormap = Colormap("inferno", normalization=Colormap.LOGARITHM)
        plot.setDefaultColormap(colormap)
        return plot

    def _updateModel(self, model):
        settings = model.experimentSettingsModel()

        self._calibrant.setModel(settings.calibrantModel())
        self._detector.setModel(settings.detectorModel())
        self._image.setModel(settings.imageFile())
        self._mask.setModel(settings.maskFile())
        self._dark.setModel(settings.darkFile())
        self._spline.setModel(settings.splineFile())
        self._calibrantPreview.setCalibrant(settings.calibrantModel().calibrant())

        adaptor = WavelengthToEnergyAdaptor(self, settings.wavelength())
        self._wavelength.setModel(settings.wavelength())
        self._energy.setModel(adaptor)

        settings.image().changed.connect(self.__imageUpdated)

        settings.calibrantModel().changed.connect(self.__calibrantChanged)
        settings.detectorModel().changed.connect(self.__detectorModelUpdated)
        settings.wavelength().changed.connect(self.__waveLengthChanged)
        self.__detectorModelUpdated()

    def __waveLengthChanged(self):
        settings = self.model().experimentSettingsModel()
        self._calibrantPreview.setWaveLength(settings.wavelength().value())

    def __calibrantChanged(self):
        settings = self.model().experimentSettingsModel()
        self._calibrantPreview.setCalibrant(settings.calibrantModel().calibrant())
        # FIXME debug purpous
        self.printSelectedCalibrant()

    def __detectorModelUpdated(self):
        detector = self.model().experimentSettingsModel().detectorModel().detector()

        self._detectorSizeUnit.setVisible(detector is not None)
        if detector is None:
            self._detectorSize.setText("")
        else:
            text = [str(s) for s in detector.max_shape]
            text = u" × ".join(text)
            self._detectorSize.setText(text)

        self.__updateSplineFileVisibility()
        self.__updateDetector()

    def __displayError(self, label, message=""):
        self._error.setVisible(True)
        self._binning.setVisible(False)
        self._binningLabel.setVisible(False)
        # self._nextStep.setEnabled(False)
        self._error.setText(label)
        self._error.setToolTip(message)

    def __updateDetectorTemplate(self):
        try:
            detector = self.model().experimentSettingsModel().detector()
        except Exception:
            detector = self.model().experimentSettingsModel().detectorModel().detector()

        if detector is None:
            self._detectorSize.setText("")
            self.__plot.removeMarker("xmin")
            self.__plot.removeMarker("xmax")
            self.__plot.removeMarker("ymin")
            self.__plot.removeMarker("ymax")
        else:
            try:
                binning = detector.get_binning()
            except Exception:
                binning = 1, 1
            # clamping
            if binning[0] == 0:
                binning = 1, binning[1]
            if binning[1] == 0:
                binning = binning[0], 1

            shape = detector.max_shape[1] // binning[1], detector.max_shape[0] // binning[0]
            self.__plot.addXMarker(x=0, legend="xmin", color="grey")
            self.__plot.addXMarker(x=shape[0], legend="xmax", color="grey")
            self.__plot.addYMarker(y=0, legend="ymin", color="grey")
            self.__plot.addYMarker(y=shape[1], legend="ymax", color="grey")
            dummy = numpy.array([[[0xF0, 0xF0, 0xF0]]], dtype=numpy.uint8)
            self.__plot.addImage(data=dummy, scale=shape, legend="dummy", z=-10, replace=False)
            self.__plot.resetZoom()

    def __updateDetector(self):
        image = self.model().experimentSettingsModel().image().value()
        if image is None:
            self.__displayError("No image")
            self.__updateDetectorTemplate()
            return
        try:
            detector = self.model().experimentSettingsModel().detector()
            if detector is None:
                self.__displayError("No detector")
                self.__updateDetectorTemplate()
                return
            binning = detector.get_binning()
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            self.__displayError("Sizes not valid", "Inconsistency between image and detector")
            self.__updateDetectorTemplate()
            return

        self._detectorSizeUnit.setVisible(detector is not None)
        self.__updateDetectorTemplate()
        if detector.guess_binning(image):
            text = [str(s) for s in binning]
            text = u" × ".join(text)
            self._binning.setText(text)
            self._binning.setVisible(True)
            self._binningLabel.setVisible(True)
            self._error.setVisible(False)
        else:
            self.__displayError("Sizes not valid", "Inconsistency between image and detector")

    def __updateSplineFileVisibility(self):
        detector = self.model().experimentSettingsModel().detectorModel().detector()
        if detector is not None:
            enabled = detector.__class__.HAVE_TAPER
        else:
            enabled = False
        self._spline.setEnabled(enabled)
        self._splineLoader.setEnabled(enabled)

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self._imageSize.setVisible(image is not None)
        self._imageSizeLabel.setVisible(image is not None)
        self._imageSizeUnit.setVisible(image is not None)
        if image is not None:
            text = [str(s) for s in image.shape]
            text = u" × ".join(text)
            self._imageSize.setText(text)

        image = self.model().experimentSettingsModel().image().value()
        self.__plot.addImage(image, legend="image", z=-1, replace=False)
        self.__plot.resetZoom()
        self.__updateDetector()

    def createImageDialog(self, title, forMask=False):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = OrderedDict()
        extensions["EDF image files"] = "*.edf"
        extensions["TIFF image files"] = "*.tif *.tiff"
        extensions["NumPy binary files"] = "*.npy"
        extensions["CBF files"] = "*.cbf"
        extensions["MarCCD image files"] = "*.mccd"
        if forMask:
            extensions["Fit2D mask files"] = "*.msk"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    @contextmanager
    def getImageFromDialog(self, title, forMask=False):
        dialog = self.createImageDialog(title, forMask)

        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            yield None
            return

        self.__dialogState = dialog.saveState()
        filename = dialog.selectedFiles()[0]
        try:
            with fabio.open(filename) as image:
                yield image
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
            yield None
        except KeyboardInterrupt:
            raise

    def createSplineDialog(self, title):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = OrderedDict()
        extensions["Spline files"] = "*.spline"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def createCalibrantDialog(self, title):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = OrderedDict()
        extensions["Calibrant files"] = "*.D"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def loadImage(self):
        with self.getImageFromDialog("Load calibration image") as image:
            if image is not None:
                settings = self.model().experimentSettingsModel()
                settings.imageFile().setValue(str(image.filename))
                settings.image().setValue(image.data.copy())

    def loadMask(self):
        with self.getImageFromDialog("Load mask image", forMask=True) as image:
            if image is not None:
                settings = self.model().experimentSettingsModel()
                settings.maskFile().setValue(image.filename)
                settings.mask().setValue(image.data)

    def loadDark(self):
        with self.getImageFromDialog("Load dark image") as image:
            if image is not None:
                settings = self.model().experimentSettingsModel()
                settings.darkFile().setValue(image.filename)
                settings.dark().setValue(image.data)

    def loadCalibrant(self):
        dialog = self.createCalibrantDialog("Load calibrant file")

        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            return

        self.__dialogState = dialog.saveState()
        filename = dialog.selectedFiles()[0]
        try:
            calibrant = Calibrant(filename=filename)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
            return
        except KeyboardInterrupt:
            raise

        try:
            settings = self.model().experimentSettingsModel()
            settings.calibrantModel().setCalibrant(calibrant)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def loadSpline(self):
        dialog = self.createSplineDialog("Load spline image")

        if self.__dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(self.__dialogState)

        result = dialog.exec_()
        if not result:
            return

        self.__dialogState = dialog.saveState()
        filename = dialog.selectedFiles()[0]
        try:
            settings = self.model().experimentSettingsModel()
            settings.splineFile().setValue(filename)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def printSelectedCalibrant(self):
        print(self.model().experimentSettingsModel().calibrantModel().calibrant())
