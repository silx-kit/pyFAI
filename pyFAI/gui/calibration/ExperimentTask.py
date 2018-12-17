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
__date__ = "06/12/2018"

import fabio
import numpy
import logging
from contextlib import contextmanager

import silx.gui.plot
from silx.gui import qt
from silx.gui import icons

import pyFAI.utils
from pyFAI.calibrant import Calibrant
from pyFAI.gui.calibration.AbstractCalibrationTask import AbstractCalibrationTask
import pyFAI.detectors
from .DetectorSelectorDrop import DetectorSelectorDrop
from .helper.SynchronizeRawView import SynchronizeRawView
from .CalibrationContext import CalibrationContext
from ..utils import units
from ..utils import FilterBuilder
from .helper.SynchronizePlotBackground import SynchronizePlotBackground

_logger = logging.getLogger(__name__)


class ExperimentTask(AbstractCalibrationTask):

    def _initGui(self):
        qt.loadUi(pyFAI.utils.get_ui_file("calibration-experiment.ui"), self)
        icon = icons.getQIcon("pyfai:gui/icons/task-settings")
        self.setWindowIcon(icon)

        self.initNextStep()

        self._imageLoader.clicked.connect(self.loadImage)
        self._maskLoader.clicked.connect(self.loadMask)
        self._darkLoader.clicked.connect(self.loadDark)
        self._customDetector.clicked.connect(self.__customDetector)

        self.__plot = self.__createPlot(parent=self._imageHolder)
        self.__plot.setObjectName("plot-experiment")
        self.__plotBackground = SynchronizePlotBackground(self.__plot)

        layout = qt.QVBoxLayout(self._imageHolder)
        layout.addWidget(self.__plot)
        layout.setContentsMargins(1, 1, 1, 1)
        self._imageHolder.setLayout(layout)

        self._detectorFileDescription.setElideMode(qt.Qt.ElideMiddle)

        self._calibrant.setFileLoadable(True)
        self._calibrant.sigLoadFileRequested.connect(self.loadCalibrant)

        self.__synchronizeRawView = SynchronizeRawView()
        self.__synchronizeRawView.registerTask(self)
        self.__synchronizeRawView.registerPlot(self.__plot)

    def __createPlot(self, parent):
        plot = silx.gui.plot.PlotWidget(parent=parent)
        plot.setKeepDataAspectRatio(True)
        plot.setDataMargins(0.1, 0.1, 0.1, 0.1)
        plot.setGraphXLabel("Y")
        plot.setGraphYLabel("X")

        colormap = CalibrationContext.instance().getRawColormap()
        plot.setDefaultColormap(colormap)

        from silx.gui.plot import tools
        toolBar = tools.InteractiveModeToolBar(parent=self, plot=plot)
        plot.addToolBar(toolBar)
        toolBar = tools.ImageToolBar(parent=self, plot=plot)
        colormapDialog = CalibrationContext.instance().getColormapDialog()
        toolBar.getColormapAction().setColorDialog(colormapDialog)
        plot.addToolBar(toolBar)

        return plot

    def _updateModel(self, model):
        self.__synchronizeRawView.registerModel(model.rawPlotView())

        settings = model.experimentSettingsModel()

        self._calibrant.setModel(settings.calibrantModel())
        self._detectorLabel.setDetectorModel(settings.detectorModel())
        self._image.setModel(settings.imageFile())
        self._mask.setModel(settings.maskFile())
        self._dark.setModel(settings.darkFile())

        self._wavelength.setModelUnit(units.Unit.METER_WL)
        self._wavelength.setDisplayedUnit(units.Unit.ANGSTROM)
        self._energy.setModelUnit(units.Unit.METER_WL)
        self._energy.setDisplayedUnit(units.Unit.ENERGY)
        self._wavelength.setModel(settings.wavelength())
        self._energy.setModel(settings.wavelength())

        settings.image().changed.connect(self.__imageUpdated)

        settings.calibrantModel().changed.connect(self.__calibrantChanged)
        settings.detectorModel().changed.connect(self.__detectorModelUpdated)
        settings.wavelength().changed.connect(self.__waveLengthChanged)

        settings.changed.connect(self.__settingsChanged)

        self.__imageUpdated()
        self.__waveLengthChanged()
        self.__calibrantChanged()
        self.__detectorModelUpdated()
        self.__settingsChanged()

    def __settingsChanged(self):
        settings = self.model().experimentSettingsModel()
        settings.detectorModel().changed.connect(self.__detectorModelUpdated)

        image = settings.image().value()
        detectorModel = settings.detectorModel().detector()
        calibrantModel = settings.calibrantModel().calibrant()
        wavelength = settings.wavelength().value()

        warnings = []

        if image is None:
            warnings.append("An image have to be specified")
        if detectorModel is None:
            warnings.append("A detector have to be specified")
        if calibrantModel is None:
            warnings.append("A calibrant have to be specified")
        if wavelength is None:
            warnings.append("An energy have to be specified")
        if image is not None and calibrantModel is not None:
            try:
                detector = settings.detector()
                binning = detector.guess_binning(image)
                if not binning:
                    raise Exception("inconsistancy")
            except Exception:
                warnings.append("Inconsistancy between sizes of image and detector")

        self._globalWarnings = warnings
        self.updateNextStepStatus()

    def nextStepWarning(self):
        if len(self._globalWarnings) == 0:
            return None
        else:
            warning = ""
            for w in self._globalWarnings:
                warning += "<li>%s</li>" % w
            warning = "<ul>%s</ul>" % warning
            warning = "<html>%s</html>" % warning
            return warning

    def __customDetector(self):
        settings = self.model().experimentSettingsModel()
        detector = settings.detectorModel().detector()
        popup = DetectorSelectorDrop(self)
        popupParent = self._customDetector
        pos = popupParent.mapToGlobal(popupParent.rect().bottomRight())
        pos = pos + popup.rect().topLeft() - popup.rect().topRight()
        popup.move(pos)
        popup.show()

        dialog = qt.QDialog(self)
        dialog.setWindowTitle("Detector selection")
        layout = qt.QVBoxLayout(dialog)
        layout.addWidget(popup)

        buttonBox = qt.QDialogButtonBox(qt.QDialogButtonBox.Ok |
                                        qt.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(dialog.accept)
        buttonBox.rejected.connect(dialog.reject)
        layout.addWidget(buttonBox)

        # It have to be here to set the focus on the right widget
        popup.setDetector(detector)
        result = dialog.exec_()
        if result:
            newDetector = popup.detector()
            settings.detectorModel().setDetector(newDetector)

    def __waveLengthChanged(self):
        settings = self.model().experimentSettingsModel()
        self._calibrantPreview.setWaveLength(settings.wavelength().value())

    def __calibrantChanged(self):
        settings = self.model().experimentSettingsModel()
        self._calibrantPreview.setCalibrant(settings.calibrantModel().calibrant())

    def __detectorModelUpdated(self):
        detector = self.model().experimentSettingsModel().detectorModel().detector()

        self._detectorSizeUnit.setVisible(detector is not None)
        if detector is None:
            self._detectorLabel.setStyleSheet("QLabel { color: red }")
            self._detectorSize.setText("")
            self._detectorFileDescription.setVisible(False)
            self._detectorFileDescriptionTitle.setVisible(False)
        else:
            self._detectorLabel.setStyleSheet("QLabel { }")
            text = [str(s) for s in detector.max_shape]
            text = u" × ".join(text)
            self._detectorSize.setText(text)

            if detector.HAVE_TAPER or detector.__class__ == pyFAI.detectors.Detector:
                fileDescription = detector.get_splineFile()
            elif isinstance(detector, pyFAI.detectors.NexusDetector):
                fileDescription = detector.filename
            else:
                fileDescription = None
            if fileDescription is not None:
                fileDescription = fileDescription.strip()
            if fileDescription == "":
                fileDescription = None

            self._detectorFileDescription.setVisible(fileDescription is not None)
            self._detectorFileDescriptionTitle.setVisible(fileDescription is not None)
            self._detectorFileDescription.setText(fileDescription if fileDescription else "")

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
                self._error.setVisible(False)
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

    def __imageUpdated(self):
        image = self.model().experimentSettingsModel().image().value()
        self._imageSize.setVisible(image is not None)
        self._imageSizeLabel.setVisible(image is not None)
        self._imageSizeUnit.setVisible(image is not None)
        if image is not None:
            self.__plot.addImage(image, legend="image", z=-1, replace=False, copy=False)
            text = [str(s) for s in image.shape]
            text = u" × ".join(text)
            self._imageSize.setText(text)
        else:
            self.__plot.removeImage("image")
            self._imageSize.setText("")

        self.__plot.resetZoom()
        self.__updateDetector()

    def createImageDialog(self, title, forMask=False, previousFile=None):
        dialog = CalibrationContext.instance().createFileDialog(self, previousFile=previousFile)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        builder = FilterBuilder.FilterBuilder()
        builder.addImageFormat("EDF image files", "edf")
        builder.addImageFormat("TIFF image files", "tif tiff")
        builder.addImageFormat("NumPy binary files", "npy")
        builder.addImageFormat("CBF files", "cbf")
        builder.addImageFormat("MarCCD image files", "mccd")
        if forMask:
            builder.addImageFormat("Fit2D mask files", "msk")
        dialog.setNameFilters(builder.getFilters())
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    @contextmanager
    def getImageFromDialog(self, title, forMask=False, previousFile=None):
        dialog = self.createImageDialog(title, forMask, previousFile=previousFile)

        result = dialog.exec_()
        if not result:
            yield None
            return

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

    def createCalibrantDialog(self, title):
        dialog = CalibrationContext.instance().createFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("Calibrant files", "D d DS ds")

        dialog.setNameFilters(builder.getFilters())
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def loadImage(self):
        settings = self.model().experimentSettingsModel()
        previousFile = settings.imageFile().value()
        with self.getImageFromDialog("Load calibration image", previousFile=previousFile) as image:
            if image is not None:
                settings.imageFile().setValue(str(image.filename))
                settings.image().setValue(image.data.copy())

    def loadMask(self):
        settings = self.model().experimentSettingsModel()
        previousFile = settings.maskFile().value()
        with self.getImageFromDialog("Load mask image", previousFile=previousFile, forMask=True) as image:
            if image is not None:
                settings.maskFile().setValue(image.filename)
                settings.mask().setValue(image.data)

    def loadDark(self):
        settings = self.model().experimentSettingsModel()
        previousFile = settings.darkFile().value()
        with self.getImageFromDialog("Load dark image", previousFile=previousFile) as image:
            if image is not None:
                settings.darkFile().setValue(image.filename)
                settings.dark().setValue(image.data)

    def loadCalibrant(self):
        dialog = self.createCalibrantDialog("Load calibrant file")

        result = dialog.exec_()
        if not result:
            return

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
