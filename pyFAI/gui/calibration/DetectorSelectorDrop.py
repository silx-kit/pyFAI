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
__date__ = "13/08/2018"

import os
import logging
import collections

from silx.gui import qt

import pyFAI.utils
import pyFAI.detectors
from .DetectorModel import AllDetectorModel
from .DetectorModel import DetectorFilter
from .model.DataModel import DataModel
from . import validators

_logger = logging.getLogger(__name__)


class DetectorSelectorDrop(qt.QWidget):

    _ManufacturerRole = qt.Qt.UserRole

    _CustomDetectorRole = qt.Qt.UserRole

    def __init__(self, parent):
        super(DetectorSelectorDrop, self).__init__(parent)
        qt.loadUi(pyFAI.utils.get_ui_file("detector-selection-drop.ui"), self)

        self.__detector = None
        self.__dialogState = None

        model = self.__createManufacturerModel()
        self._manufacturerList.setModel(model)
        selection = self._manufacturerList.selectionModel()
        selection.selectionChanged.connect(self.__manufacturerChanged)

        model = AllDetectorModel(self)
        modelFilter = DetectorFilter(self)
        modelFilter.setSourceModel(model)
        self._modelList.setModel(modelFilter)
        selection = self._modelList.selectionModel()
        selection.selectionChanged.connect(self.__modelChanged)

        customModel = qt.QStandardItemModel(self)
        item = qt.QStandardItem("From file")
        item.setData("FILE", role=self._CustomDetectorRole)
        customModel.appendRow(item)
        item = qt.QStandardItem("Manual definition")
        item.setData("MANUAL", role=self._CustomDetectorRole)
        customModel.appendRow(item)
        self._customList.setModel(customModel)
        self._customList.setFixedHeight(self._customList.sizeHintForRow(0) * 2)
        selection = self._customList.selectionModel()
        selection.selectionChanged.connect(self.__customSelectionChanged)

        self.__splineFile = DataModel()
        self._splineFile.setModel(self.__splineFile)
        self._splineLoader.clicked.connect(self.loadSplineFile)
        self.__splineFile.changed.connect(self.__splineFileChanged)
        self._splineError.setVisible(False)

        self.__descriptionFile = DataModel()
        self.__descriptionFile.changed.connect(self.__descriptionFileChanged)
        self._fileSelection.setModel(self.__descriptionFile)
        self._fileLoader.clicked.connect(self.__loadDetectorFormFile)
        self._fileResult.setVisible(False)
        self._fileError.setVisible(False)

        validator = validators.IntegerAndEmptyValidator()
        validator.setBottom(0)
        self._detectorWidth.setValidator(validator)
        self._detectorHeight.setValidator(validator)

        self.__detectorWidth = DataModel()
        self.__detectorHeight = DataModel()
        self.__pixelWidth = DataModel()
        self.__pixelHeight = DataModel()

        self._detectorWidth.setModel(self.__detectorWidth)
        self._detectorHeight.setModel(self.__detectorHeight)
        self._pixelWidth.setModel(self.__pixelWidth)
        self._pixelHeight.setModel(self.__pixelHeight)

        self._customResult.setVisible(False)
        self._customError.setVisible(False)
        self.__detectorWidth.changed.connect(self.__customDetectorChanged)
        self.__detectorHeight.changed.connect(self.__customDetectorChanged)
        self.__pixelWidth.changed.connect(self.__customDetectorChanged)
        self.__pixelHeight.changed.connect(self.__customDetectorChanged)
        self.__customDetector = None

    def __splineFileChanged(self):
        splineFile = self.__splineFile.value()
        if splineFile is not None:
            splineFile = splineFile.strip()
            if splineFile == "":
                splineFile = None

        if splineFile is None:
            # No file, no error
            self._splineError.setVisible(False)
            return

        try:
            import pyFAI.spline
            pyFAI.spline.Spline(splineFile)
            self._splineError.setVisible(False)
        except Exception as e:
            self._splineError.setVisible(True)
            self._splineError.setText(e.args[0])
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog

    def __customDetectorChanged(self):
        detectorWidth = self.__detectorWidth.value()
        detectorHeight = self.__detectorHeight.value()
        pixelWidth = self.__pixelWidth.value()
        pixelHeight = self.__pixelHeight.value()

        self._customResult.setVisible(False)
        self._customError.setVisible(False)
        self.__customDetector = None

        if pixelWidth is None or pixelHeight is None:
            self._customError.setVisible(True)
            self._customError.setText("Pixel size expected")
            return

        if detectorWidth is None or detectorHeight is None:
            self._customError.setVisible(True)
            self._customError.setText("Detector size expected")
            return

        maxShape = detectorWidth, detectorHeight
        detector = pyFAI.detectors.Detector(
            pixel1=pixelWidth / 10.0**6,
            pixel2=pixelHeight / 10.0**6,
            max_shape=maxShape)
        self.__customDetector = detector
        self._customResult.setVisible(True)
        self._customResult.setText("Detector configured")

        detector = pyFAI.detectors.Detector(
            pixel1=pixelWidth / 10.0**6,
            pixel2=pixelHeight / 10.0**6,
            max_shape=maxShape)
        self.__customDetector = detector
        self._customResult.setVisible(True)
        self._customResult.setText("Detector configured")

    def createSplineDialog(self, title):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = collections.OrderedDict()
        extensions["Spline files"] = "*.spline"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def loadSplineFile(self):
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
            self.__splineFile.setValue(filename)
        except Exception as e:
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def __descriptionFileChanged(self):
        filename = self.__descriptionFile.value()
        self._fileResult.setVisible(False)
        self._fileError.setVisible(False)
        self.__detectorFromFile = None

        if not os.path.exists(filename):
            self._fileError.setVisible(True)
            self._fileError.setText("File not found")
            return

        # TODO: this test should be reworked in case of another extension
        if filename.endswith(".spline"):
            try:
                self.__detectorFromFile = pyFAI.detectors.Detector(splineFile=filename)
                self._fileResult.setVisible(True)
                self._fileResult.setText("Spline detector loaded")
            except Exception as e:
                self._fileError.setVisible(True)
                self._fileError.setText(e.args[0])
                _logger.error(e.args[0])
                _logger.debug("Backtrace", exc_info=True)
                # FIXME Display error dialog
            except KeyboardInterrupt:
                raise
            return
        else:
            try:
                self.__detectorFromFile = pyFAI.detectors.NexusDetector(filename=filename)
                self._fileResult.setVisible(True)
                self._fileResult.setText("HDF5 detector loaded")
            except Exception as e:
                self._fileError.setVisible(True)
                self._fileError.setText(e.args[0])
                _logger.error(e.args[0])
                _logger.debug("Backtrace", exc_info=True)
                # FIXME Display error dialog
            except KeyboardInterrupt:
                raise

    def __loadDetectorFormFile(self):
        dialog = self.createFileDialog("Load detector from HDF5 file")
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
        self.__descriptionFile.setValue(filename)

    def createFileDialog(self, title, h5file=True, splineFile=True):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = collections.OrderedDict()
        if h5file:
            extensions["HDF5 files"] = "*.h5"
        if splineFile:
            extensions["Spline files"] = "*.spline"

        filters = []
        filters.append("All supported files (%s)" % " ".join(extensions.values()))
        for name, extension in extensions.items():
            filters.append("%s (%s)" % (name, extension))
        filters.append("All files (*)")

        dialog.setNameFilters(filters)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def setDetector(self, detector):
        if self.__detector == detector:
            return
        self.__detector = detector
        if self.__detector is None:
            self.__selectNoDetector()
        elif self.__detector.__class__ is pyFAI.detectors.NexusDetector:
            self.__selectNexusDetector(self.__detector)
        elif self.__detector.__class__ is pyFAI.detectors.Detector:
            if self.__detector.get_splineFile() is not None:
                self.__selectSplineDetector(self.__detector)
            else:
                self.__selectCustomDetector(self.__detector)
        else:
            self.__selectRegistreredDetector(detector)

    def detector(self):
        field = self.currentCustomField()
        if field == "FILE":
            return self.__detectorFromFile
        elif field == "MANUAL":
            return self.__customDetector
        elif field is None:
            classDetector = self.currentDetectorClass()
            if classDetector is None:
                return None
            detector = classDetector()
            if detector.HAVE_TAPER:
                splineFile = self.__splineFile.value()
                if splineFile is not None:
                    splineFile = splineFile.strip()
                    if splineFile == "":
                        splineFile = None
                detector.set_splineFile(splineFile)
            return detector
        else:
            assert(False)

    def __createManufacturerModel(self):
        manufacturers = set([])
        for detector in pyFAI.detectors.ALL_DETECTORS.values():
            manufacturers.add(detector.MANUFACTURER)

        hasOther = None in manufacturers
        manufacturers.remove(None)
        manufacturers = sorted(list(manufacturers))

        model = qt.QStandardItemModel()

        item = qt.QStandardItem("All")
        item.setData("*", role=self._ManufacturerRole)
        model.appendRow(item)

        # TODO rework this thing with a delegate
        separator = qt.QStandardItem("                  ")
        separator.setSelectable(False)
        separator.setEnabled(False)
        stricked = separator.font()
        stricked.setStrikeOut(True)
        separator.setFont(stricked)
        model.appendRow(separator)

        for manufacturer in manufacturers:
            item = qt.QStandardItem(manufacturer)
            item.setData(manufacturer, role=self._ManufacturerRole)
            model.appendRow(item)

        if hasOther:
            item = qt.QStandardItem("Other")
            item.setData(None, role=self._ManufacturerRole)
            model.appendRow(item)

        return model

    def __selectNoDetector(self):
        self.__setManufacturer("*")

    def __selectNexusDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__descriptionFile.lockSignals()
        self.__descriptionFile.setValue(detector.filename)
        # FIXME: THe unlock send signals, then it's not the way to avoid processing
        self.__descriptionFile.unlockSignals()
        self.__detectorFromFile = detector
        # Update the GUI
        self.__setCustomField("FILE")

    def __selectSplineDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__descriptionFile.lockSignals()
        self.__descriptionFile.setValue(detector.get_splineFile())
        # FIXME: THe unlock send signals, then it's not the way to avoid processing
        self.__descriptionFile.unlockSignals()
        self.__detectorFromFile = detector
        # Update the GUI
        self.__setCustomField("FILE")

    def __selectCustomDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__detectorWidth.changed.disconnect(self.__customDetectorChanged)
        self.__detectorHeight.changed.disconnect(self.__customDetectorChanged)
        self.__pixelWidth.changed.disconnect(self.__customDetectorChanged)
        self.__pixelHeight.changed.disconnect(self.__customDetectorChanged)

        self.__detectorWidth.setValue(detector.max_shape[0])
        self.__detectorHeight.setValue(detector.max_shape[1])
        self.__pixelWidth.setValue(detector.pixel1 * 10.0**6)
        self.__pixelHeight.setValue(detector.pixel2 * 10.0**6)

        self.__customDetector = detector
        self.__detectorWidth.changed.connect(self.__customDetectorChanged)
        self.__detectorHeight.changed.connect(self.__customDetectorChanged)
        self.__pixelWidth.changed.connect(self.__customDetectorChanged)
        self.__pixelHeight.changed.connect(self.__customDetectorChanged)
        # Update the GUI
        self.__setCustomField("MANUAL")

    def __selectRegistreredDetector(self, detector):
        self.__setManufacturer(detector.MANUFACTURER)
        model = self._modelList.model()
        index = model.indexFromDetector(detector.__class__)
        selection = self._modelList.selectionModel()
        selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
        self._modelList.scrollTo(index)

        splineFile = detector.get_splineFile()
        if splineFile is not None:
            self.__splineFile.setValue(splineFile)

    def __setManufacturer(self, manufacturer):
        model = self._manufacturerList.model()
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            storedManufacturer = model.data(index, role=self._ManufacturerRole)
            if manufacturer == storedManufacturer:
                selection = self._manufacturerList.selectionModel()
                selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
                self._manufacturerList.scrollTo(index)
                return

    def __setCustomField(self, field):
        model = self._customList.model()
        fieldIndex = None
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            storedField = index.data(role=self._CustomDetectorRole)
            if field == storedField:
                fieldIndex = index
                break
        if fieldIndex is None:
            assert(False)

        selection = self._customList.selectionModel()
        selection.select(fieldIndex, qt.QItemSelectionModel.ClearAndSelect)

    def currentManufacturer(self):
        indexes = self._manufacturerList.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._manufacturerList.model()
        return model.data(index, role=self._ManufacturerRole)

    def currentCustomField(self):
        indexes = self._customList.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._customList.model()
        return model.data(index, role=self._CustomDetectorRole)

    def currentDetectorClass(self):
        indexes = self._modelList.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._modelList.model()
        return model.data(index, role=AllDetectorModel.CLASS_ROLE)

    def __modelChanged(self, selected, deselected):
        model = self.currentDetectorClass()
        splineAvailable = model is not None and model.HAVE_TAPER
        self._splinePanel.setVisible(splineAvailable)

    def __manufacturerChanged(self, selected, deselected):
        # Clean up custom selection
        selection = self._customList.selectionModel()
        selection.reset()
        self._customList.repaint()

        manufacturer = self.currentManufacturer()
        model = self._modelList.model()
        model.setManufacturerFilter(manufacturer)
        self._stacked.setCurrentWidget(self._modelPanel)

    def __customSelectionChanged(self, selected, deselected):
        # Clean up manufacurer selection
        selection = self._modelList.selectionModel()
        selection.reset()
        selection = self._manufacturerList.selectionModel()
        selection.reset()
        self._modelList.repaint()
        self._manufacturerList.repaint()

        field = self.currentCustomField()
        if field == "FILE":
            self._stacked.setCurrentWidget(self._filePanel)
        elif field == "MANUAL":
            self._stacked.setCurrentWidget(self._manualPanel)
        else:
            assert(False)
