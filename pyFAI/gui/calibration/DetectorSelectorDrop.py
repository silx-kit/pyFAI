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
__date__ = "09/08/2018"

import os
import logging
import collections

from silx.gui import qt

import pyFAI.utils
import pyFAI.detectors
from .DetectorModel import AllDetectorModel
from .DetectorModel import DetectorFilter
from .model.DataModel import DataModel

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
        selection.selectionChanged.connect(self.__customDetectorChanged)

        self.__hdf5File = DataModel()
        self.__hdf5File.changed.connect(self.__nexusFileChanged)
        self._hdf5Selection.setModel(self.__hdf5File)
        self._hdf5Loader.clicked.connect(self.__loadDetectorFormFile)
        self._hdf5Result.setVisible(False)
        self._hdf5Error.setVisible(False)

    def __nexusFileChanged(self):
        filename = self.__hdf5File.value()
        self._hdf5Result.setVisible(False)
        self._hdf5Error.setVisible(False)
        self.__hdf5Detector = None

        if not os.path.exists(filename):
            self._hdf5Error.setVisible(True)
            self._hdf5Error.setText("File not found")
            return

        try:
            self.__hdf5Detector = pyFAI.detectors.NexusDetector(filename=filename)
            self._hdf5Result.setVisible(True)
            self._hdf5Result.setText("Detector loaded")
        except Exception as e:
            self._hdf5Error.setVisible(True)
            self._hdf5Error.setText(e.args[0])
            _logger.error(e.args[0])
            _logger.debug("Backtrace", exc_info=True)
            # FIXME Display error dialog
        except KeyboardInterrupt:
            raise

    def __loadDetectorFormFile(self):
        dialog = self.createHdf5Dialog("Load detector from HDF5 file")
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
        self.__hdf5File.setValue(filename)

    def createHdf5Dialog(self, title):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        extensions = collections.OrderedDict()
        extensions["HDF5 files"] = "*.h5"

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
        elif isinstance(self.__detector, pyFAI.detectors.NexusDetector):
            self.__selectNexusDetector(self.__detector)
        else:
            self.__selectRegistreredDetector(detector)

    def detector(self):
        field = self.currentCustomField()
        if field == "FILE":
            return self.__hdf5Detector
        elif field == "MANUAL":
            # TODO: Not implemented
            raise NotImplementedError()
        elif field is None:
            classDetector = self.currentDetectorClass()
            if classDetector is None:
                return None
            return classDetector()
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
        self.__hdf5File.lockSignals()
        self.__hdf5File.setValue(detector.filename)
        self.__hdf5File.unlockSignals()
        self.__hdf5Detector = detector
        # Update the GUI
        self.__setCustomField("FILE")

    def __selectRegistreredDetector(self, detector):
        self.__setManufacturer(detector.MANUFACTURER)
        model = self._modelList.model()
        index = model.indexFromDetector(detector.__class__)
        selection = self._modelList.selectionModel()
        selection.select(index, qt.QItemSelectionModel.ClearAndSelect)
        self._modelList.scrollTo(index)

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

    def __manufacturerChanged(self, selected, deselected):
        # Clean up custom selection
        selection = self._customList.selectionModel()
        selection.reset()
        self._customList.repaint()

        manufacturer = self.currentManufacturer()
        model = self._modelList.model()
        model.setManufacturerFilter(manufacturer)
        self._stacked.setCurrentWidget(self._modelPanel)

    def __customDetectorChanged(self, selected, deselected):
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
