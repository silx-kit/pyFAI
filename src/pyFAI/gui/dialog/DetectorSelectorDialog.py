# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2025 European Synchrotron Radiation Facility
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

__authors__ = ["Valentin Valls", "Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "21/11/2025"

import os
import logging
import textwrap
from silx.gui import qt
from ...utils import get_ui_file
from ... import detectors
from ..widgets.model.AllDetectorItemModel import AllDetectorItemModel
from ..widgets.model.DetectorFilterProxyModel import DetectorFilterProxyModel
from ..model.DataModel import DataModel
from ..utils import validators, block_signals
from ..ApplicationContext import ApplicationContext
from ..utils import FilterBuilder
from ...detectors.sensors import SensorConfig

_logger = logging.getLogger(__name__)


class DetectorSelectorDrop(qt.QWidget):
    _ManufacturerRole = qt.Qt.UserRole

    _CustomDetectorRole = qt.Qt.UserRole

    def __init__(self, parent=None):
        super(DetectorSelectorDrop, self).__init__(parent)
        qt.loadUi(get_ui_file("detector-selection-drop.ui"), self)

        self.__dialogState = None
        self.__detector = None
        self.__customDetector = None
        self.__detectorFromFile = None

        model = self.__createManufacturerModel()
        self._manufacturerList.setModel(model)
        selection = self._manufacturerList.selectionModel()
        selection.selectionChanged.connect(self.__manufacturerChanged)

        model = AllDetectorItemModel(self)
        modelFilter = DetectorFilterProxyModel(self)
        modelFilter.setSourceModel(model)

        self._detectorView.setModel(modelFilter)
        self._detectorView.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self._detectorView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self._detectorView.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self._detectorView.setWordWrap(False)

        header = self._detectorView.horizontalHeader()
        # Manufacturer first
        self.MANUFACTURER_COLUMN = 1
        header.moveSection(self.MANUFACTURER_COLUMN, 0)
        if qt.qVersion() < "5.0":
            header.setSectionResizeMode = self.setResizeMode
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)

        selection = self._detectorView.selectionModel()
        selection.selectionChanged.connect(self.__modelChanged)
        self._detectorView.doubleClicked.connect(self.__selectAndAccept)

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
        self._splinePanel.setVisible(False)

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

        self._initOrientation()
        self._initSensor()
        # By default select all the manufacturers
        self.__selectAllRegistreredDetector()

    def _initOrientation(self):
        for item in detectors.orientation.Orientation:
            if item.available:
                self._detectorOrientation.addItem(
                    f"{item.value}: {item.name} ", userData=item
                )

        self._detectorOrientation.currentIndexChanged.connect(self.__orientationChanged)
        default = detectors.orientation.Orientation(3)
        self._detectorOrientation.setCurrentIndex(
            self._detectorOrientation.findData(default)
        )

    def _initSensor(self):
        "Wire signals/slots"
        self._detectorSensorThickness.setValidator(qt.QDoubleValidator(0.0, 1e4, 1))
        self._resetSensor()
        self._detectorSensorMaterials.currentIndexChanged.connect(self.__sensorChanged)
        self._detectorSensorThickness.currentTextChanged.connect(self.__sensorChanged)

    def _resetSensor(self, detector=None):
        """populate the 2 comboBox with information from the detector.
        By default, expose all possible settings"""
        with block_signals(self._detectorSensorMaterials),  block_signals(self._detectorSensorThickness):
            # Flush
            self._detectorSensorMaterials.clear()
            self._detectorSensorThickness.clear()
            info = []  # used to populate SensorInfo

            # Repopulate
            if detector and detector.SENSORS:
                materials = set()  # Avoid duplicated population
                thicknesses = set()
                for config in detector.SENSORS:
                    info.append(str(config))
                    mat = config.material
                    thick = config.thickness
                    if mat not in materials:
                        self._detectorSensorMaterials.addItem(
                            mat.name, userData=mat
                        )
                        materials.add(mat)
                    if thick not in thicknesses:
                        stg = f"{1e6 * thick:4.0f}" if thick else ""
                        self._detectorSensorThickness.addItem(stg, userData=thick)
                        thicknesses.add(thick)
            else:
                self._detectorSensorMaterials.addItem(" ", userData=None)
                for key, value in detectors.sensors.ALL_MATERIALS.items():
                    self._detectorSensorMaterials.addItem(key, userData=value)
                self._detectorSensorThickness.addItem("", userData=None)
            self._detectorSensorInfo.setText("|".join(info))

        if isinstance(detector, detectors.Detector):
            self.setSensorConfig(detector.sensor)

    def getSensorConfig(self) -> SensorConfig | None:
        sensor_material = self._detectorSensorMaterials.currentData()
        if sensor_material:
            sensor = SensorConfig(sensor_material)
            thickness = self._detectorSensorThickness.currentText()  # Not data since can be user modified
            if thickness.strip():
                sensor.thickness = 1e-6 * float(thickness)
            else:
                sensor.thickness = None
        else:
            sensor = None
        return sensor

    def setSensorConfig(self, sensor: SensorConfig | None = None):
        if sensor is None:
            sensor = SensorConfig(None, None)
        index = self._detectorSensorThickness.findData(sensor.thickness)
        with block_signals(self._detectorSensorThickness):
            if index < 0:
                self._detectorSensorThickness.addItem(
                    f"{1e6 * sensor.thickness:4.0f}" if sensor.thickness else "",
                    userData=sensor.thickness
                )
                index = self._detectorSensorThickness.findData(sensor.thickness)
            self._detectorSensorThickness.setCurrentIndex(index)

        index = self._detectorSensorMaterials.findData(sensor.material)
        with block_signals(self._detectorSensorMaterials):
            if index < 0:
                self._detectorSensorMaterials.addItem(
                    sensor.material.name if sensor.material else "",
                    userData=sensor.material
                )
                index = self._detectorSensorMaterials.findData(sensor.material)
            self._detectorSensorMaterials.setCurrentIndex(index)

    def __sensorChanged(self, **kwargs):
        # Finally set sensor config of all possible the detectors
        sensor = self.getSensorConfig()
        if self.__customDetector:
            self.__customDetector.sensor = sensor
        if self.__detectorFromFile:
            self.__detectorFromFile.sensor = sensor
        if self.__detector:
            self.__detector.sensor = sensor

    def getOrientation(self, idx=None):
        if idx is None:
            return self._detectorOrientation.currentData()
        else:
            return self._detectorOrientation.itemData(idx)

    def __orientationChanged(self, idx):
        orientation = self._detectorOrientation.itemData(idx)
        self._detectorOrientationLabel.setText(textwrap.fill(orientation.__doc__, 40))

        # Finally set the detector orientation
        if self.__customDetector:
            self.__customDetector._orientation = orientation
        if self.__detectorFromFile:
            self.__detectorFromFile._orientation = orientation
        if self.__detector:
            self.__detector._orientation = orientation

    def __selectAndAccept(self):
        # FIXME: This has to be part of the dialog, and not here
        window = self.window()
        if isinstance(window, qt.QDialog):
            window.accept()

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

    def __customDetectorChanged(self, *args, **kwargs):
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
        detector = detectors.Detector(
            pixel1=pixelWidth * 1e-6,
            pixel2=pixelHeight * 1e-6,
            max_shape=maxShape,
            orientation=self.getOrientation(),
            sensor=self.getSensorConfig(),
        )
        self.__customDetector = detector
        self._customResult.setVisible(True)
        self._customResult.setText("Detector configured")

    def createSplineDialog(self, title, previousFile):
        dialog = ApplicationContext.instance().createFileDialog(
            self, previousFile=previousFile
        )
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("Spline files", "spline")

        dialog.setNameFilters(builder.getFilters())
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def loadSplineFile(self):
        previousFile = self.__splineFile.value()
        dialog = self.createSplineDialog("Load spline image", previousFile=previousFile)
        result = (dialog).exec()
        if not result:
            return
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

        if not filename:
            self._fileError.setVisible(False)
            self._fileError.setText("")
            return

        if not os.path.exists(filename):
            self._fileError.setVisible(True)
            self._fileError.setText("File not found")
            return

        # TODO: this test should be reworked in case of another extension
        if filename.endswith(".spline"):
            try:
                self.__detectorFromFile = detectors.Detector(
                    splinefile=filename,
                    orientation=self.getOrientation(),
                    sensor=self.getSensorConfig(),
                )
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
                self.__detectorFromFile = detectors.NexusDetector(
                    filename=filename,
                    orientation=self.getOrientation(),
                    sensor=self.getSensorConfig(),
                )
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
        previousFile = self.__descriptionFile.value()
        dialog = self.createFileDialog(
            "Load detector from HDF5 file", previousFile=previousFile
        )
        result = (dialog).exec()
        if not result:
            return
        filename = dialog.selectedFiles()[0]
        self.__descriptionFile.setValue(filename)

    def createFileDialog(self, title, h5file=True, splineFile=True, previousFile=None):
        dialog = ApplicationContext.instance().createFileDialog(
            self, previousFile=previousFile
        )
        dialog.setWindowTitle(title)
        dialog.setModal(True)

        builder = FilterBuilder.FilterBuilder()
        if h5file:
            builder.addFileFormat("HDF5 files", "h5")
        if splineFile:
            builder.addFileFormat("Spline files", "spline")

        dialog.setNameFilters(builder.getFilters())
        dialog.setFileMode(qt.QFileDialog.ExistingFile)
        return dialog

    def setDetector(self, detector):
        # _logger.info("in setDetector: %s", detector)
        if self.__detector == detector:
            return
        self.__detector = detector
        # set orientation:
        orientation = detector.orientation
        self._detectorOrientation.setCurrentIndex(
            self._detectorOrientation.findData(orientation)
        )
        if self.__detector is None:
            self.__selectNoDetector()
        elif self.__detector.__class__ is detectors.NexusDetector:
            self.__selectNexusDetector(self.__detector)
        elif self.__detector.__class__ is detectors.Detector:
            if self.__detector.splinefile is not None:
                self.__selectSplineDetector(self.__detector)
            else:
                self.__selectCustomDetector(self.__detector)
        else:
            self.__selectRegistreredDetector(detector)
        # # Finally, set the senor information
        self.setSensorConfig(detector.sensor)

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
            try:
                detector = classDetector(
                    orientation=self.getOrientation(), sensor=self.getSensorConfig()
                )
            except TypeError as err:
                _logger.error(err)
                detector = classDetector(orientation=self.getOrientation())
            if detector.HAVE_TAPER:
                splineFile = self.__splineFile.value()
                if splineFile is not None:
                    splineFile = splineFile.strip()
                    if splineFile == "":
                        splineFile = None
                detector.splinefile = splineFile
            return detector
        else:
            raise RuntimeError("field should be FILE, MANUAL or eventually None")

    def __createManufacturerModel(self):
        manufacturers = set([])
        for detector in detectors.ALL_DETECTORS.values():
            manufacturer = detector.MANUFACTURER
            if isinstance(manufacturer, list):
                manufacturer = set(manufacturer)
                if None in manufacturer:
                    manufacturer.remove(None)
                manufacturers |= manufacturer
            else:
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
        self.__initSensors()

    def __selectNexusDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__descriptionFile.lockSignals()
        self.__descriptionFile.setValue(detector.filename)
        # FIXME: THe unlock send signals, then it's not the way to avoid processing
        self.__descriptionFile.unlockSignals()
        self.__detectorFromFile = detector
        # Update the GUI
        self.__setCustomField("FILE")
        self._customList.setFocus(qt.Qt.NoFocusReason)

    def __selectSplineDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__descriptionFile.lockSignals()
        self.__descriptionFile.setValue(detector.splinefile)
        # FIXME: THe unlock send signals, then it's not the way to avoid processing
        self.__descriptionFile.unlockSignals()
        self.__detectorFromFile = detector
        # Update the GUI
        self.__setCustomField("FILE")
        self._customList.setFocus(qt.Qt.NoFocusReason)

    def __selectCustomDetector(self, detector):
        """Select and display the detector using zero copy."""
        self.__detectorWidth.changed.disconnect(self.__customDetectorChanged)
        self.__detectorHeight.changed.disconnect(self.__customDetectorChanged)
        self.__pixelWidth.changed.disconnect(self.__customDetectorChanged)
        self.__pixelHeight.changed.disconnect(self.__customDetectorChanged)

        if detector.max_shape is None:
            self.__detectorWidth.setValue(None)
            self.__detectorHeight.setValue(None)
        else:
            self.__detectorWidth.setValue(detector.max_shape[0])
            self.__detectorHeight.setValue(detector.max_shape[1])
        if detector.pixel1 is not None:
            self.__pixelWidth.setValue(detector.pixel1 * 1e6)
        else:
            self.__pixelWidth.setValue(None)
        if detector.pixel2 is not None:
            self.__pixelHeight.setValue(detector.pixel2 * 1e6)
        else:
            self.__pixelHeight.setValue(None)

        self.__customDetector = detector
        self.__detectorWidth.changed.connect(self.__customDetectorChanged)
        self.__detectorHeight.changed.connect(self.__customDetectorChanged)
        self.__pixelWidth.changed.connect(self.__customDetectorChanged)
        self.__pixelHeight.changed.connect(self.__customDetectorChanged)
        # Update the GUI
        self.__setCustomField("MANUAL")
        self._customList.setFocus(qt.Qt.NoFocusReason)

    def __selectAllRegistreredDetector(self):
        headerView = self._detectorView.horizontalHeader()
        headerView.setSectionHidden(self.MANUFACTURER_COLUMN, True)

        model = self._manufacturerList.model()
        selectionModel = self._manufacturerList.selectionModel()
        index = 0
        indexStart = model.index(index, 0)
        indexEnd = model.index(index, model.columnCount() - 1)
        selection = qt.QItemSelection(indexStart, indexEnd)
        selectionModel.select(selection, qt.QItemSelectionModel.ClearAndSelect)

    def __selectRegistreredDetector(self, detector):
        headerView = self._detectorView.horizontalHeader()
        headerView.setSectionHidden(self.MANUFACTURER_COLUMN, False)

        manufacturer = detector.MANUFACTURER
        if isinstance(manufacturer, list):
            manufacturer = manufacturer[0]
        self.__setManufacturer(manufacturer)
        model = self._detectorView.model()
        index = model.indexFromDetector(detector.__class__, manufacturer)
        index = index.row()
        selectionModel = self._detectorView.selectionModel()
        indexStart = model.index(index, 0)
        indexEnd = model.index(index, model.columnCount() - 1)
        selection = qt.QItemSelection(indexStart, indexEnd)
        selectionModel.select(selection, qt.QItemSelectionModel.ClearAndSelect)
        self._detectorView.scrollTo(indexStart, qt.QAbstractItemView.PositionAtCenter)

        splineFile = detector.splinefile
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
                self._manufacturerList.scrollTo(
                    index, qt.QAbstractItemView.PositionAtCenter
                )
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
            raise RuntimeError("fieldIndex is not defined in any row of the list")

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
        indexes = self._detectorView.selectedIndexes()
        if len(indexes) == 0:
            return None
        index = indexes[0]
        model = self._detectorView.model()
        return model.data(index, role=AllDetectorItemModel.CLASS_ROLE)

    def __modelChanged(self, selected, deselected):
        model = self.currentDetectorClass()
        splineAvailable = model is not None and model.HAVE_TAPER
        self._splinePanel.setVisible(splineAvailable)
        # _logger.info("in __modelChanged: %s %s",type(model), model)
        if isinstance(self.__detector, model):
            # more precise, contains sensor information
            self._resetSensor(detector=self.__detector)
        else:
            self._resetSensor(detector=model)

    def __manufacturerChanged(self, selected, deselected):
        # Clean up custom selection
        selection = self._customList.selectionModel()
        selection.reset()
        self._customList.repaint()

        manufacturer = self.currentManufacturer()
        headerView = self._detectorView.horizontalHeader()
        headerView.setSectionHidden(self.MANUFACTURER_COLUMN, manufacturer != "*")

        model = self._detectorView.model()
        model.setManufacturerFilter(manufacturer)
        self._stacked.setCurrentWidget(self._modelPanel)

    def __customSelectionChanged(self, selected, deselected):
        # Clean up manufacturer selection
        selection = self._detectorView.selectionModel()
        selection.reset()
        selection = self._manufacturerList.selectionModel()
        selection.reset()
        self._detectorView.repaint()
        self._manufacturerList.repaint()

        field = self.currentCustomField()
        if field == "FILE":
            self._stacked.setCurrentWidget(self._filePanel)
        elif field == "MANUAL":
            self._stacked.setCurrentWidget(self._manualPanel)
        else:
            raise RuntimeError("Field is neither FILE nor MANUAL nor None")


class DetectorSelectorDialog(qt.QDialog):
    def __init__(self, parent=None):
        super(DetectorSelectorDialog, self).__init__(parent=parent)
        self.setWindowTitle("Detector selection")

        self.__content = DetectorSelectorDrop(self)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__content)

        buttonBox = qt.QDialogButtonBox(
            qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addWidget(buttonBox)

    def selectDetector(self, detector):
        """
        Select a detector.

        :param pyFAI.detectors.Detector detector: Detector to select in this
            dialog
        """
        self.__content.setDetector(detector)

    def selectedDetector(self):
        """
        Returns the selected detector.

        :rtype: pyFAI.detectors.Detector
        """
        return self.__content.detector()
