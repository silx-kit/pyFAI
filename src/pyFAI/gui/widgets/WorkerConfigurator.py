#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2024 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""Module containing a widget to configure pyFAI integration.
"""

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/01/2025"
__status__ = "development"

import logging
import json
import os.path
import numpy
logger = logging.getLogger(__name__)

from silx.gui import qt

from ..dialog.DetectorSelectorDialog import DetectorSelectorDialog
from ..dialog.OpenClDeviceDialog import OpenClDeviceDialog
from ..dialog.GeometryDialog import GeometryDialog
from ..dialog.IntegrationMethodDialog import IntegrationMethodDialog
from ...utils import float_, str_, get_ui_file
from ...units import RADIAL_UNITS, to_unit
from ..model.GeometryModel import GeometryModel
from ..model.DataModel import DataModel
from ..utils import units
from ...utils import stringutil
from ..utils import FilterBuilder
from ..model.ImageModel import ImageFilenameModel
from ..utils import validators
from ...io.ponifile import PoniFile
from ...io import integration_config
from ... import method_registry
from ...containers import PolarizationDescription
from ... import detector_factory
from ...integrator import load_engines


class _WorkerModel(object):

    def __init__(self):
        self.maskFileModel = ImageFilenameModel()
        self.darkFileModel = ImageFilenameModel()
        self.flatFileModel = ImageFilenameModel()


class WorkerConfigurator(qt.QWidget):
    """Frame displaying integration configuration which can be used as input
    param of the ~`pyFAI.worker.Worker`.
    """

    def __init__(self, parent=None):
        qt.QWidget.__init__(self, parent)
        filename = get_ui_file("worker-configurator.ui")
        qt.loadUi(filename, self)

        self.__model = _WorkerModel()

        self.__openclDevice = None
        self.__method = None

        self.__geometryModel = GeometryModel()
        self.__detector = None
        self.__only1dIntegration = False

        self.geometry_label.setGeometryModel(self.__geometryModel)

        # Connect widget to edit the wavelength
        wavelengthUnit = DataModel()
        wavelengthUnit.setValue(units.Unit.ENERGY)
        self.wavelengthEdit.setModel(self.__geometryModel.wavelength())
        self.wavelengthEdit.setDisplayedUnitModel(wavelengthUnit)
        self.wavelengthEdit.setModelUnit(units.Unit.METER_WL)
        self.wavelengthUnit.setUnitModel(wavelengthUnit)
        self.wavelengthUnit.setUnitEditable(True)

        self.load_detector.clicked.connect(self.selectDetector)
        self.opencl_config_button.clicked.connect(self.selectOpenclDevice)
        self.method_config_button.clicked.connect(self.selectMethod)
        self.show_geometry.clicked.connect(self.selectGeometry)

        # Connect file selection windows
        self.file_import.clicked.connect(self.__selectFile)
        self.poni_save.clicked.connect(self.__savePoni)

        # Connect mask/dark/flat
        self.mask_file.setModel(self.__model.maskFileModel)
        self.file_mask_file.setDialogTitle("Open a mask image")
        self.file_mask_file.setModel(self.__model.maskFileModel)
        self.__model.maskFileModel.changed.connect(self.__maskFileChanged)
        self.dark_current.setModel(self.__model.darkFileModel)
        self.file_dark_current.setDialogTitle("Open a dark image")
        self.file_dark_current.setModel(self.__model.darkFileModel)
        self.__model.darkFileModel.changed.connect(self.__darkFileChanged)
        self.flat_field.setModel(self.__model.flatFileModel)
        self.file_flat_field.setDialogTitle("Open a flatfield image")
        self.file_flat_field.setModel(self.__model.flatFileModel)
        self.__model.flatFileModel.changed.connect(self.__flatFileChanged)

        self.do_2D.toggled.connect(self.__dimChanged)

        npt_validator = qt.QIntValidator()
        npt_validator.setBottom(1)
        self.nbpt_rad.setValidator(npt_validator)
        self.nbpt_azim.setValidator(npt_validator)
        self.radial_unit.setUnits(RADIAL_UNITS.values())
        self.radial_unit.model().setValue(RADIAL_UNITS["2th_deg"])

        self.radial_unit.setShortNameDisplay(True)
        self.radial_unit.model().changed.connect(self.__radialUnitUpdated)
        self.__radialUnitUpdated()

        doubleOrEmptyValidator = validators.AdvancedDoubleValidator(self)
        doubleOrEmptyValidator.setAllowEmpty(True)
        self.normalization_factor.setValidator(doubleOrEmptyValidator)
        self.normalization_factor.setText("1.0")

        for value in [None, "poisson", "azimuthal"]:
            if value:
                text = value.capitalize()
            else:
                text = ""
            self.error_model.addItem(text, value)
        self.error_model.setCurrentIndex(0)

        for value in ["integrate",
                      "sigma_clip_ng",
                      ]:
            text = value
            self.integrator_name.addItem(text, value)
        self.integrator_name.setCurrentIndex(0)
        self.sigmaclip_threshold.setText("5.0")
        self.sigmaclip_maxiter.setText("5")

        self.__configureDisabledStates()

        self.setDetector(None)
        self.__setMethod(None)

    def __configureDisabledStates(self):
        self.do_mask.clicked.connect(self.__updateDisabledStates)
        self.do_dark.clicked.connect(self.__updateDisabledStates)
        self.do_flat.clicked.connect(self.__updateDisabledStates)
        self.do_dummy.clicked.connect(self.__updateDisabledStates)
        self.do_polarization.clicked.connect(self.__updateDisabledStates)
        self.do_radial_range.clicked.connect(self.__updateDisabledStates)
        self.do_azimuthal_range.clicked.connect(self.__updateDisabledStates)
        self.do_normalization.clicked.connect(self.__updateDisabledStates)
        self.integrator_name.currentTextChanged.connect(self.__updateDisabledStates)

        self.__updateDisabledStates()

    def __updateDisabledStates(self):
        self.mask_file.setEnabled(self.do_mask.isChecked())
        self.dark_current.setEnabled(self.do_dark.isChecked())
        self.flat_field.setEnabled(self.do_flat.isChecked())
        self.val_dummy.setEnabled(self.do_dummy.isChecked())
        self.delta_dummy.setEnabled(self.do_dummy.isChecked())
        self.polarization_factor.setEnabled(self.do_polarization.isChecked())
        enabled = self.do_radial_range.isChecked()
        self.radial_range_min.setEnabled(enabled)
        self.radial_range_max.setEnabled(enabled)
        enabled = self.do_azimuthal_range.isChecked()
        self.azimuth_range_min.setEnabled(enabled)
        self.azimuth_range_max.setEnabled(enabled)
        self.normalization_factor.setEnabled(self.do_normalization.isChecked())
        self.monitor_name.setEnabled(self.do_normalization.isChecked())
        self.sigmaclip_threshold.setEnabled(self.integrator_name.currentText() == "sigma_clip_ng")
        self.sigmaclip_maxiter.setEnabled(self.integrator_name.currentText() == "sigma_clip_ng")

    def set1dIntegrationOnly(self, only1d):
        """Enable only 1D integration for this widget."""
        if only1d:
            self.do_2D.setChecked(False)
        self.do_2D.setVisible(not only1d)
        self.__only1dIntegration = only1d

    def __getRadialUnit(self):
        unit = self.radial_unit.model().value()
        if unit is not None:
            return unit
        logger.warning("Undefined unit !!! falling back on 2th_deg")
        return RADIAL_UNITS["2th_deg"]

    def __getRadialNbpt(self):
        value = str(self.nbpt_rad.text()).strip()
        if value == "":
            return None
        return int(value)

    def __getAzimuthalNbpt(self):
        value = str(self.nbpt_azim.text()).strip()
        if value == "":
            return None
        return int(value)

    def getPoniDict(self):
        poni = {}
        poni["wavelength"] = self.__geometryModel.wavelength().value()
        poni["dist"] = self.__geometryModel.distance().value()
        poni["poni1"] = self.__geometryModel.poni1().value()
        poni["poni2"] = self.__geometryModel.poni2().value()
        poni["rot1"] = self.__geometryModel.rotation1().value()
        poni["rot2"] = self.__geometryModel.rotation2().value()
        poni["rot3"] = self.__geometryModel.rotation3().value()
        if self.__detector is not None:
            poni["detector"] = self.__detector.__class__.__name__
            poni["detector_config"] = self.__detector.get_config()
        return poni

    def getWorkerConfig(self):
        """Read the configuration of the plugin and returns it as a WorkerConfig instance

        :return: WorkerConfig instance
        """

        def splitFiles(filenames):
            """In case files was provided with comma.

            The file brower was in this case not working, but the returned
            config will be valid.
            """
            filenames = filenames.strip()
            if filenames == "":
                return None
            return [name.strip() for name in filenames.split("|")]

        wc = integration_config.WorkerConfig(application="pyfai-integrate",
                                             poni=self.getPoniDict())
        # pre-processing
        if self.do_mask.isChecked():
            wc.mask_image = str_(self.mask_file.text()).strip()
        if self.do_dark.isChecked():
            wc.dark_current_image = splitFiles(self.dark_current.text())
        if self.do_flat.isChecked():
            wc.flat_field_image = splitFiles(self.flat_field.text())
        if self.do_polarization.isChecked():
            wc.polarization_factor = PolarizationDescription(float_(self.polarization_factor.value()), 0.0)
        if self.do_dummy.isChecked():
            wc.val_dummy = self._float("val_dummy", None)
            wc.delta_dummy = self._float("delta_dummy", None)

        # integration
        if self.do_2D.isChecked():
            wc.nbpt_azim = self.__getAzimuthalNbpt()
        wc.nbpt_rad = self.__getRadialNbpt()
        wc.unit = str(self.radial_unit.model().value())
        if self.do_radial_range.isChecked():
            wc.radial_range = [self._float("radial_range_min", -numpy.inf),
                               self._float("radial_range_max", numpy.inf)]
        if self.do_azimuthal_range.isChecked():
            wc.azimuth_range = [self._float("azimuth_range_min", -numpy.inf),
                                self._float("azimuth_range_max", numpy.inf)]

        # processing-config
        wc.chi_discontinuity_at_0 = bool(self.chi_discontinuity_at_0.isChecked())
        wc.do_solid_angle = bool(self.do_solid_angle.isChecked())
        wc.error_model = self.error_model.currentData()

        method = self.__method
        if method is not None:
            wc.method = (method.split, method.algo, method.impl)
            if method.impl == "opencl":
                wc.opencl_device = self.__openclDevice

        if self.do_normalization.isChecked():
            value = self.normalization_factor.text()
            if value != "":
                try:
                    value = float(value)
                except ValueError:
                    value = None
                if value not in [1.0, None]:
                    wc.normalization_factor = value

            value = self.monitor_name.text()
            if value != "":
                value = str(value)
                wc.monitor_name = value

        if self.integrator_name.currentText() == "sigma_clip_ng":
            wc.nbpt_azim = 1
            wc.integrator_method = "sigma_clip_ng"
            wc.extra_options = {"thres": float(self.sigmaclip_threshold.text()),
                                "max_iter": float(self.sigmaclip_maxiter.text()),
                               }
        return wc

    def getConfig(self):
        """Read the configuration of the plugin and returns it as a dictionary

        :return: dict with all information
        """
        return self.getWorkerConfig().as_dict()

    def setConfig(self, dico):
        """Setup the widget from its description

        :param dico: dictionary with description of the widget
        :type dico: dict
        """

        self.setWorkerConfig(integration_config.WorkerConfig.from_dict(dico, inplace=False))

    def setWorkerConfig(self, wc):
        """Setup the widget from its description

        :param wc: WorkerConfig instance with the description of the widget
        :type dico: WorkerConfig
        """

        def normalizeFiles(filenames):
            """Normalize different versions of the filename list.

            FIXME: The file brower will not work, but the returned config will
            be valid
            """
            if filenames is None:
                return ""
            if isinstance(filenames, list):
                return "|".join(filenames)
            filenames = filenames.strip()
            return filenames

        # Clean up the GUI
        self.setDetector(None)
        self.__geometryModel.wavelength().setValue(None)
        self.__geometryModel.distance().setValue(None)
        self.__geometryModel.poni1().setValue(None)
        self.__geometryModel.poni2().setValue(None)
        self.__geometryModel.rotation1().setValue(None)
        self.__geometryModel.rotation2().setValue(None)
        self.__geometryModel.rotation3().setValue(None)

        poni_dict = wc.poni
        if isinstance(poni_dict, dict):
            if "wavelength" in poni_dict:
                self.__geometryModel.wavelength().setValue(poni_dict["wavelength"])
            if "dist" in poni_dict:
                self.__geometryModel.distance().setValue(poni_dict["dist"])
            if "poni1" in poni_dict:
                self.__geometryModel.poni1().setValue(poni_dict["poni1"])
            if "poni2" in poni_dict:
                self.__geometryModel.poni2().setValue(poni_dict["poni2"])
            if "rot1" in poni_dict:
                self.__geometryModel.rotation1().setValue(poni_dict["rot1"])
            if "rot2" in poni_dict:
                self.__geometryModel.rotation2().setValue(poni_dict["rot2"])
            if "rot3" in poni_dict:
                self.__geometryModel.rotation3().setValue(poni_dict["rot3"])

        # reader = integration_config.ConfigurationReader(dico)

        # detector
        if "detector" in poni_dict:
            detector = detector_factory(poni_dict["detector"], poni_dict.get("detector_config"))
            self.setDetector(detector)

        self.val_dummy.setText(str_(wc.val_dummy))
        self.delta_dummy.setText(str_(wc.delta_dummy))
        self.__model.maskFileModel.setFilename(str_(wc.mask_image))
        self.__model.darkFileModel.setFilename(normalizeFiles(wc.dark_current_image))
        self.__model.flatFileModel.setFilename(normalizeFiles(wc.flat_field_image))
        if isinstance(wc.polarization_factor, float):
            self.polarization_factor.setValue(wc.polarization_factor)
        elif isinstance(wc.polarization_factor, (tuple, list)):
            self.polarization_factor.setValue(wc.polarization_factor[0])
        self.nbpt_rad.setText(str_(wc.nbpt_rad))
        self.nbpt_azim.setText(str_(wc.nbpt_azim))
        self.chi_discontinuity_at_0.setChecked(wc.chi_discontinuity_at_0)
        self.radial_range_min.setText(str_(wc.radial_range_min))
        self.radial_range_max.setText(str_(wc.radial_range_max))
        self.azimuth_range_min.setText(str_(wc.azimuth_range_min))
        self.azimuth_range_max.setText(str_(wc.azimuth_range_max))
        self.do_solid_angle.setChecked(wc.do_solid_angle)
        self.do_dummy.setChecked(wc.do_dummy)
        self.do_dark.setChecked(wc.do_dark)
        self.do_flat.setChecked(wc.do_flat)
        self.do_polarization.setChecked(wc.do_polarization)
        self.do_mask.setChecked(wc.do_mask)
        self.do_radial_range.setChecked(wc.do_radial_range)
        self.do_azimuthal_range.setChecked(wc.do_azimuthal_range)
        self.__setNormalization(wc.normalization_factor, wc.monitor_name)

        value = wc.unit
        if value is not None:
            unit = to_unit(value)
            self.radial_unit.model().setValue(unit)

        value = wc.error_model
        index = self.error_model.findData(value)
        self.error_model.setCurrentIndex(index)

        dim = 2 if wc.do_2D else 1
        method = wc.method
        target = wc.opencl_device
        if isinstance(target, list):
            target = tuple(target)

        if method is None:
            lngm = load_engines.PREFERED_METHODS_2D[0] if dim == 2 else load_engines.PREFERED_METHODS_1D[0]
            method = lngm.method
        elif isinstance(method, (str,)):
            method = method_registry.Method.parsed(method)
            method = method.fixed(dim=dim, target=target)
        elif isinstance(method, (list, tuple)):
            if len(method) == 3:
                split, algo, impl = method
                method = method_registry.Method(dim, split, algo, impl, target)
            elif 3 < len(method) <= 5:
                method = method_registry.Method(*method)
            else:
                raise TypeError(f"Method size {len(method)} is unsupported, method={method}.")

        self.__setMethod(method)
        self.__setOpenclDevice(method.target)

        self.do_2D.setChecked(wc.do_2D)
        if self.__only1dIntegration:
            # Force unchecked
            self.do_2D.setChecked(False)

        integrator_name = wc.integrator_method or "integrate"
        self.integrator_name.setCurrentText(integrator_name)
        if integrator_name.startswith("sigma_clip"):
            extra_options = wc.extra_options or {}
            self.sigmaclip_threshold.setText(str(extra_options.get("thres", 5.0)))
            self.sigmaclip_maxiter.setText(str(extra_options.get("max_iter", 5)))

        self.__updateDisabledStates()

    def getOpenFileName(self, title):
        """Display a dialog to select a filename and return it.

        Returns None if nothing selected.

        This code is compatible PyQt4/PyQt5 which is not the case for static
        functions provided by `qt.QFileDialog`.
        """
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle(title)
        dialog.setModal(True)
        dialog.setFileMode(qt.QFileDialog.ExistingFile)

        result = dialog.exec_()
        if not result:
            return None

        filename = dialog.selectedFiles()[0]
        return filename

    def selectDetector(self):
        dialog = DetectorSelectorDialog(self)
        dialog.selectDetector(self.__detector)
        result = dialog.exec_()
        if result:
            newDetector = dialog.selectedDetector()
            self.setDetector(newDetector)

    def __radialUnitUpdated(self):
        unit = self.__getRadialUnit()
        # FIXME extract the unit
        if unit.unit_symbol == "?":
            name = stringutil.latex_to_unicode(unit.short_name)
            toolTip = "The unit for the quantity %s is not expressible." % name
        else:
            toolTip = ""
        symbol = stringutil.latex_to_unicode(unit.unit_symbol)
        self._radialRangeUnit.setText(symbol)
        self._radialRangeUnit.setToolTip(toolTip)

    def selectGeometry(self):
        dialog = GeometryDialog(self)
        dialog.setGeometryModel(self.__geometryModel)
        dialog.setDetector(self.__detector)
        result = dialog.exec_()
        if result:
            geometry = dialog.geometryModel()
            if geometry.isValid(checkWaveLength=False):
                self.__geometryModel.setFrom(geometry)
            else:
                qt.QMessageBox.critical(self, "Geometry ignored", "Provided geometry is not consistent")

    def selectOpenclDevice(self):
        dialog = OpenClDeviceDialog(self)
        dialog.selectDevice(self.__openclDevice)
        result = dialog.exec_()
        if result:
            device = dialog.device()
            self.__setOpenclDevice(device)

    def selectMethod(self):
        dialog = IntegrationMethodDialog(self)
        dialog.selectMethod(self.__method)
        result = dialog.exec_()
        if result:
            method = dialog.selectedMethod()
            dim = 2 if self.do_2D.isChecked() else 1
            method = method.fixed(dim=dim)
            self.__setMethod(method)

    def __setNormalization(self, normalizationFactor, monitorName):
        factor = str(normalizationFactor) if normalizationFactor is not None else "1.0"
        self.normalization_factor.setText(factor)
        name = str(monitorName) if monitorName is not None else ""
        self.monitor_name.setText(name)
        enabled = normalizationFactor is not None or monitorName is not None
        self.do_normalization.setChecked(enabled)

    def setDetector(self, detector):
        self.__detector = detector
        self.detector_label.setDetector(detector)

    def __selectFile(self):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Open a poni file")
        dialog.setModal(True)
        dialog.setAcceptMode(qt.QFileDialog.AcceptOpen)

        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("PONI files", "poni")
        builder.addFileFormat("JSON files", "json")
        dialog.setNameFilters(builder.getFilters())

        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        if filename.endswith(".json"):
            self.loadFromJsonFile(filename)
        elif filename.endswith(".poni"):
            self.loadFromPoniFile(filename)
        else:
            logger.error("File %s unsupported", filename)

    def __savePoni(self):
        dialog = qt.QFileDialog(self)
        dialog.setWindowTitle("Save poni file")
        dialog.setModal(True)
        dialog.setAcceptMode(qt.QFileDialog.AcceptSave)

        builder = FilterBuilder.FilterBuilder()
        builder.addFileFormat("PONI files", "poni")
#         builder.addFileFormat("JSON files", "json")
        dialog.setNameFilters(builder.getFilters())

        result = dialog.exec_()
        if not result:
            return

        filename = dialog.selectedFiles()[0]
        poni = PoniFile(data=self.getPoniDict())
        with open(filename, 'w') as fd:
            poni.write(fd)

    def __maskFileChanged(self):
        model = self.__model.maskFileModel
        if model.hasFilename():
            self.do_mask.setChecked(True)
            self.__updateDisabledStates()

    def __darkFileChanged(self):
        model = self.__model.darkFileModel
        if model.hasFilename():
            self.do_dark.setChecked(True)
            self.__updateDisabledStates()

    def __flatFileChanged(self):
        model = self.__model.flatFileModel
        if model.hasFilename():
            self.do_flat.setChecked(True)
            self.__updateDisabledStates()

    def loadFromJsonFile(self, filename):
        """Initialize the widget using a json file."""
        if not os.path.isfile(filename):
            logger.error("No such file: %s", filename)
            return
        with open(filename) as f:
            config = json.load(f)
        self.setConfig(config)

    def loadFromPoniFile(self, ponifile):
        try:
            poni = PoniFile(ponifile)
        except Exception as error:
            # FIXME: An error has to be displayed in the GUI
            logger.error("file %s does not look like a poni-file, error %s", ponifile, error)
            return

        model = self.__geometryModel
        model.distance().setValue(poni.dist)
        model.poni1().setValue(poni.poni1)
        model.poni2().setValue(poni.poni2)
        model.rotation1().setValue(poni.rot1)
        model.rotation2().setValue(poni.rot2)
        model.rotation3().setValue(poni.rot3)
        model.wavelength().setValue(poni.wavelength)

        self.setDetector(poni.detector)

    def _float(self, kw, default=0):
        fval = default
        txtval = str(self.__dict__[kw].text())
        if txtval:
            try:
                fval = float(txtval)
            except ValueError:
                logger.error("Unable to convert %s to float: %s", kw, txtval)
        return fval

    def __setOpenclDevice(self, device):
        self.__openclDevice = device
        self.opencl_label.setDevice(device)

    def __dimChanged(self):
        if self.__method is None:
            return
        _dim, split, algo, impl, target = self.__method
        dim = 2 if self.do_2D.isChecked() else 1
        method = method_registry.Method(dim=dim, split=split, algo=algo, impl=impl, target=target)
        self.__setMethod(method)

    def __setMethod(self, method):
        self.__method = method
        self.methodLabel.setMethod(method)
        openclEnabled = (method.impl if method is not None else "") == "opencl"
        self.opencl_title.setEnabled(openclEnabled)
        self.opencl_label.setEnabled(openclEnabled)
        self.opencl_config_button.setEnabled(openclEnabled)
