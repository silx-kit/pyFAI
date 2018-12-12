#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2013-2018 European Synchrotron Radiation Facility, Grenoble, France
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
from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "12/12/2018"
__status__ = "development"

import logging

logger = logging.getLogger(__name__)

from silx.gui import qt

from ..calibration.DetectorSelectorDrop import DetectorSelectorDrop
from ..dialog.OpenClDeviceDialog import OpenClDeviceDialog
from ..dialog.GeometryDialog import GeometryDialog
from ...detectors import detector_factory
from ...opencl import ocl
from ...utils import float_, str_, get_ui_file
from ...azimuthalIntegrator import AzimuthalIntegrator
from ...units import RADIAL_UNITS, to_unit
from ..calibration.model.GeometryModel import GeometryModel
from ..calibration.model.DataModel import DataModel
from ..utils import units


class WorkerConfigurator(qt.QWidget):
    """Frame displaying integration configuration which can be used as input
    param of the ~`pyFAI.worker.Worker`.
    """

    def __init__(self, parent):
        qt.QWidget.__init__(self, parent)
        filename = get_ui_file("worker-configurator.ui")
        qt.loadUi(filename, self)

        self._openclDevice = "any"
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
        self.opencl_config_button.clicked.connect(self.selectOpenClDevice)
        self.show_geometry.clicked.connect(self.showGeometry)

        # connect file selection windows
        self.file_import.clicked.connect(self.selectPoniFile)
        self.file_mask_file.clicked.connect(self.select_maskfile)
        self.file_dark_current.clicked.connect(self.select_darkcurrent)
        self.file_flat_field.clicked.connect(self.select_flatfield)

        npt_validator = qt.QIntValidator()
        npt_validator.setBottom(1)
        self.nbpt_rad.setValidator(npt_validator)
        self.nbpt_azim.setValidator(npt_validator)
        self.radial_unit.setUnits(RADIAL_UNITS.values())
        self.radial_unit.model().setValue(RADIAL_UNITS["2th_deg"])

        self.radial_unit.model().changed.connect(self.__radialUnitUpdated)
        self.__radialUnitUpdated()

        self.__configureDisabledStates()

        self.setDetector(None)

    def __configureDisabledStates(self):
        self.do_mask.clicked.connect(self.__updateDisabledStates)
        self.do_dark.clicked.connect(self.__updateDisabledStates)
        self.do_flat.clicked.connect(self.__updateDisabledStates)
        self.do_dummy.clicked.connect(self.__updateDisabledStates)
        self.do_polarization.clicked.connect(self.__updateDisabledStates)
        self.do_radial_range.clicked.connect(self.__updateDisabledStates)
        self.do_azimuthal_range.clicked.connect(self.__updateDisabledStates)
        self.do_poisson.clicked.connect(self.__updateDisabledStates)

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
        self.error_selection.setEnabled(self.do_poisson.isChecked())

    def set1dIntegrationOnly(self, only1d):
        """Enable only 1D integration for this widget."""
        if only1d:
            self.do_2D.setChecked(False)
        self.do_2D.setVisible(not only1d)
        self.__only1dIntegration = only1d

    def getRadialUnit(self):
        unit = self.radial_unit.model().value()
        if unit is not None:
            return unit
        logger.warning("Undefined unit !!! falling back on 2th_deg")
        return RADIAL_UNITS["2th_deg"]

    def getIntegrationKind(self):
        if self.do_2D.isChecked():
            return "2d"
        else:
            return "1d"

    def getCorrectSolidAngle(self):
        return bool(self.do_solid_angle.isChecked())

    def getDummy(self):
        if bool(self.do_dummy.isChecked()):
            return float_(self.val_dummy.text())
        else:
            return None

    def getDeltaDummy(self):
        if not bool(self.do_dummy.isChecked()):
            return None
        delta_dummy = str(self.delta_dummy.text())
        if delta_dummy:
            return float(delta_dummy)
        else:
            return None

    def getPolarizationFactor(self):
        if bool(self.do_polarization.isChecked()):
            return float(self.polarization_factor.value())
        else:
            return None

    def getRadialRange(self):
        if not self.do_radial_range.isChecked():
            return None
        try:
            rad_min = float_(self.radial_range_min.text())
            rad_max = float_(self.radial_range_max.text())
        except ValueError as error:
            logger.error("error in parsing radial range: %s", error)
            return None
        result = (rad_min, rad_max)
        if result == (None, None):
            result = None
        return result

    def getAzimuthalRange(self):
        if not self.do_azimuthal_range.isChecked():
            return None
        try:
            azim_min = float_(self.azimuth_range_min.text())
            azim_max = float_(self.azimuth_range_max.text())
        except ValueError as error:
            logger.error("error in parsing azimuthal range: %s", error)
            return None
        result = (azim_min, azim_max)
        if result == (None, None):
            result = None
        return result

    def getErrorModel(self):
        if self.do_poisson.isChecked():
            return "poisson"
        else:
            return None

    def getRadialNbpt(self):
        value = str(self.nbpt_rad.text()).strip()
        if value == "":
            return None
        return int(value)

    def getAzimuthalNbpt(self):
        value = str(self.nbpt_azim.text()).strip()
        if value == "":
            return None
        return int(value)

    def getConfig(self):
        """Read the configuration of the plugin and returns it as a dictionary

        :return: dict with all information.
        """
        # FIXME: "method" is used by diff_map interation, it have to be exposed

        config = {"wavelength": self.__geometryModel.wavelength().value(),
                  "dist": self.__geometryModel.distance().value(),
                  "poni1": self.__geometryModel.poni1().value(),
                  "poni2": self.__geometryModel.poni2().value(),
                  "rot1": self.__geometryModel.rotation1().value(),
                  "rot2": self.__geometryModel.rotation2().value(),
                  "rot3": self.__geometryModel.rotation3().value(),
                  "do_dummy": bool(self.do_dummy.isChecked()),
                  "do_mask": bool(self.do_mask.isChecked()),
                  "do_dark": bool(self.do_dark.isChecked()),
                  "do_flat": bool(self.do_flat.isChecked()),
                  "do_polarization": bool(self.do_polarization.isChecked()),
                  "val_dummy": self._float("val_dummy", None),
                  "delta_dummy": self._float("delta_dummy", None),
                  "mask_file": str_(self.mask_file.text()).strip(),
                  "dark_current": str_(self.dark_current.text()).strip(),
                  "flat_field": str_(self.flat_field.text()).strip(),
                  "polarization_factor": float_(self.polarization_factor.value()),
                  "do_2D": bool(self.do_2D.isChecked()),
                  "chi_discontinuity_at_0": bool(self.chi_discontinuity_at_0.isChecked()),
                  "do_solid_angle": bool(self.do_solid_angle.isChecked()),
                  "do_radial_range": bool(self.do_radial_range.isChecked()),
                  "do_azimuthal_range": bool(self.do_azimuthal_range.isChecked()),
                  "do_poisson": bool(self.do_poisson.isChecked()),
                  "radial_range_min": self._float("radial_range_min", None),
                  "radial_range_max": self._float("radial_range_max", None),
                  "azimuth_range_min": self._float("azimuth_range_min", None),
                  "azimuth_range_max": self._float("azimuth_range_max", None),
                  "do_OpenCL": bool(self.do_OpenCL.isChecked()),
                  "unit": str(self.radial_unit.model().value()),
                  }

        if self.__method is not None:
            config["method"] = self.__method

        value = self.getRadialNbpt()
        if value is not None:
            config["nbpt_rad"] = value
        value = self.getAzimuthalNbpt()
        if value is not None:
            config["nbpt_azim"] = value

        config["detector"] = self.__detector.__class__.__name__
        config["detector_config"] = self.__detector.get_config()

        return config

    def setConfig(self, dico):
        """Setup the widget from its description

        :param dico: dictionary with description of the widget
        :type dico: dict
        """
        dico = dico.copy()

        # Clean up the GUI
        self.setDetector(None)
        self.__geometryModel.wavelength().setValue(None)
        self.__geometryModel.distance().setValue(None)
        self.__geometryModel.poni1().setValue(None)
        self.__geometryModel.poni2().setValue(None)
        self.__geometryModel.rotation1().setValue(None)
        self.__geometryModel.rotation2().setValue(None)
        self.__geometryModel.rotation3().setValue(None)

        # poni file
        # NOTE: Compatibility (poni is not stored since pyFAI v0.17)
        value = dico.pop("poni", None)
        if value:
            self.loadFromPoniFile(value)

        # geometry
        if "wavelength" in dico:
            value = dico.pop("wavelength")
            self.__geometryModel.wavelength().setValue(value)
        if "dist" in dico:
            value = dico.pop("dist")
            self.__geometryModel.distance().setValue(value)
        if "poni1" in dico:
            value = dico.pop("poni1")
            self.__geometryModel.poni1().setValue(value)
        if "poni2" in dico:
            value = dico.pop("poni2")
            self.__geometryModel.poni2().setValue(value)
        if "rot1" in dico:
            value = dico.pop("rot1")
            self.__geometryModel.rotation1().setValue(value)
        if "rot2" in dico:
            value = dico.pop("rot2")
            self.__geometryModel.rotation2().setValue(value)
        if "rot3" in dico:
            value = dico.pop("rot3")
            self.__geometryModel.rotation3().setValue(value)

        # detector
        value = dico.pop("detector_config", None)
        if value:
            # NOTE: Default way to describe a detector since pyFAI 0.17
            detector_config = value
            detector_class = dico.pop("detector")
            detector = detector_factory(detector_class, config=detector_config)
            self.setDetector(detector)
        value = dico.pop("detector", None)
        if value:
            # NOTE: Previous way to describe a detector before pyFAI 0.17
            # NOTE: pixel1/pixel2/splineFile was not parsed here
            detector_name = value.lower()
            detector = detector_factory(detector_name)

            if detector_name == "detector":
                value = dico.pop("pixel1", None)
                if value:
                    detector.set_pixel1(value)
                value = dico.pop("pixel2", None)
                if value:
                    detector.set_pixel2(value)
            else:
                # Drop it as it was not really used
                _ = dico.pop("pixel1", None)
                _ = dico.pop("pixel2", None)

            splineFile = dico.pop("splineFile", None)
            if splineFile:
                detector.set_splineFile(splineFile)

            self.setDetector(detector)

        setup_data = {"do_dummy": self.do_dummy.setChecked,
                      "do_dark": self.do_dark.setChecked,
                      "do_flat": self.do_flat.setChecked,
                      "do_polarization": self.do_polarization.setChecked,
                      "val_dummy": lambda a: self.val_dummy.setText(str_(a)),
                      "delta_dummy": lambda a: self.delta_dummy.setText(str_(a)),
                      "do_mask": self.do_mask.setChecked,
                      "mask_file": lambda a: self.mask_file.setText(str_(a)),
                      "dark_current": lambda a: self.dark_current.setText(str_(a)),
                      "flat_field": lambda a: self.flat_field.setText(str_(a)),
                      "polarization_factor": self.polarization_factor.setValue,
                      "nbpt_rad": lambda a: self.nbpt_rad.setText(str_(a)),
                      "do_2D": self.do_2D.setChecked,
                      "nbpt_azim": lambda a: self.nbpt_azim.setText(str_(a)),
                      "chi_discontinuity_at_0": self.chi_discontinuity_at_0.setChecked,
                      "do_radial_range": self.do_radial_range.setChecked,
                      "do_azimuthal_range": self.do_azimuthal_range.setChecked,
                      "do_poisson": self.do_poisson.setChecked,
                      "radial_range_min": lambda a: self.radial_range_min.setText(str_(a)),
                      "radial_range_max": lambda a: self.radial_range_max.setText(str_(a)),
                      "azimuth_range_min": lambda a: self.azimuth_range_min.setText(str_(a)),
                      "azimuth_range_max": lambda a: self.azimuth_range_max.setText(str_(a)),
                      "do_solid_angle": self.do_solid_angle.setChecked,
                      "do_OpenCL": self.do_OpenCL.setChecked}

        for key, value in setup_data.items():
            if key in dico and (value is not None):
                value(dico.pop(key))

        value = dico.pop("unit", None)
        if value is not None:
            unit = to_unit(value)
            self.radial_unit.model().setValue(unit)

        value = dico.pop("method", None)
        self.__method = value

        if setup_data.get("do_OpenCL"):
            self.__openclChanged()

        if self.__only1dIntegration:
            # Force unchecked
            self.do_2D.setChecked(False)

        if len(dico) != 0:
            for key, value in dico.items():
                logger.warning("json key '%s' unused", key)

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
        popup = DetectorSelectorDrop(self)
        popupParent = self.load_detector
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
        popup.setDetector(self.__detector)
        result = dialog.exec_()
        if result:
            newDetector = popup.detector()
            self.setDetector(newDetector)

    def __radialUnitUpdated(self):
        unit = self.getRadialUnit()
        # FIXME extract the unit
        self._radialRangeUnit.setText(str(unit))

    def showGeometry(self):
        dialog = GeometryDialog(self)
        dialog.setGeometryModel(self.__geometryModel)
        dialog.setDetector(self.__detector)
        dialog.exec_()

    def selectOpenClDevice(self):
        dialog = OpenClDeviceDialog(self)
        dialog.selectDevice(self._openclDevice)
        result = dialog.exec_()
        if result:
            self._openclDevice = dialog.device()
            self.opencl_label.setDevice(self._openclDevice)

    def setDetector(self, detector):
        self.__detector = detector
        self.detector_label.setDetector(detector)

    def selectPoniFile(self):
        ponifile = self.getOpenFileName("Open a poni file")
        if ponifile is not None:
            self.loadFromPoniFile(ponifile)

    def select_maskfile(self):
        logger.debug("select_maskfile")
        maskfile = self.getOpenFileName("Open a mask image")
        if maskfile:
            self.mask_file.setText(maskfile or "")
            self.do_mask.setChecked(True)

    def select_darkcurrent(self):
        logger.debug("select_darkcurrent")
        darkcurrent = self.getOpenFileName("Open a dark image")
        if darkcurrent:
            self.dark_current.setText(str_(darkcurrent))
            self.do_dark.setChecked(True)

    def select_flatfield(self):
        logger.debug("select_flatfield")
        flatfield = self.getOpenFileName("Open a flatfield image")
        if flatfield:
            self.flat_field.setText(str_(flatfield))
            self.do_flat.setChecked(True)

    def loadFromPoniFile(self, ponifile):
        try:
            # TODO: It should not be needed to create an AI to parse a PONI file
            ai = AzimuthalIntegrator.sload(ponifile)
        except Exception as error:
            # FIXME: An error have to be displayed in the GUI
            logger.error("file %s does not look like a poni-file, error %s", ponifile, error)
            return

        model = self.__geometryModel
        model.distance().setValue(ai.dist)
        model.poni1().setValue(ai.poni1)
        model.poni2().setValue(ai.poni2)
        model.rotation1().setValue(ai.rot1)
        model.rotation2().setValue(ai.rot2)
        model.rotation3().setValue(ai.rot3)
        # TODO: why is there an underscore to _wavelength here?
        model.wavelength().setValue(ai._wavelength)

        self.setDetector(ai.detector)

    def _float(self, kw, default=0):
        fval = default
        txtval = str(self.__dict__[kw].text())
        if txtval:
            try:
                fval = float(txtval)
            except ValueError:
                logger.error("Unable to convert %s to float: %s", kw, txtval)
        return fval

    def __openclChanged(self):
        logger.debug("do_OpenCL")
        do_ocl = bool(self.do_OpenCL.isChecked())
        if do_ocl:
            if ocl is None:
                self.do_OpenCL.setChecked(0)
                return
            if self.platform.count() == 0:
                self.platform.addItems([i.name for i in ocl.platforms])

    def getMethod(self):
        """
        Return the method name for azimuthal intgration
        """
        if self.do_OpenCL.isChecked():
            platform = ocl.get_platform(self.platform.currentText())
            pid = platform.id
            did = platform.get_device(self.device.currentText()).id
            if (pid is not None) and (did is not None):
                method = "csr_ocl_%i,%i" % (pid, did)
            else:
                method = "csr_ocl"
        else:
            method = "splitbbox"
        return method
