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

"""pyFAI-integrate

A graphical tool for performing azimuthal integration on series of files.

"""
from __future__ import absolute_import, print_function, with_statement, division

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/12/2018"
__status__ = "development"

import logging
import json
import os
import time
import threading
import os.path as op
import numpy

import fabio
from silx.gui import qt

logger = logging.getLogger(__name__)

from .. import worker
from .calibration.DetectorSelectorDrop import DetectorSelectorDrop
from .dialog.OpenClDeviceDialog import OpenClDeviceDialog
from .dialog.GeometryDialog import GeometryDialog
from ..detectors import detector_factory
from ..opencl import ocl
from ..utils import float_, int_, str_, get_ui_file
from ..io import HDF5Writer
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..units import RADIAL_UNITS, to_unit
from ..third_party import six
from .utils import projecturl
from .calibration.model.GeometryModel import GeometryModel
from .calibration.model.DataModel import DataModel
from .utils import units


class IntegrationDialog(qt.QWidget):
    """Dialog to configure an azimuthal integration.
    """

    def __init__(self, input_data=None, output_path=None, output_format=None, slow_dim=None, fast_dim=None, json_file=".azimint.json"):
        qt.QWidget.__init__(self)
        filename = get_ui_file("integration.ui")
        qt.loadUi(filename, self)

        self.units = {}
        self.input_data = input_data
        self.output_path = output_path
        self.output_format = output_format
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        self.name = None
        self._openclDevice = "any"
        self._sem = threading.Semaphore()
        self.json_file = json_file

        self.__geometryModel = GeometryModel()
        self.__detector = None

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
        self.file_import.clicked.connect(self.select_ponifile)
        self.file_mask_file.clicked.connect(self.select_maskfile)
        self.file_dark_current.clicked.connect(self.select_darkcurrent)
        self.file_flat_field.clicked.connect(self.select_flatfield)

        # connect button bar
        # FIXME: Do it
        # self.okButton = self.buttonBox.button(qt.QDialogButtonBox.Ok)
        # self.saveButton = self.buttonBox.button(qt.QDialogButtonBox.Save)
        # self.resetButton = self.buttonBox.button(qt.QDialogButtonBox.Reset)
        # self.cancelButton = self.buttonBox.button(qt.QDialogButtonBox.Cancel)
        # self.okButton.clicked.connect(self.proceed)
        self.save_json_button.clicked.connect(self.save_config)
        # self.buttonBox.helpRequested.connect(self.help)
        # self.cancelButton.clicked.connect(self.die)
        # self.resetButton.clicked.connect(self.restore)

        # FIXME: Do it
        # self.do_OpenCL.clicked.connect(self.openCL_changed)
        # self.platform.currentIndexChanged.connect(self.platform_changed)
        npt_validator = qt.QIntValidator()
        npt_validator.setBottom(1)
        self.nbpt_rad.setValidator(npt_validator)
        self.nbpt_azim.setValidator(npt_validator)
        self.radial_unit.setUnits(RADIAL_UNITS.values())
        self.radial_unit.model().setValue(RADIAL_UNITS["2th_deg"])

        self.radial_unit.model().changed.connect(self.__radialUnitUpdated)
        self.__radialUnitUpdated()

        self.__configureDisabledStates()

        # FIXME: Do it
        # self.progressBar.setValue(0)
        self.hdf5_path = None

        self.setDetector(None)
        if self.json_file is not None:
            self.restore(self.json_file)

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

    def __get_unit(self):
        unit = self.radial_unit.model().value()
        if unit is not None:
            return unit
        logger.warning("Undefined unit !!! falling back on 2th_deg")
        return RADIAL_UNITS["2th_deg"]

    def __get_correct_solid_angle(self):
        return bool(self.do_solid_angle.isChecked())

    def __get_dummy(self):
        if bool(self.do_dummy.isChecked()):
            return float_(self.val_dummy.text())
        else:
            return None

    def __get_delta_dummy(self):
        if not bool(self.do_dummy.isChecked()):
            return None
        delta_dummy = str(self.delta_dummy.text())
        if delta_dummy:
            return float(delta_dummy)
        else:
            return None

    def __get_polarization_factor(self):
        if bool(self.do_polarization.isChecked()):
            return float(self.polarization_factor.value())
        else:
            return None

    def __get_radial_range(self):
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

    def __get_azimuth_range(self):
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

    def __get_error_model(self):
        if self.do_poisson.isChecked():
            return "poisson"
        else:
            return None

    def __get_nbpt_rad(self):
        nbpt_rad = str(self.nbpt_rad.text()).strip()
        if not nbpt_rad:
            return None
        return int(nbpt_rad)

    def __get_nbpt_azim(self):
        return int(str(self.nbpt_azim.text()).strip())

    def proceed(self):
        with self._sem:
            out = None
            config = self.dump()
            logger.debug("Let's work a bit")
            ai = worker.make_ai(config)

            # Default Keyword arguments
            kwarg = {
                "unit": self.__get_unit(),
                "dummy": self.__get_dummy(),
                "delta_dummy": self.__get_delta_dummy(),
                "polarization_factor": self.__get_polarization_factor(),
                "filename": None,
                "safe": False,
                "correctSolidAngle": self.__get_correct_solid_angle(),
                "error_model": self.__get_error_model(),
                "method": self.get_method(),
                "npt_rad": self.__get_nbpt_rad()}

            if kwarg["npt_rad"] is None:
                message = "You must provide the number of output radial bins !"
                qt.QMessageBox.warning(self, "PyFAI integrate", message)
                return {}

            if self.do_2D.isChecked():
                kwarg["npt_azim"] = self.__get_nbpt_azim()
            if self.do_radial_range.isChecked():
                kwarg["radial_range"] = self.__get_radial_range()
            if self.do_azimuthal_range.isChecked():
                kwarg["azimuth_range"] = self.__get_azimuth_range()

            logger.info("Parameters for integration:%s%s" % (os.linesep,
                        os.linesep.join(["\t%s:\t%s" % (k, v) for k, v in kwarg.items()])))

            logger.debug("processing %s", self.input_data)
            start_time = time.time()
            if self.input_data is None or len(self.input_data) == 0:
                logger.warning("No input data to process")
                return

            elif "ndim" in dir(self.input_data) and self.input_data.ndim == 3:
                # We have a numpy array of dim3
                w = worker.Worker(azimuthalIntegrator=ai)
                try:
                    w.nbpt_rad = self.__get_nbpt_rad()
                    w.unit = self.__get_unit()
                    w.dummy = self.__get_dummy()
                    w.delta_dummy = self.__get_delta_dummy()
                    w.polarization_factor = self.__get_polarization_factor()
                    # NOTE: previous implementation was using safe=False, the worker use safe=True
                    w.correct_solid_angle = self.__get_correct_solid_angle()
                    w.error_model = self.__get_error_model()
                    w.method = self.get_method()
                    w.safe = False
                    if self.do_2D.isChecked():
                        w.nbpt_azim = self.__get_nbpt_azim()
                    else:
                        w.nbpt_azim = 1
                    w.radial_range = self.__get_radial_range()
                    w.azimuth_range = self.__get_azimuth_range()
                except RuntimeError as e:
                    qt.QMessageBox.warning(self, "PyFAI integrate", e.args[0] + ". Action aboreded.")
                    return {}

                if self.do_2D.isChecked():
                    out = numpy.zeros((self.input_data.shape[0], w.nbpt_azim, w.nbpt_rad), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        data = self.input_data[i]
                        out[i] = w.process(data)
                else:
                    out = numpy.zeros((self.input_data.shape[0], w.nbpt_rad), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        data = self.input_data[i]
                        result = w.process(data)
                        result = result.T[1]
                        out[i] = result

            elif "__len__" in dir(self.input_data):
                out = []
                if self.hdf5_path:
                    import h5py
                    hdf5 = h5py.File(self.output_path)
                    if self.fast_dim:
                        if "npt_azim" in kwarg:
                            _ds = hdf5.create_dataset("diffraction", (1, self.fast_dim, kwarg["npt_azim"], kwarg["npt_rad"]),
                                                      dtype=numpy.float32,
                                                      chunks=(1, self.fast_dim, kwarg["npt_azim"], kwarg["npt_rad"]),
                                                      maxshape=(None, self.fast_dim, kwarg["npt_azim"], kwarg["npt_rad"]))
                        else:
                            _ds = hdf5.create_dataset("diffraction", (1, self.fast_dim, kwarg["npt_rad"]),
                                                      dtype=numpy.float32,
                                                      chunks=(1, self.fast_dim, kwarg["npt_rad"]),
                                                      maxshape=(None, self.fast_dim, kwarg["npt_rad"]))
                    else:
                        if "npt_azim" in kwarg:
                            _ds = hdf5.create_dataset("diffraction", (1, kwarg["npt_azim"], kwarg["npt_rad"]),
                                                      dtype=numpy.float32,
                                                      chunks=(1, kwarg["npt_azim"], kwarg["npt_rad"]),
                                                      maxshape=(None, kwarg["npt_azim"], kwarg["npt_rad"]))
                        else:
                            _ds = hdf5.create_dataset("diffraction", (1, kwarg["npt_rad"]),
                                                      dtype=numpy.float32,
                                                      chunks=(1, kwarg["npt_rad"]),
                                                      maxshape=(None, kwarg["npt_rad"]))

                for i, item in enumerate(self.input_data):
                    self.progressBar.setValue(100.0 * i / len(self.input_data))
                    logger.debug("processing %s", item)
                    if isinstance(item, (six.text_type, six.binary_type)) and op.exists(item):
                        fab_img = fabio.open(item)
                        multiframe = (fab_img.nframes > 1)
                        kwarg["data"] = fab_img.data
                        kwarg["metadata"] = fab_img.header
                        if self.hdf5_path is None:
                            if self.output_path and op.isdir(self.output_path):
                                outpath = op.join(self.output_path, op.splitext(op.basename(item))[0])
                            else:
                                outpath = op.splitext(item)[0]
                            if "npt_azim" in kwarg and not multiframe:
                                kwarg["filename"] = outpath + ".azim"
                            else:
                                kwarg["filename"] = outpath + ".dat"
                    else:
                        logger.warning("item is not a file ... guessing it is a numpy array")
                        kwarg["data"] = item
                        kwarg["filename"] = None
                        multiframe = False
                    if multiframe:
                        if kwarg["filename"]:
                            outpath = op.splitext(kwarg["filename"])[0]
                        kwarg["filename"] = None
                        writer = HDF5Writer(outpath + "_pyFAI.h5")
                        writer.init(config)
                        for i in range(fab_img.nframes):
                            frame = fab_img.getframe(i)
                            kwarg["data"] = frame.data
                            kwarg["metadata"] = frame.header
                            if "npt_azim" in kwarg:
                                res = ai.integrate2d(**kwarg)
                            else:
                                if "npt_rad" in kwarg:  # convert npt_rad -> npt
                                    kwarg["npt"] = kwarg.pop("npt_rad")
                                res = ai.integrate1d(**kwarg)
                            writer.write(res, index=i)
                        writer.close()
                    else:
                        if kwarg.get("npt_azim"):
                            res = ai.integrate2d(**kwarg)
                        else:
                            if "npt_rad" in kwarg:  # convert npt_rad -> npt
                                kwarg["npt"] = kwarg.pop("npt_rad")
                            res = ai.integrate1d(**kwarg)
                    out.append(res)

                    # TODO manage HDF5 stuff !!!
            logger.info("Processing Done in %.3fs !", time.time() - start_time)
            self.progressBar.setValue(100)
        self.die()
        return out

    def die(self):
        logger.debug("bye bye")
        self.deleteLater()

    def help(self):
        logger.debug("Please, help")
        url = projecturl.get_documentation_url("man/pyFAI-integrate.html")
        qt.QDesktopServices.openUrl(qt.QUrl(url))

    def get_config(self):
        """Read the configuration of the plugin and returns it as a dictionary

        :return: dict with all information.
        """

        to_save = {"wavelength": self.__geometryModel.wavelength().value(),
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
                   "nbpt_rad": int_(self.nbpt_rad.text()),
                   "do_2D": bool(self.do_2D.isChecked()),
                   "nbpt_azim": int_(self.nbpt_azim.text()),
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

        to_save["detector"] = self.__detector.__class__.__name__
        to_save["detector_config"] = self.__detector.get_config()

        return to_save

    def dump(self, filename=None):
        """
        Dump the status of the current widget to a file in JSON

        :param filename: path where to save the config
        :type filename: string
        :return: dict with configuration
        """
        to_save = self.get_config()
        if filename is None:
            filename = self.json_file
        if filename is not None:
            logger.info("Dump to %s", filename)
            try:
                with open(filename, "w") as myFile:
                    json.dump(to_save, myFile, indent=4)
            except IOError as error:
                logger.error("Error while saving config: %s", error)
            else:
                logger.debug("Saved")
        return to_save

    def restore(self, filename=".azimint.json"):
        """Restore from JSON file the status of the current widget

        :param filename: path where the config was saved
        :type filename: str
        """
        logger.debug("Restore from %s", filename)
        if not op.isfile(filename):
            logger.error("No such file: %s", filename)
            return
        with open(filename) as f:
            data = json.load(f)
        self.set_config(data)

    def set_config(self, dico):
        """Setup the widget from its description

        :param dico: dictionary with description of the widget
        :type dico: dict
        """
        dico = dico.copy()

        # poni file
        # NOTE: Compatibility (poni is not stored since pyFAI v0.17)
        value = dico.pop("poni", None)
        if value:
            self.set_ponifile(value)

        # geometry
        value = dico.pop("wavelength", None)
        self.__geometryModel.wavelength().setValue(value)
        value = dico.pop("dist", None)
        self.__geometryModel.distance().setValue(value)
        value = dico.pop("poni1", None)
        self.__geometryModel.poni1().setValue(value)
        value = dico.pop("poni2", None)
        self.__geometryModel.poni2().setValue(value)
        value = dico.pop("rot1", None)
        self.__geometryModel.rotation1().setValue(value)
        value = dico.pop("rot2", None)
        self.__geometryModel.rotation2().setValue(value)
        value = dico.pop("rot3", None)
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

        if setup_data.get("do_OpenCL"):
            self.openCL_changed()

        if len(dico) != 0:
            for key, value in dico.items():
                logger.warning("json key '%s' unused", key)

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
        unit = self.__get_unit()
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

    def select_ponifile(self):
        ponifile = self.getOpenFileName("Open a poni file")
        if ponifile is not None:
            self.set_ponifile(ponifile)

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

    def set_ponifile(self, ponifile):
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

    def set_input_data(self, stack, stack_name=None):
        self.input_data = stack
        self.name = stack_name

    setStackDataObject = set_input_data

    def _float(self, kw, default=0):
        fval = default
        txtval = str(self.__dict__[kw].text())
        if txtval:
            try:
                fval = float(txtval)
            except ValueError:
                logger.error("Unable to convert %s to float: %s", kw, txtval)
        return fval

    def openCL_changed(self):
        logger.debug("do_OpenCL")
        do_ocl = bool(self.do_OpenCL.isChecked())
        if do_ocl:
            if ocl is None:
                self.do_OpenCL.setChecked(0)
                return
            if self.platform.count() == 0:
                self.platform.addItems([i.name for i in ocl.platforms])

    def platform_changed(self):
        logger.debug("platform_changed")
        if ocl is None:
            self.do_OpenCL.setChecked(0)
            return
        platform = ocl.get_platform(str(self.platform.currentText()))
        for i in range(self.device.count())[-1::-1]:
            self.device.removeItem(i)
        self.device.addItems([i.name for i in platform.devices])

    def get_method(self):
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
            if self.input_data is not None and len(self.input_data) > 5:
                method = "csr"
            else:
                method = "splitbbox"
        return method

    def save_config(self):
        logger.debug("save_config")

        result = qt.QFileDialog.getSaveFileName(
            caption="Save configuration as json",
            directory=self.json_file,
            filter="Config (*.json)")
        if isinstance(result, tuple):
            # PyQt5 compatibility
            result = result[0]

        json_file = result
        if json_file:
            self.dump(json_file)
