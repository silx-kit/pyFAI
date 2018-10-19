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
__date__ = "18/10/2018"
__status__ = "development"

import logging
import json
import os
import time
import threading
import math
import os.path as op
import numpy
logger = logging.getLogger(__name__)
from silx.gui import qt
import fabio
from .. import _version
from .. import worker
from ..detectors import ALL_DETECTORS, detector_factory
from ..opencl import ocl
from ..utils import float_, int_, str_, get_ui_file
from ..io import HDF5Writer
from ..azimuthalIntegrator import AzimuthalIntegrator
from ..units import RADIAL_UNITS
from ..third_party import six
from .utils import projecturl


UIC = get_ui_file("integration.ui")
LOCAL_DOC = "/usr/share/doc/pyfai-doc/html/man/pyFAI-integrate.html"
if _version.version_info.releaselevel == "final":
    version_tuple = ".".join(str(i) for i in _version.version_info[:3])
    REMOTE_DOC = "http://silx.org/doc/pyFAI/" + version_tuple + "/man/pyFAI-integrate.html"
else:
    REMOTE_DOC = "http://silx.org/doc/pyFAI/dev/man/pyFAI-integrate.html"


class AIWidget(qt.QWidget):
    """
    """
    def __init__(self, input_data=None, output_path=None, output_format=None, slow_dim=None, fast_dim=None, json_file=".azimint.json"):
        self.units = {}
        self.input_data = input_data
        self.output_path = output_path
        self.output_format = output_format
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        self.name = None
        self._sem = threading.Semaphore()
        self.json_file = json_file
        qt.QWidget.__init__(self)
        try:
            qt.loadUi(UIC, self)
        except AttributeError as _error:
            logger.error("I looks like your installation suffers from this bug: http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=697348")
            raise RuntimeError("Please upgrade your installation of PyQt (or apply the patch)")
        self.all_detectors = list(ALL_DETECTORS.keys())
        self.all_detectors.sort()
        self.detector.addItems([i.capitalize() for i in self.all_detectors])
        self.detector.setCurrentIndex(self.all_detectors.index("detector"))
        # connect file selection windows
        self.file_poni.clicked.connect(self.select_ponifile)
        self.file_splinefile.clicked.connect(self.select_splinefile)
        self.file_mask_file.clicked.connect(self.select_maskfile)
        self.file_dark_current.clicked.connect(self.select_darkcurrent)
        self.file_flat_field.clicked.connect(self.select_flatfield)
        # connect button bar
        self.okButton = self.buttonBox.button(qt.QDialogButtonBox.Ok)
        self.saveButton = self.buttonBox.button(qt.QDialogButtonBox.Save)
        self.resetButton = self.buttonBox.button(qt.QDialogButtonBox.Reset)
        self.cancelButton = self.buttonBox.button(qt.QDialogButtonBox.Cancel)
        self.okButton.clicked.connect(self.proceed)
        self.saveButton.clicked.connect(self.save_config)
        self.buttonBox.helpRequested.connect(self.help)
        self.cancelButton.clicked.connect(self.die)
        self.resetButton.clicked.connect(self.restore)

        self.detector.currentIndexChanged.connect(self.detector_changed)
        self.do_OpenCL.clicked.connect(self.openCL_changed)
        self.platform.currentIndexChanged.connect(self.platform_changed)
        self.set_validators()
        self.assign_unit()
        if self.json_file is not None:
            self.restore(self.json_file)
        self.progressBar.setValue(0)
        self.hdf5_path = None

    def assign_unit(self):
        """
        assign unit to the corresponding widget
        """
        self.units = {}
        for unit in RADIAL_UNITS:
            if unit == "2th_deg":
                self.units[unit] = self.tth_deg
            elif unit == "2th_rad":
                self.units[unit] = self.tth_rad
            elif unit == "q_nm^-1":
                self.units[unit] = self.q_nm
            elif unit == "q_A^-1":
                self.units[unit] = self.q_A
            elif unit == "r_mm":
                self.units[unit] = self.r_mm
            else:
                logger.debug("Unit unknown to GUI %s", unit)

    def set_validators(self):
        """
        Set all validators for text entries
        """
        npt_validator = qt.QIntValidator()
        npt_validator.setBottom(1)
        self.nbpt_rad.setValidator(npt_validator)
        self.nbpt_azim.setValidator(npt_validator)

        wl_validator = qt.QDoubleValidator(self)
        wl_validator.setBottom(1e-15)
        wl_validator.setTop(1e-6)
        self.wavelength.setValidator(wl_validator)

        distance_validator = qt.QDoubleValidator(self)
        distance_validator.setBottom(0)
        self.pixel1.setValidator(distance_validator)
        self.pixel2.setValidator(distance_validator)
        self.poni1.setValidator(distance_validator)
        self.poni2.setValidator(distance_validator)

        angle_validator = qt.QDoubleValidator(self)
        distance_validator.setBottom(-math.pi)
        distance_validator.setTop(math.pi)
        self.rot1.setValidator(angle_validator)
        self.rot2.setValidator(angle_validator)
        self.rot3.setValidator(angle_validator)
        # done at widget level
        # self.polarization_factor.setValidator(qt.QDoubleValidator(-1, 1, 3))

    def __get_unit(self):
        for unit, widget in self.units.items():
            if widget is not None and widget.isChecked():
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
        to_save = {"poni": str_(self.poni.text()).strip(),
                   "detector": str_(self.detector.currentText()).lower(),
                   "wavelength": self._float("wavelength", None),
                   "splineFile": str_(self.splineFile.text()).strip(),
                   "pixel1": self._float("pixel1", None),
                   "pixel2": self._float("pixel2", None),
                   "dist": self._float("dist", None),
                   "poni1": self._float("poni1", None),
                   "poni2": self._float("poni2", None),
                   "rot1": self._float("rot1", None),
                   "rot2": self._float("rot2", None),
                   "rot3": self._float("rot3", None),
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
                   "do_OpenCL": bool(self.do_OpenCL.isChecked())
                   }
        for unit, widget in self.units.items():
            if widget is not None and widget.isChecked():
                to_save["unit"] = str(unit)
                break
        else:
            logger.info("Undefined unit !!!")
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
        setup_data = {"poni": self.poni.setText,
                      # "detector": self.all_detectors[self.detector.getCurrentIndex()],
                      "wavelength": lambda a: self.wavelength.setText(str_(a)),
                      "splineFile": lambda a: self.splineFile.setText(str_(a)),
                      "pixel1": lambda a: self.pixel1.setText(str_(a)),
                      "pixel2": lambda a: self.pixel2.setText(str_(a)),
                      "dist": lambda a: self.dist.setText(str_(a)),
                      "poni1": lambda a: self.poni1.setText(str_(a)),
                      "poni2": lambda a: self.poni2.setText(str_(a)),
                      "rot1": lambda a: self.rot1.setText(str_(a)),
                      "rot2": lambda a: self.rot2.setText(str_(a)),
                      "rot3": lambda a: self.rot3.setText(str_(a)),
                      "do_dummy": self.do_dummy.setChecked,
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
                value(dico[key])
        if "unit" in dico:
            for unit, widget in self.units.items():
                if str(unit) == dico["unit"] and widget is not None:
                    widget.setChecked(True)
                    break
        if "detector" in dico:
            detector = dico["detector"].lower()
            if detector in self.all_detectors:
                self.detector.setCurrentIndex(self.all_detectors.index(detector))
        if setup_data.get("do_OpenCL"):
            self.openCL_changed()

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

    def select_ponifile(self):
        ponifile = self.getOpenFileName("Open a poni file")
        if ponifile is not None:
            self.set_ponifile(ponifile)

    def select_splinefile(self):
        logger.debug("select_splinefile")
        splinefile = self.getOpenFileName("Open a spline file")
        if splinefile:
            try:
                ai = AzimuthalIntegrator()
                ai.detector.set_splineFile(splinefile)
                self.pixel1.setText(str(ai.pixel1))
                self.pixel2.setText(str(ai.pixel2))
                self.splineFile.setText(ai.detector.splineFile or "")
            except Exception as error:
                logger.error("failed %s on %s", error, splinefile)

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

    def set_ponifile(self, ponifile=None):
        if ponifile is None:
            ponifile = str_(self.poni.text())
        else:
            if self.poni.text() != ponifile:
                self.poni.setText(ponifile)
#         try:
#             str(ponifile)
#         except UnicodeError:
#             ponifile = ponifile.encode("utf8")
#         print(ponifile, type(ponifile))
        try:
            ai = AzimuthalIntegrator.sload(ponifile)
        except Exception as error:
            ai = AzimuthalIntegrator()
            logger.error("file %s does not look like a poni-file, error %s", ponifile, error)
            return
        self.pixel1.setText(str_(ai.pixel1))
        self.pixel2.setText(str_(ai.pixel2))
        self.dist.setText(str_(ai.dist))
        self.poni1.setText(str_(ai.poni1))
        self.poni2.setText(str_(ai.poni2))
        self.rot1.setText(str_(ai.rot1))
        self.rot2.setText(str_(ai.rot2))
        self.rot3.setText(str_(ai.rot3))
        self.splineFile.setText(str_(ai.detector.splineFile))
        self.wavelength.setText(str_(ai._wavelength))

        name = ai.detector.name.lower().replace(" ", "_")
        if name not in self.all_detectors:
            name = ai.detector.name.lower().replace(" ", "")
        if name in self.all_detectors:
            self.detector.setCurrentIndex(self.all_detectors.index(name))
        else:
            self.detector.setCurrentIndex(self.all_detectors.index("detector"))

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

    def detector_changed(self):
        logger.debug("detector_changed")
        detector = str_(self.detector.currentText()).lower()
        inst = detector_factory(detector)
        if inst.force_pixel:
            self.pixel1.setText(str(inst.pixel1))
            self.pixel2.setText(str(inst.pixel2))
            self.splineFile.setText("")
        elif self.splineFile.text():
            splineFile = str_(self.splineFile.text()).strip()
            if op.isfile(splineFile):
                inst.set_splineFile(splineFile)
                self.pixel1.setText(str(inst.pixel1))
                self.pixel2.setText(str(inst.pixel2))
            else:
                logger.warning("No such spline file %s", splineFile)

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
