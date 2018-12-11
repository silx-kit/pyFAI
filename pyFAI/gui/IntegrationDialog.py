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
__date__ = "11/12/2018"
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
from .widgets.IntegrationFrame import IntegrationFrame
from ..io import HDF5Writer
from ..third_party import six
from .utils import projecturl
from ..utils import get_ui_file


class IntegrationDialog(qt.QWidget):
    """Dialog to configure an azimuthal integration.
    """

    def __init__(self, input_data=None, output_path=None, output_format=None, slow_dim=None, fast_dim=None, json_file=".azimint.json"):
        qt.QWidget.__init__(self)
        filename = get_ui_file("integration-dialog.ui")
        qt.loadUi(filename, self)

        self.__integrationFrame = IntegrationFrame(self._holder)
        layout = qt.QVBoxLayout(self._holder)
        layout.addWidget(self.__integrationFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        self._holder.setLayout(layout)

        self.input_data = input_data
        self.output_path = output_path
        self.output_format = output_format
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim

        self._sem = threading.Semaphore()
        self.json_file = json_file

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
        # self.progressBar.setValue(0)

        if self.json_file is not None:
            self.restore(self.json_file)

    def proceed(self):
        with self._sem:
            out = None
            config = self.dump()
            frame = self.self.__integrationFrame
            logger.debug("Let's work a bit")
            ai = worker.make_ai(config)

            # Default Keyword arguments
            kwarg = {
                "unit": frame.getRadialUnit(),
                "dummy": frame.getDummy(),
                "delta_dummy": frame.getDeltaDummy(),
                "polarization_factor": frame.getPolarizationFactor(),
                "filename": None,
                "safe": False,
                "correctSolidAngle": frame.getCorrectSolidAngle(),
                "error_model": frame.getErrorModel(),
                "method": frame.getMethod(),
                "npt_rad": frame.getRadialNbpt()}

            if kwarg["npt_rad"] is None:
                message = "You must provide the number of output radial bins !"
                qt.QMessageBox.warning(self, "PyFAI integrate", message)
                return {}

            if frame.getIntegrationKind() == "2d":
                kwarg["npt_azim"] = frame.getAzimuthalNbpt()
            rangeValue = frame.getRadialRange()
            if rangeValue is not None:
                kwarg["radial_range"] = rangeValue
            rangeValue = frame.getAzimuthalRange()
            if rangeValue is not None:
                kwarg["azimuth_range"] = rangeValue

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
                    w.nbpt_rad = frame.getRadialNbpt()
                    w.unit = frame.getRadialUnit()
                    w.dummy = frame.getDummy()
                    w.delta_dummy = frame.getDeltaDummy()
                    w.polarization_factor = frame.getPolarizationFactor()
                    # NOTE: previous implementation was using safe=False, the worker use safe=True
                    w.correct_solid_angle = frame.getCorrectSolidAngle()
                    w.error_model = frame.getErrorModel()
                    w.method = frame.getMethod()
                    w.safe = False
                    if frame.getIntegrationKind() == "2d":
                        w.nbpt_azim = frame.getAzimuthalNbpt()
                    else:
                        w.nbpt_azim = 1
                    w.radial_range = frame.getRadialRange()
                    w.azimuth_range = frame.getAzimuthalRange()
                except RuntimeError as e:
                    qt.QMessageBox.warning(self, "PyFAI integrate", e.args[0] + ". Action aboreded.")
                    return {}

                if frame.getIntegrationKind() == "2d":
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
                for i, item in enumerate(self.input_data):
                    self.progressBar.setValue(100.0 * i / len(self.input_data))
                    logger.debug("processing %s", item)
                    if isinstance(item, (six.text_type, six.binary_type)) and op.exists(item):
                        fab_img = fabio.open(item)
                        multiframe = (fab_img.nframes > 1)
                        kwarg["data"] = fab_img.data
                        kwarg["metadata"] = fab_img.header
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
        config = self.__integrationFrame.getConfig()
        return config

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
        self.__integrationFrame.setConfig(dico)

    def set_input_data(self, stack, stack_name=None):
        self.input_data = stack

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
