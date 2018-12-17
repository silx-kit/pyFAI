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
__date__ = "17/12/2018"
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

from .. import worker as worker_mdl
from .widgets.WorkerConfigurator import WorkerConfigurator
from ..io import DefaultAiWriter
from ..io import HDF5Writer
from ..third_party import six
from .utils import projecturl
from ..utils import get_ui_file


class IntegrationDialog(qt.QWidget):
    """Dialog to configure an azimuthal integration.
    """

    def __init__(self, input_data=None, output_path=None, json_file=".azimint.json"):
        qt.QWidget.__init__(self)
        filename = get_ui_file("integration-dialog.ui")
        qt.loadUi(filename, self)

        self.__workerConfigurator = WorkerConfigurator(self._holder)
        layout = qt.QVBoxLayout(self._holder)
        layout.addWidget(self.__workerConfigurator)
        layout.setContentsMargins(0, 0, 0, 0)
        self._holder.setLayout(layout)

        self.input_data = input_data
        self.output_path = output_path

        self._sem = threading.Semaphore()
        self.json_file = json_file

        self.batch_processing.clicked.connect(self.__batchProcess)
        self.save_json_button.clicked.connect(self.save_config)
        self.quit_button.clicked.connect(self.die)

        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

        if self.json_file is not None:
            self.restore(self.json_file)

    def __batchProcess(self):
        if self.input_data is None or len(self.input_data) == 0:
            dialog = qt.QFileDialog(directory=os.getcwd())
            dialog.setWindowTitle("Select images to integrate")
            dialog.setFileMode(qt.QFileDialog.ExistingFiles)
            result = dialog.exec_()
            if not result:
                return
            self.input_data = [str(i) for i in dialog.selectedFiles()]
            dialog.close()

        self.progressBar.setVisible(True)
        self.__workerConfigurator.setEnabled(False)

        # Needed to update the display (hide the dialog, display the bar...)
        app = qt.QApplication.instance()
        while app.hasPendingEvents():
            app.processEvents()

        self.proceed()

        qt.QMessageBox.information(self,
                                   "Integration",
                                   "Batch processing completed.")

        self.die()

    def proceed(self):
        with self._sem:
            out = None
            config = self.dump()
            logger.debug("Processing %s", self.input_data)
            start_time = time.time()
            if self.input_data is None or len(self.input_data) == 0:
                logger.warning("No input data to process")
                return

            elif hasattr(self.input_data, "ndim") and self.input_data.ndim == 3:
                # We have a numpy array of dim3
                worker = worker_mdl.Worker()
                worker.set_config(config)
                worker.safe = False

                if worker.do_2D():
                    out = numpy.zeros((self.input_data.shape[0], worker.nbpt_azim, worker.nbpt_rad), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        data = self.input_data[i]
                        out[i] = worker.process(data)
                else:
                    out = numpy.zeros((self.input_data.shape[0], worker.nbpt_rad), dtype=numpy.float32)
                    for i in range(self.input_data.shape[0]):
                        self.progressBar.setValue(100.0 * i / self.input_data.shape[0])
                        data = self.input_data[i]
                        result = worker.process(data)
                        result = result.T[1]
                        out[i] = result

            elif hasattr(self.input_data, "__len__"):
                worker = worker_mdl.Worker()
                worker.set_config(config)
                worker.safe = False
                worker.output = "raw"

                if worker.nbpt_rad is None:
                    message = "You must provide the number of output radial bins !"
                    qt.QMessageBox.warning(self, "PyFAI integrate", message)
                    return {}

                logger.info("Parameters for integration: %s", str(config))

                out = []
                for i, item in enumerate(self.input_data):
                    self.progressBar.setValue(100.0 * i / len(self.input_data))
                    logger.debug("Processing %s", item)

                    numpy_array = False
                    if isinstance(item, (six.text_type, six.binary_type)) and op.exists(item):
                        img = fabio.open(item)
                        multiframe = img.nframes > 1

                        custom_ext = True
                        if self.output_path:
                            if os.path.isdir(self.output_path):
                                outpath = os.path.join(self.output_path, os.path.splitext(os.path.basename(item))[0])
                            else:
                                outpath = os.path.abspath(self.output_path)
                                custom_ext = False
                        else:
                            outpath = os.path.splitext(item)[0]

                        if custom_ext:
                            if multiframe:
                                outpath = outpath + "_pyFAI.h5"
                            else:
                                if worker.do_2D():
                                    outpath = outpath + ".azim"
                                else:
                                    outpath = outpath + ".dat"
                    else:
                        logger.warning("Item is not a file ... guessing it is a numpy array")
                        numpy_array = True
                        multiframe = False

                    if multiframe:
                        writer = HDF5Writer(outpath)
                        writer.init(config)

                        for i in range(img.nframes):
                            fimg = img.getframe(i)
                            data = fimg.data
                            res = worker.process(data=data,
                                                 metadata=fimg.header,
                                                 writer=writer
                                                 )
                        writer.close()
                    else:
                        if numpy_array:
                            data = item
                            writer = None
                            metadata = None
                        else:
                            data = img.data
                            writer = DefaultAiWriter(outpath, worker.ai)
                            metadata = img.header
                        res = worker.process(data,
                                             writer=writer,
                                             metadata=metadata)
                        if writer:
                            writer.close()
                    out.append(res)

            logger.info("Processing Done in %.3fs !", time.time() - start_time)
            self.progressBar.setValue(100)

        # TODO: It should return nothing
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
        config = self.__workerConfigurator.getConfig()
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
        self.__workerConfigurator.setConfig(dico)

    def set_input_data(self, stack):
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
