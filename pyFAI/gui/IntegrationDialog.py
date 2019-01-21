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
__date__ = "21/01/2019"
__status__ = "development"

import logging
import json
import os.path as op

logger = logging.getLogger(__name__)

from silx.gui import qt
from silx.gui import icons

from .. import worker as worker_mdl
from .widgets.WorkerConfigurator import WorkerConfigurator
from ..io import integration_config
from .utils import projecturl
from ..utils import get_ui_file
from ..app import integrate


class IntegrationProcess(qt.QDialog, integrate.IntegrationObserver):

    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent=parent)
        integrate.IntegrationObserver.__init__(self)
        self.setWindowTitle("Processing...")

        self.__button = qt.QPushButton(self)
        self.__button.setText("Cancel")
        self.__progressBar = qt.QProgressBar(self)

        layout = qt.QHBoxLayout(self)
        layout.addWidget(self.__progressBar)
        layout.addWidget(self.__button)

    def worker_initialized(self, worker):
        """
        Called when the worker is initialized

        :param int data_count: Number of data to integrate
        """
        pass

    def processing_started(self, data_count):
        """
        Called before starting the full processing.

        :param int data_count: Number of data to integrate
        """
        self.__progressBar.setRange(0, data_count + 1)

    def processing_data(self, data_id, filename):
        """
        Start processing the data `data_id`

        :param int data_id: Id of the data
        :param str filename: Filename of the data, if any.
        """
        self.__progressBar.setValue(data_id)

    def processing_finished(self):
        """Called when the full processing is finisehd."""
        self.__progressBar.setValue(self.__progressBar.maximum())
        self.accept()


class IntegrationDialog(qt.QWidget):
    """Dialog to configure an azimuthal integration.
    """

    batchProcessRequested = qt.Signal()

    def __init__(self, input_data=None, output_path=None, json_file=".azimint.json", context=None):
        qt.QWidget.__init__(self)
        filename = get_ui_file("integration-dialog.ui")
        qt.loadUi(filename, self)

        pyfaiIcon = icons.getQIcon("pyfai:gui/images/icon")
        self.setWindowIcon(pyfaiIcon)

        self.__context = context
        if context is not None:
            context.restoreWindowLocationSettings("main-window", self)

        self.__workerConfigurator = WorkerConfigurator(self._holder)
        layout = qt.QVBoxLayout(self._holder)
        layout.addWidget(self.__workerConfigurator)
        layout.setContentsMargins(0, 0, 0, 0)
        self._holder.setLayout(layout)

        self.input_data = input_data
        self.output_path = output_path

        self.json_file = json_file

        self.batch_processing.clicked.connect(self.__fireBatchProcess)
        self.save_json_button.clicked.connect(self.save_config)
        self.quit_button.clicked.connect(self.die)

        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

        if self.json_file is not None:
            self.restore(self.json_file)

    def closeEvent(self, event):
        context = self.__context
        if context is not None:
            self.__context.saveWindowLocationSettings("main-window", self)

    def __fireBatchProcess(self):
        self.batchProcessRequested.emit()

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
        dico = integration_config.normalize(dico)
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
