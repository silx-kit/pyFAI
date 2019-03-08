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
__date__ = "25/02/2019"
__status__ = "development"

import logging
import json
import os.path as op
import time

logger = logging.getLogger(__name__)

from silx.gui import qt
from silx.gui import icons

from .. import worker as worker_mdl
from .widgets.WorkerConfigurator import WorkerConfigurator
from ..io import integration_config
from .utils import projecturl
from ..utils import get_ui_file
from ..app import integrate
from .. import containers
from pyFAI.gui.utils.eventutils import QtProxifier


class _ThreadSafeIntegrationProcess(QtProxifier):

    def is_interruption_requested(self):
        # NOTE: Not thread safe, but it usually do not call anything but only
        #       returns a boolean
        result = self._target().is_interruption_requested()
        return result

    def request_interruption(self):
        raise RuntimeError("Not supposed to be called")


class IntegrationProcess(qt.QDialog, integrate.IntegrationObserver):

    def __init__(self, parent=None):
        qt.QDialog.__init__(self, parent=parent)
        filename = get_ui_file("integration-process.ui")
        qt.loadUi(filename, self)

        integrate.IntegrationObserver.__init__(self)
        self.setWindowTitle("Processing...")
        self.__undisplayedResult = None
        self._displayResult.toggled.connect(self.__displayResultUpdated)
        self._plot.setDataMargins(0.05, 0.05, 0.05, 0.05)
        self.__firstPlot = True
        self.__lastDisplay = None
        self._progressBar.setFormat("Preprocessing...")
        self._cancelButton.clicked.connect(self.__interruptionRequested)

    def __interruptionRequested(self):
        self.setEnabled(False)
        self.request_interruption()

    def __resultReceived(self, result):
        isFiltered = not self._displayResult.isChecked()
        now = time.time()
        if self.__lastDisplay is not None:
            # Display a new result every 500ms, no more
            if now - self.__lastDisplay < 0.5:
                isFiltered = True
        if not isFiltered:
            self.__undisplayedResult = None
            self.__lastDisplay = now
            self.__displayResult(result, resetZoom=self.__firstPlot)
        else:
            self.__undisplayedResult = result
        self.__firstPlot = False

    def __displayResultUpdated(self):
        self._plot.setVisible(self._displayResult.isChecked())
        if self._displayResult.isChecked():
            if self.__undisplayedResult is not None:
                result = self.__undisplayedResult
                self.__undisplayedResult = None
                self.__displayResult(result, True)
        self.adjustSize()

    def __displayResult(self, result, resetZoom=False):
        self._plot.clear()
        if isinstance(result, containers.Integrate1dResult):
            self._plot.setGraphXLabel("Radial")
            self._plot.setGraphYLabel("Intensity")
            self._plot.addHistogram(
                legend="result1d",
                align="center",
                edges=result.radial,
                color="blue",
                histogram=result.intensity,
                resetzoom=False)
        elif isinstance(result, containers.Integrate2dResult):

            def computeLocation(result):
                # Assume that axes are linear
                if result.intensity.shape[1] > 1:
                    scaleX = (result.radial[-1] - result.radial[0]) / (result.intensity.shape[1] - 1)
                else:
                    scaleX = 1.0
                if result.intensity.shape[0] > 1:
                    scaleY = (result.azimuthal[-1] - result.azimuthal[0]) / (result.intensity.shape[0] - 1)
                else:
                    scaleY = 1.0
                halfPixel = 0.5 * scaleX, 0.5 * scaleY
                origin = (result.radial[0] - halfPixel[0], result.azimuthal[0] - halfPixel[1])
                return origin, (scaleX, scaleY)

            self._plot.setGraphXLabel("Radial")
            self._plot.setGraphYLabel("Azimuthal")
            origin, scale = computeLocation(result)
            self._plot.addImage(
                legend="result2d",
                data=result.intensity,
                origin=origin,
                scale=scale,
                resetzoom=False)
        else:
            logger.error("Unsupported result type %s", type(result))
        if resetZoom:
            self._plot.resetZoom()

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
        self._progressBar.setRange(0, data_count + 1)

    def processing_data(self, data_info, approximate_count=None):
        """
        Start processing the data `data_info`

        :param DataInfo data_info: Contains data and metadata from the data
            to integrate
        :param int approximate_count: If set, the amount of total data to
            process have changed
        """
        if approximate_count is not None:
            self._progressBar.setRange(0, approximate_count + 1)
        self._progressBar.setValue(data_info.data_id)
        if data_info.source_filename is None:
            filename = ""
        elif len(data_info.source_filename) > 20:
            filename = op.basename(data_info.source_filename)
        else:
            filename = data_info.source_filename
        self._progressBar.setFormat("%s (%%p%%)..." % filename)

    def data_result(self, data_info, result):
        self.__resultReceived(result)

    def processing_interrupted(self):
        self.__was_interrupted = True

    def processing_succeeded(self):
        self.__was_interrupted = False

    def processing_finished(self):
        """Called when the full processing is finisehd."""
        self._progressBar.setValue(self._progressBar.maximum())
        self.__lastResult = None
        if self.__was_interrupted:
            self.reject()
        else:
            self.accept()

    def createObserver(self, qtSafe=True):
        """Returns a processing observer connected to this widget.

        :param bool qtSafe: If True the returned observer can be called from
            any thread. Else it have to be called from the main Qt thread.
        :rtype: integrate.IntegrationObserver
        """
        if qtSafe:
            return _ThreadSafeIntegrationProcess(self)
        else:
            self


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
        self.__content = qt.QWidget(self)
        layout = qt.QVBoxLayout(self.__content)
        layout.addWidget(self.__workerConfigurator)

        self._holder.setWidget(self.__content)
        self._holder.minimumSizeHint = self.__minimumScrollbarSizeHint
        size = self.__content.minimumSizeHint()
        self._holder.setMaximumHeight(size.height() + 2)
        size = self.minimumSizeHint() - self._holder.minimumSizeHint() + size
        self.setMaximumHeight(size.height() + 2)

        self.input_data = input_data
        self.output_path = output_path

        self.json_file = json_file

        self.batch_processing.clicked.connect(self.__fireBatchProcess)
        self.save_json_button.clicked.connect(self.save_config)
        self.quit_button.clicked.connect(self.die)

        if self.json_file is not None:
            self.restore(self.json_file)

    def __minimumScrollbarSizeHint(self):
        size = self.__content.minimumSizeHint()
        extend = self.style().pixelMetric(qt.QStyle.PM_ScrollBarExtent)
        size = qt.QSize(size.width() + extend, 100)
        return size

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
