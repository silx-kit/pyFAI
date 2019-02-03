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
__date__ = "01/02/2019"

import logging

from silx.gui import qt
from silx.utils import html

import pyFAI.detectors


_logger = logging.getLogger(__name__)


class DetectorLabel(qt.QLabel):
    """Readonly line display"""

    _BASE_TEMPLATE = "<html><head/><body>%s</body></html>"

    _MANUFACTURER_TEMPLATE = "<span style=\"vertical-align:sub;\">%s</span>"

    _TOOLTIP_TEMPLATE = ("<html>"
                         "<ul>"
                         "<li><b>Model:</b> {model}</li>"
                         "<li><b>Manufacturer:</b> {manufacturer}</li>"
                         "<li><b>Type:</b> {kind}</li>"
                         "</ul>"
                         "</html>")

    _MODEL_TEMPLATE = "%s"

    def __init__(self, parent=None):
        super(DetectorLabel, self).__init__(parent)
        self.__model = None
        self.__detector = None

    def dragEnterEvent(self, event):
        if self.__model is not None:
            if event.mimeData().hasFormat("text/uri-list"):
                event.acceptProposedAction()

    def dropEvent(self, event):
        mimeData = event.mimeData()
        if not mimeData.hasUrls():
            qt.QMessageBox.critical(self, "Drop cancelled", "A file is expected")
            return

        urls = mimeData.urls()
        if len(urls) > 1:
            qt.QMessageBox.critical(self, "Drop cancelled", "A single file is expected")
            return

        try:
            path = urls[0].toLocalFile()
            detector = pyFAI.detectors.detector_factory(path)
        except IOError as e:
            _logger.error("Error while loading dropped URL %s", e, exc_info=True)
            qt.QMessageBox.critical(self, "Drop cancelled", str(e))
            return
        except Exception as e:
            _logger.error("Error while reading dropped URL %s", e, exc_info=True)
            qt.QMessageBox.critical(self, "Drop cancelled", str(e))
            return

        if self.__model is None:
            _logger.error("No model defined")
            return

        self.__model.setDetector(detector)

    def __getModelName(self, detector):
        if isinstance(detector, pyFAI.detectors.NexusDetector):
            if hasattr(detector, "name"):
                name = detector.name
                if name is not None:
                    return name

        detectorClass = detector.__class__

        modelName = None
        if hasattr(detectorClass, "aliases"):
            if len(detectorClass.aliases) > 0:
                modelName = detectorClass.aliases[0]
        if modelName is None:
            modelName = detectorClass.__name__
        return modelName

    def detector(self):
        if self.__detector is not None:
            return self.__detector
        if self.__model is not None:
            detector = self.__model.detector()
            return detector
        return None

    def __updateDisplay(self):
        detector = self.detector()
        if detector is None:
            self.setText("No detector")
            self.setToolTip("No detector")
            return

        if detector.__class__ is pyFAI.detectors.NexusDetector:
            model = self.__getModelName(detector)
            manufacturer = "Not specified"
            kind = "Nexus definition"
            if detector.filename:
                kind = "%s (%s)" % (kind, detector.filename)

            description = self._MANUFACTURER_TEMPLATE % html.escape("NeXus")
            description += self._MODEL_TEMPLATE % html.escape(model)

            data = {
                "kind": html.escape(kind),
                "manufacturer": html.escape(manufacturer),
                "model": html.escape(model),
            }
            tooltip = self._TOOLTIP_TEMPLATE.format(**data)

        elif detector.__class__ is pyFAI.detectors.Detector:
            description = self._MODEL_TEMPLATE % "Custom detector"
            tooltip = description

        else:
            manufacturer = detector.MANUFACTURER
            if isinstance(manufacturer, list):
                manufacturer = manufacturer[0]
            model = self.__getModelName(detector)
            description = self._MODEL_TEMPLATE % html.escape(model)
            if manufacturer is not None:
                manufacturer = html.escape(manufacturer)
                description = self._MANUFACTURER_TEMPLATE % manufacturer + " " + description
            else:
                manufacturer = "Not specified"

            if detector.__class__.__module__.startswith("pyFAI.detectors."):
                kind = "pyFAI definition"
            else:
                kind = "Custom definition"

            data = {
                "kind": html.escape(kind),
                "manufacturer": html.escape(manufacturer),
                "model": html.escape(model),
            }
            tooltip = self._TOOLTIP_TEMPLATE.format(**data)

        text = self._BASE_TEMPLATE % description
        self.setText(text)
        self.setToolTip(tooltip)

    def setDetectorModel(self, model):
        self.__detector = None
        if self.__model is not None:
            self.__model.changed.disconnect(self.__modelChanged)
        self.__model = model
        if self.__model is not None:
            self.__model.changed.connect(self.__modelChanged)
        self.__modelChanged()

    def __modelChanged(self):
        self.__updateDisplay()

    def detectorModel(self):
        return self.__model

    def setDetector(self, detector):
        self.__model = None
        self.__detector = detector
        self.__updateDisplay()
