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
__date__ = "06/12/2018"

from silx.gui import qt
from silx.utils import html

import pyFAI.detectors


class DetectorLabel(qt.QLabel):
    """Readonly line display"""

    _BASE_TEMPLATE = "<html><head/><body>%s</body></html>"

    _MANUFACTURER_TEMPLATE = "<span style=\"vertical-align:sub;\">%s</span>"

    _MODEL_TEMPLATE = "%s"

    def __init__(self, parent=None):
        super(DetectorLabel, self).__init__(parent)
        self.__model = None
        self.__detector = None

    def __getModelName(self, detectorClass):
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
            className = detector.__class__.__name__
            name = self.__getModelName(detector.__class__)

            if className == name:
                description = self._MODEL_TEMPLATE % html.escape(className)
            else:
                description = self._MANUFACTURER_TEMPLATE % html.escape(className)
                description += self._MODEL_TEMPLATE % html.escape(name)

        elif detector.__class__ is pyFAI.detectors.Detector:
            description = self._MODEL_TEMPLATE % "Custom detector"

        else:
            manufacturer = detector.MANUFACTURER
            if isinstance(manufacturer, list):
                manufacturer = manufacturer[0]
            model = self.__getModelName(detector.__class__)
            description = self._MODEL_TEMPLATE % html.escape(model)
            if manufacturer is not None:
                description = self._MANUFACTURER_TEMPLATE % html.escape(manufacturer) + " " + description

        text = self._BASE_TEMPLATE % description
        self.setText(text)
        self.setToolTip(description)

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
