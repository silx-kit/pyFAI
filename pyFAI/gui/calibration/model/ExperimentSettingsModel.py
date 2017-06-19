# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
__date__ = "29/05/2017"

from .AbstractModel import AbstractModel
from .DetectorModel import DetectorModel
from .CalibrantModel import CalibrantModel
from .DataModel import DataModel


class ExperimentSettingsModel(AbstractModel):

    def __init__(self, parent=None):
        super(ExperimentSettingsModel, self).__init__(parent)
        self.__image = DataModel()
        self.__mask = DataModel()
        self.__dark = DataModel()
        self.__imageFile = DataModel()
        self.__maskFile = DataModel()
        self.__darkFile = DataModel()
        self.__splineFile = DataModel()
        self.__wavelength = DataModel()
        self.__polarizationFactor = DataModel()
        self.__calibrantModel = CalibrantModel()
        self.__detectorModel = DetectorModel()

        self.__image.changed.connect(self.wasChanged)
        self.__mask.changed.connect(self.wasChanged)
        self.__dark.changed.connect(self.wasChanged)
        self.__imageFile.changed.connect(self.wasChanged)
        self.__splineFile.changed.connect(self.wasChanged)
        self.__maskFile.changed.connect(self.wasChanged)
        self.__darkFile.changed.connect(self.wasChanged)
        self.__wavelength.changed.connect(self.wasChanged)
        self.__polarizationFactor.changed.connect(self.wasChanged)
        self.__calibrantModel.changed.connect(self.wasChanged)
        self.__detectorModel.changed.connect(self.wasChanged)

    def isValid(self):
        return True

    def calibrantModel(self):
        return self.__calibrantModel

    def detectorModel(self):
        return self.__detectorModel

    def detector(self):
        detector = self.__detectorModel.detector()
        splineFile = self.__splineFile.value()
        image = self.__image.value()
        if detector is None:
            return None

        # Do not create another instance of the detector
        # While things are not fixed as expected
        # detector = detector.__class__()

        if detector.__class__.HAVE_TAPER:
            if splineFile is not None:
                detector.set_splineFile(splineFile)

        if image is not None:
            detector.guess_binning(image)

        return detector

    def splineFile(self):
        return self.__splineFile

    def image(self):
        return self.__image

    def mask(self):
        return self.__mask

    def dark(self):
        return self.__dark

    def imageFile(self):
        return self.__imageFile

    def maskFile(self):
        return self.__maskFile

    def darkFile(self):
        return self.__darkFile

    def wavelength(self):
        return self.__wavelength

    def polarizationFactor(self):
        return self.__polarizationFactor
