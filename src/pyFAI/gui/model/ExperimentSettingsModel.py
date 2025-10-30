# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2025 European Synchrotron Radiation Facility
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

__authors__ = ["V. Valls", "Jérôme Kieffer"]
__license__ = "MIT"
__date__ = "30/10/2025"

import logging
from .AbstractModel import AbstractModel
from .DetectorModel import DetectorModel
from .CalibrantModel import CalibrantModel
from .DataModel import DataModel
from .MaskedImageModel import MaskedImageModel
from .ImageModel import ImageFromFilenameModel
from .FilenameModel import FilenameModel
from .PreProcessedImageModel import PreProcessedImageModel

_logger = logging.getLogger(__name__)

class ExperimentSettingsModel(AbstractModel):

    def __init__(self, parent=None):
        super(ExperimentSettingsModel, self).__init__(parent)
        self.__image = ImageFromFilenameModel()
        self.__mask = ImageFromFilenameModel()
        self.__maskedImage = MaskedImageModel(None, self.__image, self.__mask)
        self.__isDetectorMask = True
        self.__dark = ImageFromFilenameModel()
        self.__flat = ImageFromFilenameModel()
        self.__preprocessed_image = PreProcessedImageModel(
            parent=None,
            image=self.__image,
            mask=self.__mask,
            dark=self.__dark,
            flat=self.__flat,
        )

        self.__wavelength = DataModel()
        self.__polarizationFactor = DataModel()
        self.__calibrantModel = CalibrantModel()
        self.__detectorModel = DetectorModel()
        self.__poniFile = FilenameModel()
        self.__jsonFile = FilenameModel()
        self.__parallaxCorrection = DataModel()

        self.__image.changed.connect(self.wasChanged)
        self.__image.filenameChanged.connect(self.wasChanged)
        self.__mask.changed.connect(self.wasChanged)
        self.__mask.filenameChanged.connect(self.wasChanged)
        self.__wavelength.changed.connect(self.wasChanged)
        self.__polarizationFactor.changed.connect(self.wasChanged)
        self.__calibrantModel.changed.connect(self.wasChanged)
        self.__detectorModel.changed.connect(self.wasChanged)
        self.__poniFile.changed.connect(self.wasChanged)
        self.__jsonFile.changed.connect(self.wasChanged)
        self.__parallaxCorrection.changed.connect(self.wasChanged)

        self.__dark.changed.connect(self.wasChanged)
        self.__dark.filenameChanged.connect(self.wasChanged)
        self.__flat.changed.connect(self.wasChanged)
        self.__flat.filenameChanged.connect(self.wasChanged)

        self.__image.changed.connect(self.__updateDetectorMask)
        self.__detectorModel.changed.connect(self.__updateDetectorMask)
        self.__mask.changed.connect(self.__notAnymoreADetectorMask)


    def __repr__(self) -> str:
        res = [f"wavelength: {self.__wavelength.value()}",
               f"polarization: {self.__polarizationFactor.value()}",
               f"__calibrantModel: {self.__calibrant}",
               f"detectorModel: {self.__detectorModel}",
               f"poniFile: {self.__poniFile}",
               f"self.__jsonFile: {self.__jsonFile}",
               f"parallaxCorrection: {self.__parallaxCorrection.value}"]
        return ", ".join(res)

    def __updateDetectorMask(self):
        if self.mask().filename() is not None:
            # It exists a custom mask
            return

        if not self.__isDetectorMask:
            # It was not set by this process
            # Then it was customed by the user
            return

        detector = self.__detectorModel.detector()
        if detector is None:
            mask = None
        else:
            image = self.__image.value()
            if image is not None:
                detector.guess_binning(image)
                try:
                    mask = detector.dynamic_mask(image)
                except ValueError as err:
                    _logger.warning(f"{type(err)}: {err} \nDetector shape: {detector.shape} and image shape: {image.shape}")
                    mask = detector.mask
            else:
                mask = detector.mask
            # Here mask can be None
            # For example if image do not feet the detector

        if mask is not None:
            mask = mask.copy()
        self.__mask.changed.disconnect(self.__notAnymoreADetectorMask)
        self.__mask.setValue(mask)
        self.__isDetectorMask = True
        self.__mask.changed.connect(self.__notAnymoreADetectorMask)

    def __notAnymoreADetectorMask(self):
        self.__isDetectorMask = False

    def isValid(self):
        return True

    def calibrantModel(self):
        return self.__calibrantModel

    def detectorModel(self):
        return self.__detectorModel

    def detector(self):
        """Detector getter synchronizing internal detector configuration to
        match the input image.
        """
        detector = self.__detectorModel.detector()
        image = self.__image.value()
        if detector is None:
            return None

        # Do not create another instance of the detector
        # While things are not fixed as expected
        # detector = detector.__class__()

        # TODO: guess_binning should only be called when image or detector have changed

        if image is not None:
            detector.guess_binning(image)

        return detector

    def image(self):
        return self.__image

    def mask(self):
        return self.__mask

    def maskedImage(self):
        return self.__maskedImage

    def wavelength(self):
        return self.__wavelength

    def polarizationFactor(self):
        return self.__polarizationFactor

    def poniFile(self):
        return self.__poniFile

    def jsonFile(self):
        return self.__jsonFile

    def dark(self):
        return self.__dark

    def flat(self):
        return self.__flat

    def preprocessedImage(self):
        return self.__preprocessed_image

    def parallaxCorrection(self):
        return self.__parallaxCorrection