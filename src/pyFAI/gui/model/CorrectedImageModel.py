__authors__ = ["E. Gutierrez-Fernandez"]
__license__ = "MIT"

import numpy
from .DataModel import DataModel
from .DarkedMaskedImageModel import DarkedMaskedImageModel

class CorrectedImageModel(DataModel):
    """Image cleaned up by setting masked pixels to NaN"""

    def __init__(
            self,
            parent=None,
            image_model_data: DarkedMaskedImageModel=None,
            image_model_background: DarkedMaskedImageModel=None,
            flat=None,
        ):
        super().__init__(parent=parent)
        self.__image_model_data = image_model_data
        self.__image_model_background = image_model_background
        self.__flat = flat
        flat.changed.connect(self.__invalidateValue)
        self.__value = None

    def __computeImageData(self):
        image = self.__image_model_data.value()
        if image is None:
            return None

        if self.__image_model_background:
            image_background = self.__image_model_background.value()
            if image_background is None:
                return image
            image = image - image_background

        if self.__flat:
            flat = self.__flat.value()
            if flat.shape != image.shape:
                return image
            image = image / flat
        return image

    def __invalidateValue(self):
        self.__value = None
        self.wasChanged()

    def value(self):
        if self.__value is None:
            self.__value = self.__computeImageData()
        return self.__value
