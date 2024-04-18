
__authors__ = ["E. Gutierrez-Fernandez"]
__license__ = "MIT"

import numpy
from .DataModel import DataModel


class DarkedMaskedImageModel(DataModel):
    """Masked Image and dark-current subtracted"""

    def __init__(self, parent=None, image=None, mask=None, dark=None):
        super().__init__(parent=parent)
        self.__image = image
        self.__mask = mask
        self.__dark = dark
        image.changed.connect(self.__invalidateValue)
        mask.changed.connect(self.__invalidateValue)
        dark.changed.connect(self.__invalidateValue)
        self.__value = None

    def __computeImageData(self):
        image = self.__image.value()
        if image is None:
            return None
        mask = self.__mask.value()
        if mask is None:
            return image
        if mask.shape != image.shape:
            return image
        image = image.astype(copy=True, dtype=numpy.float32)
        image[mask != 0] = float("nan")
        dark = self.__dark.value()
        if dark.shape != image.shape:
            return image
        image = image - dark
        return image

    def __invalidateValue(self):
        self.__value = None
        self.wasChanged()

    def value(self):
        if self.__value is None:
            self.__value = self.__computeImageData()
        return self.__value
