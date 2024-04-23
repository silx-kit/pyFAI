__authors__ = ["E. Gutierrez-Fernandez"]
__license__ = "MIT"

from .DataModel import DataModel
import logging
logger = logging.getLogger(__name__)

try:
    from pyFAI.ext.preproc import preproc
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    from pyFAI.engines.preproc import preproc

class PreProcessedImageModel(DataModel):
    """Image cleaned up by setting masked pixels to NaN"""

    def __init__(
            self, 
            parent=None,
            image=None,
            mask=None,
            dark=None,
            flat=None,
        ):
        super().__init__(parent=parent)
        self.__image = image
        self.__mask = mask
        self.__dark = dark
        self.__flat = flat
        image.changed.connect(self.__invalidateValue)
        if mask is not None:
            mask.changed.connect(self.__invalidateValue)
        if dark is not None:
            dark.changed.connect(self.__invalidateValue)
        if flat is not None:
            flat.changed.connect(self.__invalidateValue)
        self.__value = None

    def __computeImageData(self):
        image = self.__image.value()
        if image is None:
            return None
        
        if self.__mask is not None:
            mask = self.__mask.value()
            if mask.shape != image.shape:
                mask = None
        else:
            mask = None
        
        if self.__dark is not None:
            dark = self.__dark.value()
            if dark.shape != image.shape:
                dark = None
        else:
            dark = None

        if self.__flat is not None:
            flat = self.__flat.value()
            if flat.shape != image.shape:
                flat = None
        else:
            flat = None

        return preproc(
            raw=image,
            dark=dark,
            flat=flat,
            mask=mask,
        )

    def __invalidateValue(self):
        self.__value = None
        self.wasChanged()

    def value(self):
        if self.__value is None:
            self.__value = self.__computeImageData()
        return self.__value
