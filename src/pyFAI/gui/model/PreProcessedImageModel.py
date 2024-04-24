#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2012-2024 European Synchrotron Radiation Facility, Grenoble, France
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


__authors__ = ["E. Gutierrez-Fernandez", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "23/04/2024"

from .DataModel import DataModel
import logging
logger = logging.getLogger(__name__)

try:
    from pyFAI.ext.preproc import preproc
except ImportError as err:
    logger.warning("ImportError pyFAI.ext.preproc %s", err)
    from pyFAI.engines.preproc import preproc

class PreProcessedImageModel(DataModel):
    """Preprocessed image: masked pixels to NaN, dark-current subtracted and normalized to flat-field"""

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
            if mask is not None and mask.shape != image.shape:
                mask = None
        else:
            mask = None

        if self.__dark is not None:
            dark = self.__dark.value()
            if dark is not None and dark.shape != image.shape:
                dark = None
        else:
            dark = None

        if self.__flat is not None:
            flat = self.__flat.value()
            if flat is not None and flat.shape != image.shape:
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
