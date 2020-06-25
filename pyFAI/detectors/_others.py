#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2020 European Synchrotron Radiation Facility, Grenoble, France
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

"""
Description of other detectors.
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "25/06/2020"
__status__ = "production"


import numpy
import logging
logger = logging.getLogger(__name__)
import json
from collections import OrderedDict
from ._common import Detector


class Fairchild(Detector):
    """
    Fairchild Condor 486:90 detector
    """
    MANUFACTURER = "Fairchild Semiconductor"

    force_pixel = True
    uniform_pixel = True
    aliases = ["Fairchild", "Condor", "Fairchild Condor 486:90"]
    MAX_SHAPE = (4096, 4096)

    def __init__(self, pixel1=15e-6, pixel2=15e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Titan(Detector):
    """
    Titan CCD detector from Agilent. Mask not handled
    """
    MANUFACTURER = ["Agilent", "Oxford Diffraction"]

    force_pixel = True
    MAX_SHAPE = (2048, 2048)
    aliases = ["Titan 2k x 2k", "Titan 2k x 2k", "OXD Titan", "Agilent Titan"]
    uniform_pixel = True

    def __init__(self, pixel1=60e-6, pixel2=60e-6):
        Detector.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Dexela2923(Detector):
    """
    Dexela CMOS family detector
    """
    force_pixel = True
    aliases = ["Dexela 2923"]
    MAX_SHAPE = (3888, 3072)

    def __init__(self, pixel1=75e-6, pixel2=75e-6):
        super(Dexela2923, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Basler(Detector):
    """
    Basler camera are simple CCD camara over GigaE

    """
    MANUFACTURER = "Basler"

    force_pixel = True
    aliases = ["aca1300"]
    MAX_SHAPE = (966, 1296)

    def __init__(self, pixel=3.75e-6):
        super(Basler, self).__init__(pixel1=pixel, pixel2=pixel)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return {"pixel": self._pixel1}

    def set_config(self, config):
        """Sets the configuration of the detector.

        The configuration is either a python dictionary or a JSON string or a
        file containing this JSON configuration

        keys in that dictionary are:  pixel

        :param config: string or JSON-serialized dict
        :return: self
        """
        if not isinstance(config, dict):
            try:
                config = json.loads(config)
            except Exception as err:  # IGNORE:W0703:
                logger.error("Unable to parse config %s with JSON: %s, %s",
                             config, err)
                raise err
        pixel = config.get("pixel")
        if pixel:
            self.set_pixel1(pixel)
            self.set_pixel2(pixel)
        return self


class Perkin(Detector):
    """
    Perkin detector

    """
    MANUFACTURER = "Perkin Elmer"

    aliases = ["Perkin detector", "Perkin Elmer"]
    force_pixel = True
    MAX_SHAPE = (4096, 4096)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 200e-6

    def __init__(self, pixel1=200e-6, pixel2=200e-6):
        super(Perkin, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(2 * pixel1 / self.DEFAULT_PIXEL1), int(2 * pixel2 / self.DEFAULT_PIXEL2))
            self.shape = tuple(s // b for s, b in zip(self.max_shape, self._binning))
        else:
            self.shape = (2048, 2048)
            self._binning = (2, 2)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Pixium(Detector):
    """PIXIUM 4700 detector

    High energy X ray diffraction using the Pixium 4700 flat panel detector
    J E Daniels, M Drakopoulos, et al.; Journal of Synchrotron Radiation 16(Pt 4):463-8 · August 2009
    """
    MANUFACTURER = "Thales Electronics"

    aliases = ["Pixium 4700", "Pixium 4700 detector", "Thales Electronics"]
    force_pixel = True
    MAX_SHAPE = (1910, 2480)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 154e-6

    def __init__(self, pixel1=308e-6, pixel2=308e-6):
        """Defaults to 2x2 binning
        """
        super(Pixium, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(round(pixel1 / self.DEFAULT_PIXEL1)),
                             int(round(pixel2 / self.DEFAULT_PIXEL2)))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class Apex2(Detector):
    """BrukerApex2 detector

    Actually a derivative from the Fairchild detector with higher binning
    """
    MANUFACTURER = "Bruker"

    aliases = ["ApexII", "Bruker"]
    force_pixel = True
    MAX_SHAPE = (1024, 1024)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 60e-6

    def __init__(self, pixel1=120e-6, pixel2=120e-6):
        """Defaults to 2x2 binning
        """
        super(Apex2, self).__init__(pixel1=pixel1, pixel2=pixel2)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(round(pixel1 / self.DEFAULT_PIXEL1)),
                             int(round(pixel2 / self.DEFAULT_PIXEL2)))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class RaspberryPi5M(Detector):
    """5 Mpix detector from Raspberry Pi

    """
    aliases = ["Picam v1"]
    force_pixel = True
    MAX_SHAPE = (1944, 2592)

    def __init__(self, pixel1=1.4e-6, pixel2=1.4e-6):
        super(RaspberryPi5M, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))


class RaspberryPi8M(Detector):
    """8 Mpix detector from Raspberry Pi

    """
    aliases = ["Picam v2"]
    force_pixel = True
    MAX_SHAPE = (2464, 3280)

    def __init__(self, pixel1=1.12e-6, pixel2=1.12e-6):
        super(RaspberryPi8M, self).__init__(pixel1=pixel1, pixel2=pixel2)

    def get_config(self):
        """Return the configuration with arguments to the constructor

        :return: dict with param for serialization
        """
        return OrderedDict((("pixel1", self._pixel1),
                            ("pixel2", self._pixel2)))
