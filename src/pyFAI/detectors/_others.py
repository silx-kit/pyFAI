#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2025 European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "04/11/2025"
__status__ = "production"

import logging
from ._common import Detector
logger = logging.getLogger(__name__)


class Fairchild(Detector):
    """
    Fairchild Condor 486:90 detector
    """
    MANUFACTURER = "Fairchild Semiconductor"

    force_pixel = True
    PIXEL_SIZE = (15e-6, 15e-6)
    uniform_pixel = True
    aliases = ["Fairchild", "Condor", "Fairchild Condor 486:90"]
    MAX_SHAPE = (4096, 4096)


class Titan(Detector):
    """
    Titan CCD detector from Agilent. Mask not handled
    """
    MANUFACTURER = ["Agilent", "Oxford Diffraction"]

    force_pixel = True
    PIXEL_SIZE = (60e-6, 60e-6)
    MAX_SHAPE = (2048, 2048)
    aliases = ["Titan 2k x 2k", "Titan 2k x 2k", "OXD Titan", "Agilent Titan"]
    uniform_pixel = True


class Dexela2923(Detector):
    """
    Dexela CMOS family detector
    """
    force_pixel = True
    PIXEL_SIZE = (75e-6, 75e-6)
    aliases = ["Dexela 2923"]
    MAX_SHAPE = (3888, 3072)


class Basler(Detector):
    """
    Basler camera are simple CCD camara over GigaE

    """
    MANUFACTURER = "Basler"
    force_pixel = True
    PIXEL_SIZE = (3.75e-6, 3.75e-6)
    aliases = ["aca1300"]
    MAX_SHAPE = (966, 1296)


class Perkin(Detector):
    """
    Perkin detector

    """
    MANUFACTURER = "Perkin Elmer"

    aliases = ["Perkin detector", "Perkin Elmer"]
    force_pixel = True
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 200e-6
    PIXEL_SIZE = (DEFAULT_PIXEL1, DEFAULT_PIXEL2)
    MAX_SHAPE = (4096, 4096)


    def __init__(self, pixel1=200e-6, pixel2=200e-6, max_shape=None, orientation=0):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (pixel1 != self.PIXEL_SIZE[0]) or (pixel2 != self.PIXEL_SIZE[1]):
            self._binning = (int(2 * pixel1 / self.DEFAULT_PIXEL1), int(2 * pixel2 / self.DEFAULT_PIXEL2))
            self.shape = tuple(s // b for s, b in zip(self.max_shape, self._binning))
        else:
            self.shape = (2048, 2048)
            self._binning = (2, 2)


class Pixium(Detector):
    """PIXIUM 4700 detector

    High energy X ray diffraction using the Pixium 4700 flat panel detector
    J E Daniels, M Drakopoulos, et al.; Journal of Synchrotron Radiation 16(Pt 4):463-8 · August 2009
    """
    MANUFACTURER = "Thales Electronics"

    aliases = ["Pixium 4700", "Pixium 4700 detector", "Thales Electronics"]
    force_pixel = True
    PIXEL_SIZE = (308e-6, 308e-6)
    MAX_SHAPE = (1910, 2480)
    DEFAULT_PIXEL1 = DEFAULT_PIXEL2 = 154e-6

    def __init__(self, pixel1=308e-6, pixel2=308e-6, max_shape=None, orientation=0):
        """Defaults to 2x2 binning
        """
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (pixel1 != self.DEFAULT_PIXEL1) or (pixel2 != self.DEFAULT_PIXEL2):
            self._binning = (int(round(pixel1 / self.DEFAULT_PIXEL1)),
                             int(round(pixel2 / self.DEFAULT_PIXEL2)))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))


class Apex2(Detector):
    """BrukerApex2 detector

    Actually a derivative from the Fairchild detector with higher binning
    """
    MANUFACTURER = "Bruker"
    aliases = ["ApexII", "Bruker"]
    force_pixel = True
    PIXEL_SIZE = (120e-6, 120e-6)
    MAX_SHAPE = (1024, 1024)

    def __init__(self, pixel1=120e-6, pixel2=120e-6, max_shape=None, orientation=0):
        """Defaults to 2x2 binning
        """
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=max_shape, orientation=orientation)
        if (pixel1 != self.PIXEL_SIZE[0]) or (pixel2 != self.PIXEL_SIZE[1]):
            self._binning = (int(round(pixel1 / self.PIXEL_SIZE[0])),
                             int(round(pixel2 / self.PIXEL_SIZE[1])))
            self.shape = tuple(s // b for s, b in zip(self.MAX_SHAPE, self._binning))


class RaspberryPi5M(Detector):
    """5 Mpix detector from Raspberry Pi
    """
    aliases = ["Picam v1"]
    force_pixel = True
    PIXEL_SIZE = (1.4e-6, 1.4e-6)
    MAX_SHAPE = (1944, 2592)


class RaspberryPi8M(Detector):
    """8 Mpix detector from Raspberry Pi
    """
    aliases = ["Picam v2"]
    force_pixel = True
    PIXEL_SIZE = (1.12e-6, 1.12e-6)
    MAX_SHAPE = (2464, 3280)


class RaspberryPi12M(Detector):
    """8 Mpix detector from Raspberry Pi
    """
    aliases = ["Picam HQ"]
    force_pixel = True
    PIXEL_SIZE = (1.55e-6, 1.55e-6)
    MAX_SHAPE = (3040, 4056)
