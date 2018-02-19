#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2017-2018 European Synchrotron Radiation Facility, Grenoble, France
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
Description of the `Rayonix <https://www.rayonix.com/>`_ detectors.
"""

from __future__ import print_function, division, absolute_import, with_statement

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "19/02/2018"
__status__ = "production"


import numpy
from ._common import Detector

import logging
logger = logging.getLogger(__name__)


class Rayonix(Detector):
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 32e-6}
    MAX_SHAPE = (4096, 4096)

    def __init__(self, pixel1=32e-6, pixel2=32e-6):
        super(Rayonix, self).__init__(pixel1=pixel1, pixel2=pixel2)
        binning = [1, 1]
        for b, p in self.BINNED_PIXEL_SIZE.items():
            if p == pixel1:
                binning[0] = b
            if p == pixel2:
                binning[1] = b
        self._binning = tuple(binning)
        self.shape = tuple(s // b for s, b in zip(self.max_shape, binning))

    def get_binning(self):
        return self._binning

    def set_binning(self, bin_size=(1, 1)):
        """
        Set the "binning" of the detector,

        :param bin_size: set the binning of the detector
        :type bin_size: int or (int, int)
        """
        if "__len__" in dir(bin_size) and len(bin_size) >= 2:
            bin_size = int(round(float(bin_size[0]))), int(round(float(bin_size[1])))
        else:
            b = int(round(float(bin_size)))
            bin_size = (b, b)
        if bin_size != self._binning:
            if (bin_size[0] in self.BINNED_PIXEL_SIZE) and (bin_size[1] in self.BINNED_PIXEL_SIZE):
                self._pixel1 = self.BINNED_PIXEL_SIZE[bin_size[0]]
                self._pixel2 = self.BINNED_PIXEL_SIZE[bin_size[1]]
            else:
                logger.warning("Binning factor (%sx%s) is not an official value for Rayonix detectors", bin_size[0], bin_size[1])
                self._pixel1 = self.BINNED_PIXEL_SIZE[1] / float(bin_size[0])
                self._pixel2 = self.BINNED_PIXEL_SIZE[1] / float(bin_size[1])
            self._binning = bin_size
            self.shape = (self.max_shape[0] // bin_size[0],
                          self.max_shape[1] // bin_size[1])
    binning = property(get_binning, set_binning)

    def __repr__(self):
        return "Detector %s\t PixelSize= %.3e, %.3e m" % \
            (self.name, self._pixel1, self._pixel2)

    def guess_binning(self, data):
        """
        Guess the binning/mode depending on the image shape
        :param data: 2-tuple with the shape of the image or the image with a .shape attribute.
        :return: True if the data fit the detector
        :rtype: bool
        """
        if "shape" in dir(data):
            shape = data.shape
        elif "__len__" in dir(data):
            shape = tuple(data[:2])
        else:
            logger.warning("No shape available to guess the binning: %s", data)
            self._binning = 1, 1
            self._pixel1 = self.BINNED_PIXEL_SIZE[1]
            self._pixel2 = self.BINNED_PIXEL_SIZE[1]
            return False

        bin1 = self.max_shape[0] // shape[0]
        bin2 = self.max_shape[1] // shape[1]
        self._binning = (bin1, bin2)
        self.shape = shape
        if bin1 not in self.BINNED_PIXEL_SIZE or bin2 not in self.BINNED_PIXEL_SIZE:
            self._binning = 1, 1
            self._pixel1 = self.BINNED_PIXEL_SIZE[1]
            self._pixel2 = self.BINNED_PIXEL_SIZE[1]
            result = False
        else:
            self._pixel1 = self.BINNED_PIXEL_SIZE[bin1]
            self._pixel2 = self.BINNED_PIXEL_SIZE[bin2]
            result = True
        self._mask = False
        self._mask_crc = None
        return result


class Rayonix133(Rayonix):
    """
    Rayonix 133 2D CCD detector detector also known as mar133

    Personnal communication from M. Blum

    What should be the default binning factor for those cameras ?

    Circular detector
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 32e-6,
                         2: 64e-6,
                         4: 128e-6,
                         8: 256e-6,
                         }
    MAX_SHAPE = (4096, 4096)
    aliases = ["MAR133"]

    def __init__(self, pixel1=64e-6, pixel2=64e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        """Circular mask"""
        c = [i // 2 for i in self.shape]
        x, y = numpy.ogrid[:self.shape[0], :self.shape[1]]
        mask = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2) > (c[0] ** 2)
        return mask.astype(numpy.int8)


class RayonixSx165(Rayonix):
    """
    Rayonix sx165 2d Detector also known as MAR165.

    Circular detector
    """
    BINNED_PIXEL_SIZE = {1: 39.5e-6,
                         2: 79e-6,
                         3: 118.616e-6,  # image shape is then 1364 not 1365 !
                         4: 158e-6,
                         8: 316e-6,
                         }
    MAX_SHAPE = (4096, 4096)
    aliases = ["MAR165", "Rayonix Sx165"]
    force_pixel = True

    def __init__(self, pixel1=39.5e-6, pixel2=39.5e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)

    def calc_mask(self):
        """Circular mask"""
        c = [i // 2 for i in self.shape]
        x, y = numpy.ogrid[:self.shape[0], :self.shape[1]]
        mask = ((x + 0.5 - c[0]) ** 2 + (y + 0.5 - c[1]) ** 2) > (c[0] ** 2)
        return mask.astype(numpy.int8)


class RayonixSx200(Rayonix):
    """
    Rayonix sx200 2d CCD Detector.

    Pixel size are personnal communication from M. Blum.
    """
    BINNED_PIXEL_SIZE = {1: 48e-6,
                         2: 96e-6,
                         3: 144e-6,
                         4: 192e-6,
                         8: 384e-6,
                         }
    MAX_SHAPE = (4096, 4096)
    aliases = ["Rayonix sx200"]

    def __init__(self, pixel1=48e-6, pixel2=48e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixLx170(Rayonix):
    """
    Rayonix lx170 2d CCD Detector (2x1 CCDs).

    Nota: this is the same for lx170hs
    """
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (1920, 3840)
    force_pixel = True
    aliases = ["Rayonix LX170", "Rayonix LX170-HS", "Rayonix LX170 HS", "RayonixLX170HS"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx170(Rayonix):
    """
    Rayonix mx170 2d CCD Detector (2x2 CCDs).

    Nota: this is the same for mx170hs
    """
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (3840, 3840)
    aliases = ["Rayonix MX170", "Rayonix MX170-HS", "RayonixMX170HS", "Rayonix MX170 HS"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixLx255(Rayonix):
    """
    Rayonix lx255 2d Detector (3x1 CCDs)

    Nota: this detector is also called lx255hs
    """
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (1920, 5760)
    aliases = ["Rayonix LX255", "Rayonix LX255-HS", "Rayonix LX 255HS", "RayonixLX225HS"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx225(Rayonix):
    """
    Rayonix mx225 2D CCD detector detector

    Nota: this is the same definition for mx225he
    Personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 36.621e-6,
                         2: 73.242e-6,
                         3: 109.971e-6,
                         4: 146.484e-6,
                         8: 292.969e-6
                         }
    MAX_SHAPE = (6144, 6144)
    aliases = ["Rayonix MX225"]

    def __init__(self, pixel1=73.242e-6, pixel2=73.242e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx225hs(Rayonix):
    """
    Rayonix mx225hs 2D CCD detector detector

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 39.0625e-6,
                         2: 78.125e-6,
                         3: 117.1875e-6,
                         4: 156.25e-6,
                         5: 195.3125e-6,
                         6: 234.3750e-6,
                         8: 312.5e-6,
                         10: 390.625e-6,
                         }
    MAX_SHAPE = (5760, 5760)
    aliases = ["Rayonix MX225HS", "Rayonix MX225 HS"]

    def __init__(self, pixel1=78.125e-6, pixel2=78.125e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx300(Rayonix):
    """
    Rayonix mx300 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 36.621e-6,
                         2: 73.242e-6,
                         3: 109.971e-6,
                         4: 146.484e-6,
                         8: 292.969e-6
                         }
    MAX_SHAPE = (8192, 8192)
    aliases = ["Rayonix mx300"]

    def __init__(self, pixel1=73.242e-6, pixel2=73.242e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx300hs(Rayonix):
    """
    Rayonix mx300hs 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 39.0625e-6,
                         2: 78.125e-6,
                         3: 117.1875e-6,
                         4: 156.25e-6,
                         5: 195.3125e-6,
                         6: 234.3750e-6,
                         8: 312.5e-6,
                         10: 390.625e-6
                         }
    MAX_SHAPE = (7680, 7680)
    aliases = ["Rayonix MX300HS", "Rayonix MX300 HS"]

    def __init__(self, pixel1=78.125e-6, pixel2=78.125e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx340hs(Rayonix):
    """
    Rayonix mx340hs 2D detector (4x4 CCDs)

    Pixel size from a personnal communication from M. Blum
    """
    force_pixel = True
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (7680, 7680)
    aliases = ["Rayonix MX340HS", "Rayonix MX340HS"]

    def __init__(self, pixel1=88.5417e-6, pixel2=88.5417e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixSx30hs(Rayonix):
    """
    Rayonix sx30hs 2D CCD camera (1 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1: 15.625e-6,
                         2: 31.25e-6,
                         3: 46.875e-6,
                         4: 62.5e-6,
                         5: 78.125e-6,
                         6: 93.75e-6,
                         8: 125.0e-6,
                         10: 156.25e-6
                         }
    MAX_SHAPE = (1920, 1920)
    aliases = ["Rayonix SX30HS", "Rayonix SX30 HS"]

    def __init__(self, pixel1=15.625e-6, pixel2=15.625e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixSx85hs(Rayonix):
    """
    Rayonix sx85hs 2D CCD camera (1 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (1920, 1920)
    aliases = ["Rayonix SX85HS", "Rayonix SX85 HS"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx425hs(Rayonix):
    """
    Rayonix mx425hs 2D CCD camera (5x5 CCD chip)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1: 44.2708e-6,
                         2: 88.5417e-6,
                         3: 132.8125e-6,
                         4: 177.0833e-6,
                         5: 221.3542e-6,
                         6: 265.625e-6,
                         8: 354.1667e-6,
                         10: 442.7083e-6
                         }
    MAX_SHAPE = (9600, 9600)
    aliases = ["Rayonix MX425HS", "Rayonix MX425 HS"]

    def __init__(self, pixel1=44.2708e-6, pixel2=44.2708e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)


class RayonixMx325(Rayonix):
    """
    Rayonix mx325 and mx325he 2D detector (4x4 CCD chips)

    Pixel size from a personnal communication from M. Blum
    """
    BINNED_PIXEL_SIZE = {1: 39.673e-6,
                         2: 79.346e-6,
                         3: 119.135e-6,
                         4: 158.691e-6,
                         8: 317.383e-6
                         }
    MAX_SHAPE = (8192, 8192)
    aliases = ["Rayonix MX325"]

    def __init__(self, pixel1=79.346e-6, pixel2=79.346e-6):
        Rayonix.__init__(self, pixel1=pixel1, pixel2=pixel2)
