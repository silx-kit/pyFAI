# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2023-2023 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

"""Orientation of detectors:

This module contains mostly an enum with the associated documentation
For orientation description, see http://sylvana.net/jpegcrop/exif_orientation.html
"""


__author__ = "Jérôme Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/12/2023"
__status__ = "stable"

from enum import IntEnum

class Orientation(IntEnum):
    """Names come from the position of the origin when looking at the sample from behind the camera.

    When looking from the sample to the detector, the right & left are swapped

    Some names (index 5-8) are inspired from
    https://learn.microsoft.com/en-us/uwp/api/windows.storage.fileproperties.photoorientation
    but unsupported
    """
    Unspecified = 0
    TopLeft = 1
    TopRight = 2
    BottomRight = 3
    BottomLeft = 4
    Transpose = 5
    Rotate270 = 6
    Transverse = 7
    Rotate90 = 8

Orientation(0).__doc__ = "An orientation flag is not set."
Orientation(0).available = False
Orientation(1).__doc__ = "Camera default. Origin at the top left of the image when looking at the sample."
Orientation(1).available = True
Orientation(2).__doc__ = "Origin at the top left of the image when looking from the sample."
Orientation(2).available = True
Orientation(3).__doc__ = "Native orientation of pyFAI. Origin at the bottom left when looking from the sample."
Orientation(3).available = True
Orientation(4).__doc__ = "Origin at the bottom left when looking at the sample."
Orientation(4).available = True
Orientation(5).__doc__ = "Rotate the photo counter-clockwise 270 degrees and then flip it horizontally. Unsupported for now."
Orientation(5).available = False
Orientation(6).__doc__ = "Rotate the photo counter-clockwise 270 degrees. Unsupported for now."
Orientation(6).available = False
Orientation(7).__doc__ = "Rotate the photo counter-clockwise 90 degrees and then flip it horizontally. Unsupported for now."
Orientation(7).available = False
Orientation(8).__doc__ = "Rotate the photo counter-clockwise 90 degrees. Unsupported for now."
Orientation(8).available = False
