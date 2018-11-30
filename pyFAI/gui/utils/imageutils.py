# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides convenient functions to use with Qt objects.
"""

from __future__ import division


__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "27/11/2018"


import numpy

from silx.gui import qt
from silx.gui import colors


def convertArrayToQImage(image):
    """Convert an array-like RGB888 image to a QImage.

    The created QImage is using a copy of the array data.

    Limitation: Only supports RGB888 and RGBA8888 format.

    :param image: Array-like image data
    :type image: numpy.ndarray of uint8 of dimension HxWx3
    :return: Corresponding Qt image
    :rtype: QImage
    """
    # Possible extension: add a format argument to support more formats

    image = numpy.array(image, copy=False, order='C', dtype=numpy.uint8)

    height, width, depth = image.shape

    if depth == 3:
        qimage = qt.QImage(
            image.data,
            width,
            height,
            image.strides[0],  # bytesPerLine
            qt.QImage.Format_RGB888)
    elif depth == 4:
        qimage = qt.QImage(
            image.data,
            width,
            height,
            image.strides[0],  # bytesPerLine
            qt.QImage.Format_RGBA8888)
    else:
        assert(False)

    return qimage.copy()  # Making a copy of the image and its data


def maskArrayToRgba(mask, falseColor, trueColor):
    """
    Returns an RGBA uint8 numpy array using colors to map True (usually masked pixels)
    and Flase (valid pixel) from
    the mask array.
    """
    trueColor = numpy.array(colors.rgba(trueColor))
    trueColor = (trueColor * 256.0).clip(0, 255).astype(numpy.uint8)
    falseColor = numpy.array(colors.rgba(falseColor))
    falseColor = (falseColor * 256.0).clip(0, 255).astype(numpy.uint8)
    shape = mask.shape[0], mask.shape[1], 4
    image = numpy.empty(shape, dtype=numpy.uint8)
    image[mask] = trueColor
    image[~mask] = falseColor
    return image
