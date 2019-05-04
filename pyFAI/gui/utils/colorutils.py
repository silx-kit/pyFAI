# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "05/04/2019"


from silx.gui import qt
from silx.gui.colors import Colormap


def getFreeColorRange(colormap):
    """
    Returns a list of 10 colors in range not covered by colormap.

    :rtype: List[qt.QColor]
    """
    name = colormap['name']
    colormap = Colormap(name)

    # extract all hues from colormap
    colors = colormap.getNColors()
    hues = []
    for c in colors:
        c = qt.QColor.fromRgb(c[0], c[1], c[2])
        hues.append(c.hueF())

    # search the bigger empty hue range
    current = (0, 0.0, 0.2)
    hues = filter(lambda x: x >= 0, set(hues))
    hues = list(sorted(hues))
    if len(hues) > 1:
        for i in range(len(hues)):
            h1 = hues[i]
            h2 = hues[(i + 1) % len(hues)]
            if h2 < h1:
                h2 = h2 + 1.0

            diff = h2 - h1
            if diff > 0.5:
                diff = 1.0 - diff

            if diff > current[0]:
                current = diff, h1, h2
    elif len(hues) == 1:
        h = (hues[0] + 0.5) % 1.0
        current = (0, h - 0.1, h + 0.1)
    else:
        pass

    h1, h2 = current[1:]
    delta = (h2 - h1) / 6.0

    # move the range from the colormap
    h1, h2 = h1 + delta, h2 - delta
    hmin = (h1 + h2) / 2.0

    # generate colors with 3 hsv control points
    # (h1, 1, 1), (hmid, 1, 0.5), (h2, 1, 1)
    colors = []
    for i in range(5):
        h = h1 + (hmin - h1) * (i / 5.0)
        v = 0.5 + 0.1 * (5 - i)
        c = qt.QColor.fromHsvF(h % 1.0, 1.0, v)
        colors.append(c)
    for i in range(5):
        h = hmin + (h2 - hmin) * (i / 5.0)
        v = 0.5 + 0.1 * (i)
        c = qt.QColor.fromHsvF(h % 1.0, 1.0, v)
        colors.append(c)
    return colors
