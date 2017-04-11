# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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

"""Jupyter helper functions
"""

from __future__ import division, print_function

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "11/04/2017"
__status__ = "Development"
__docformat__ = 'restructuredtext'

import numpy
from pylab import subplots, legend


def display(img=None, cp=None, ai=None, label=None, sg=None, ax=None):
    """Display an image with the control points and the calibrated rings
    in Jupyter notebooks

    :param img: 2D numpy array with an image
    :param cp: ControlPoint instance
    :param ai: azimuthal integrator for iso-2th curves
    :param label: name of the curve
    :param sg: single geometry object regrouping img, cp and ai
    :param ax: subplot object to display in, if None, a new one is created.
    :rerturn: Matplotlib subplot
    """
    if ax is None:
        _fig, ax = subplots()
    if sg is not None:
        if img is None:
            img = sg.image
        if cp is None:
            cp = sg.control_points
        if ai is None:
            ai = sg.geometry_refinement
        if label is None:
            label = sg.label

    ax.imshow(numpy.arcsinh(img), origin="lower", cmap="inferno")
    ax.set_title(label)
    if cp is not None:
        for lbl in cp.get_labels():
            pt = numpy.array(cp.get(lbl=lbl).points)
            ax.scatter(pt[:, 1], pt[:, 0], label=lbl)
        if ai is not None and cp.calibrant is not None:
            tth = cp.calibrant.get_2th()
            ttha = ai.twoThetaArray()
            ax.contour(ttha, levels=tth, cmap="autumn", linewidths=2, linestyles="dashed")
        legend()
    return ax
