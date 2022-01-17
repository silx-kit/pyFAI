#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""
The Matplotlib-calibration widget specialized for Qt

A tool for determining the geometry of a detector using a reference sample.

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "17/01/2022"
__status__ = "development"

import logging
logger = logging.getLogger(__name__)
try:
    from silx.gui import qt
except ImportError:
    logger.debug("Backtrace", exc_info=True)
    qt = None

if qt is not None:
    from .utils import update_fig, maximize_fig
    from .matplotlib import matplotlib, pyplot, colors
    from . import utils as gui_utils

from .mpl_calib import MplCalibWidget


class QtMplCalibWidget(MplCalibWidget):
    """Specialized class based on Qt
    
    Provides a unified interface for:
    - display the image
    - Plot group of points
    - overlay the contour-plot for 2th values
    - shade out some part of the image
    - annotate rings
    """

    def init(self, pick=True):
        if self.fig is None:
            self.fig, (self.ax, self.axc) = pyplot.subplots(1, 2, gridspec_kw={"width_ratios": (10, 1)})
            self.ax.set_ylabel('y in pixels')
            self.ax.set_xlabel('x in pixels')
            # self.axc.yaxis.set_label_position('left')
            # self.axc.set_ylabel("Colorbar")
            toolbar = self.fig.canvas.toolbar
            toolbar.addSeparator()
            a = toolbar.addAction('Opts', self.onclick_option)
            a.setToolTip('open options window')
            if pick:
                label = qt.QLabel("Ring #", toolbar)
                toolbar.addWidget(label)
                self.spinbox = qt.QSpinBox(toolbar)
                self.spinbox.setMinimum(0)
                self.sb_action = toolbar.addWidget(self.spinbox)
                a = toolbar.addAction('Refine', self.onclick_refine)
                a.setToolTip('switch to refinement mode')
                self.ref_action = a
                self.mpl_connectId = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def update(self):
        if self.fig:
            update_fig(self.fig)

    def maximize(self, update=True):
        if self.fig:
            maximize_fig(self.fig)
            if update:
                self.update()

    def get_ring_value(self):
        "Return the value of the ring widget"
        return self.spinbox.value()
