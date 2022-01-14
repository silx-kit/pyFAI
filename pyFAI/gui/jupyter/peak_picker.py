#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2021-2021 European Synchrotron Radiation Facility, Grenoble, France
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

"""Semi-graphical tool for peak-picking and extracting visually control points
from an image with Debye-Scherer rings in Jupyter environment"""

__authors__ = ["Philipp Hans", "Jérôme Kieffer"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "14/01/2022"
__status__ = "production"

import logging
logger = logging.getLogger(__name__)
import numpy
from matplotlib.pyplot import subplots
from ..peak_picker import PeakPicker as _PeakPicker, preprocess_image
from .calib import JupyCalibWidget
try:
    import ipywidgets as widgets
    from IPython.display import display
except ModuleNotFoundError:
    logger.error("`ipywidgets` and `IPython` are needed to perform the calibration in Jupyter")


class JupyPeakPicker(_PeakPicker):

    def gui(self, log=False, maximize=False, pick=True):
        """
        :param log: show z in log scale
        """
        data_disp, bounds = preprocess_image(self.data, False, 1e-3)
        if self.widget is None:
            self.widget = JupyCalibWidget(click_cb=self.onclick,
                                         refine_cb=None,
                                         option_cb=None,)
            self.widget.init(image=data_disp, bounds=bounds)
        else:
            self.widget.imshow(data_disp, bounds=bounds, log=True, update=False)
        if self.detector:
            self.widget.set_detector(self.detector, update=False)
        if maximize:
            self.widget.maximize()
        else:
            display(self.widget)  # .update()

    def onclick(self, *args):
        _PeakPicker.onclick(self, *args)
