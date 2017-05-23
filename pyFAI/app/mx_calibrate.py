#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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

MX-calibrate is a tool to calibrate the distance of a detector from a set of powder diffraction patterns

Idea:

MX-calibrate -e 12 --spacing dSpacing.D file1.edf file2.edf file3.edf

calibrate the by hand the most distant frame then calibrate subsequently all frames

"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/05/2017"
__satus__ = "development"

import logging
import pyFAI.calibration
try:
    from pyFAI.third_party import six
except (ImportError, Exception):
    import six
try:
    from rfoo.utils import rconsole
    rconsole.spawn_server()
except ImportError:
    logging.debug("No socket opened for debugging. Please install rfoo")


def main():
    c = pyFAI.calibration.MultiCalib()
    c.parse()
    c.read_pixelsSize()
    c.read_dSpacingFile()
    c.process()
    c.regression()
    six.moves.input("Press enter to quit")

if __name__ == "__main__":
    main()
