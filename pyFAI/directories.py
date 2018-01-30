#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2014-2018 European Synchrotron Radiation Facility, Grenoble, France
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

Contains the directory name where data are:
 * gui directory with graphical user interface files
 * openCL directory with OpenCL kernels
 * calibrants directory with d-spacing files describing calibrants
 * testimages: if does not exist: create it.

This file is very short and simple in such a way to be mangled by installers
It is used by pyFAI.utils._get_data_path

See bug #144 for discussion about implementation
https://github.com/silx-kit/pyFAI/issues/144
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "10/01/2018"
__status__ = "development"

import os
import getpass
import tempfile
import logging
logger = logging.getLogger(__name__)

PYFAI_DATA = "/usr/share/pyFAI"

# testimage contains the directory name where
data_dir = None
if "PYFAI_DATA" in os.environ:
    data_dir = os.environ.get("PYFAI_DATA")
    if not os.path.exists(data_dir):
        logger.warning("data directory %s does not exist", data_dir)
elif os.path.isdir(PYFAI_DATA):
    data_dir = PYFAI_DATA
else:
    data_dir = ""

# testimages contains the directory name where test images are located
testimages = None
if "PYFAI_TESTIMAGES" in os.environ:
    testimages = os.environ.get("PYFAI_TESTIMAGES")
    if not os.path.exists(testimages):
        logger.warning("testimage directory %s does not exist", testimages)
else:
    testimages = os.path.join(data_dir, "testimages")
    if not os.path.isdir(testimages):
        # create a temporary folder
        testimages = os.path.join(tempfile.gettempdir(), "pyFAI_testimages_%s" % (getpass.getuser()))
        if not os.path.exists(testimages):
            try:
                os.makedirs(testimages)
            except OSError as err:
                logger.warning("Creating test_directory %s ended in error %s, probably a race condition", testimages, err)
