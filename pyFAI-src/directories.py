#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""

Contains the directory name where data are:
 * gui directory with graphical user interface files
 * openCL directory with OpenCL kernels
 * calibrants directory with d-spacing files describing calibrants
 * testimages: if does not exist: create it.

This file is very short and simple in such a way to be mangled by installers
It is used by pyFAI.utils._get_data_path

See bug #144 for discussion about implementation
https://github.com/kif/pyFAI/issues/144
"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "22/06/2015"
__status__ = "development"

import os, getpass, tempfile
import logging
logger = logging.getLogger("pyFAI.directories")

PYFAI_DATA = "/usr/share/pyFAI"
PYFAI_TESTIMAGES = "/usr/share/pyFAI/testimages"

# testimage contains the directory name where
data_dir = None
if "PYFAI_DATA" in os.environ:
    data_dir = os.environ.get("PYFAI_DATA")
    if not os.path.exists(data_dir):
        logger.warning("data directory %s does not exist" % data_dir)
elif os.path.isdir(PYFAI_DATA):
    data_dir = PYFAI_DATA
else:
    data_dir = ""

# testimages contains the directory name where test images are located
testimages = None
if "PYFAI_TESTIMAGES" in os.environ:
    testimages = os.environ.get("PYFAI_TESTIMAGES")
    if not os.path.exists(testimages):
        logger.warning("testimage directory %s does not exist" % testimages)
else:
    testimages = os.path.join(data_dir, "testimages")
    if not os.path.isdir(testimages):
        # create a temporary folder
        testimages = os.path.join(tempfile.gettempdir(), "pyFAI_testimages_%s" % (getpass.getuser()))
        if not os.path.exists(testimages):
            try:
                os.makedirs(testimages)
            except OSError as err:
                logger.warning("Creating test_directory %s ended in error %s, probably a race condition" % (testimages, err))


