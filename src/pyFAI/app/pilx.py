#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2024-2025 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Loïc Huder (loic.huder@ESRF.eu)
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

"""Tool to visualize diffraction maps."""

__author__ = "Loïc Huder"
__contact__ = "loic.huder@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "18/06/2025"
__status__ = "development"

from silx.gui import qt
import sys
import argparse
import logging
from ..gui.pilx.MainWindow import MainWindow
from .. import version as pyFAI_version, date as pyFAI_date
from ..io.nexus import Nexus

logger = logging.getLogger(__name__)


def guess_file_type(filename, default="diffmap"):
    """return the type of HDF5-file to set the proper reader"""

    def read_str(entry, key):
        try:
            raw = entry[key][()]
        except:
            res = ""
        else:
            if isinstance(raw, bytes):
                raw = raw.decode()
            res = raw.lower()
        return res

    with Nexus(filename, "r") as nxs:
        entry = nxs.get_entries()[0]
        program_name = read_str(entry, "program_name")
        title = read_str(entry,"title")

    if program_name == "bm29.mesh" or title == "biosaxs mesh experiment":
        filetype = "bm29"
    elif program_name=="pyfai" or title == "diff_map":
        filetype = "diffmap"
    else:
        logger.warning(f"Unable to identify file: '{filename}' with program_name: '{program_name}' and title: '{title}'")
        filetype = ""
    logger.info(f"detected file type: {filetype}")
    return filetype or default


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-d", "--data", dest="data_path",
                        help="inner path to the dataset with the Raw Data, by default '/entry_0000/measurement/images_0001'",
                        default=None, type=str)
    parser.add_argument("-p", "--nxprocess", dest="nxprocess_path",
                        help="inner path to the Nexus process with the integrated data, by default '/entry_0000/pyFAI'",
                        default=None,type=str)
    parser.add_argument("--reader", help="select the default reader among `diffmap` and `bm29`",
                        default="auto", type=str)
    version = f"pyFAI-diffmap-view version {pyFAI_version}: {pyFAI_date}"
    parser.add_argument("-V", "--version", action='version', version=version)
    parser.add_argument("-v", "--verbose", help="increase verbosity",
                        action='count', default=0)
    options = parser.parse_args(args)
    if options.verbose == 0:
        logging.basicConfig(level=logging.WARNING)
    elif options.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    elif options.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)

    reader = options.reader.lower()
    if reader == "auto":
        reader = guess_file_type(options.filename)
    if reader == "bm29":
        data_path = "/entry_0000/1_mesh/sources/images_0000"
        nxprocess_path = "/entry_0000/1_mesh"
    elif reader == "diffmap":
        data_path = "/entry_0000/measurement/images_0001"
        nxprocess_path = "/entry_0000/pyFAI"

    nxprocess_path = options.nxprocess_path if options.nxprocess_path is not None else nxprocess_path
    data_path = options.data_path if options.data_path is not None else data_path

    app = qt.QApplication([])
    window = MainWindow()
    window.initData(file_name=options.filename,
                    dataset_path=data_path,
                    nxprocess_path=nxprocess_path,
                    )
    window.show()
    return app.exec()

if __name__ == "__main__":
    main()
