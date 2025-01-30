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
__date__ = "28/01/2025"
__status__ = "development"

from silx.gui import qt
import sys
import argparse
from ..gui.pilx.MainWindow import MainWindow
from .. import version as pyFAI_version, __date__ as pyFAI_date


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-d", "--data", dest="data_path",
                        help="inner path to the dataset with the Raw Data",
                        default="/entry_0000/measurement/images_0001",)
    parser.add_argument("-p", "--nxprocess", dest="nxprocess_path",
                        help="inner path to the Nexus process with the integrated Data",
                        default="/entry_0000/pyFAI",)
    version = f"pyFAI-diffmap-view version {pyFAI_version}: {pyFAI_date}"
    parser.add_argument("-V", "--version", action='version', version=version)

    options = parser.parse_args(args)

    app = qt.QApplication([])
    window = MainWindow()
    window.initData(file_name=options.filename,
                    dataset_path=options.data_path,
                    nxprocess_path=options.nxprocess_path,
                    )
    window.show()
    return app.exec()

if __name__ == "__main__":
    main()
