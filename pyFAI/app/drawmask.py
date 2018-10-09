#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2016-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#             V. Aramdo Solé <sole@esrf.fr>
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


"""Use silx library to provide a widget to customize a mask """

__authors__ = ["Jerome Kieffer", "Valentin Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "09/10/2018"
__satus__ = "Production"

import os
import numpy
import logging
logging.basicConfig(level=logging.INFO)
logging.captureWarnings(True)
import fabio

_logger = logging.getLogger("drawmask")

import silx
if silx.version_info < (0, 2):
    raise ImportError("Silx version 0.2 or higher expected")
import silx.gui.plot
from silx.gui import qt
import pyFAI.utils
from argparse import ArgumentParser


class AbstractMaskImageWidget(qt.QMainWindow):
    """
    Abstract window application which allow to create a mask manually.
    """
    def __init__(self):
        qt.QMainWindow.__init__(self)
        self.setWindowTitle("pyFAI drawmask")
        self.__outputFile = None

        self._saveAndClose = qt.QPushButton("Save mask and quit")
        self._saveAndClose.clicked.connect(self.saveAndClose)
        self._saveAndClose.setMinimumHeight(50)

    def setOutputFile(self, outputFile):
        self.__outputFile = outputFile

    def saveAndClose(self):
        if self.__outputFile is None:
            filename = qt.QFileDialog.getSaveFileName(self)
            if isinstance(filename, tuple):
                # Compatibility with PyQt5
                filename = filename[0]

            if filename is None or filename == "":
                # Cancel the closing of the application
                return
            self.__outputFile = filename

        mask = self.getSelectionMask()
        fabio.edfimage.edfimage(data=mask).write(self.__outputFile)
        print("Mask-file saved into %s" % (self.__outputFile))
        self.close()


class MaskImageWidget(AbstractMaskImageWidget):
    """
    Window application which allow to create a mask manually.
    It is based on Silx library widgets.
    """
    def __init__(self):
        AbstractMaskImageWidget.__init__(self)

        self.__plot2D = silx.gui.plot.Plot2D()
        self.__plot2D.setKeepDataAspectRatio(True)
        if hasattr(self.__plot2D, "getMaskAction"):
            # silx 0.5 and later
            maskAction = self.__plot2D.getMaskAction()
        else:
            # silx 0.4 and previous
            maskAction = self.__plot2D.maskAction
        maskAction.setVisible(False)
        self.__maskPanel = silx.gui.plot.MaskToolsWidget.MaskToolsWidget(plot=self.__plot2D)
        try:
            colormap = {
                'name': "inferno",
                'normalization': 'log',
                'autoscale': True,
                'vmax': None,
                'vmin': None,
            }
            self.__plot2D.setDefaultColormap(colormap)
        except Exception:
            _logger.error("Impossible to change the default colormap. Source code not compatible.", exc_info=True)
        self.__maskPanel.setDirection(qt.QBoxLayout.TopToBottom)
        self.__maskPanel.setMultipleMasks("single")

        panelLayout = qt.QVBoxLayout()
        panelLayout.addWidget(self.__maskPanel)
        panelLayout.setStretch(0, 1)
        panelLayout.addWidget(self._saveAndClose)

        widget = qt.QWidget()
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout = qt.QHBoxLayout()
        layout.addWidget(self.__plot2D)
        layout.setStretch(0, 1)
        layout.addLayout(panelLayout)
        widget.setLayout(layout)

        self.setCentralWidget(widget)

    def setImageData(self, image):
        self.__plot2D.addImage(image)

    def getSelectionMask(self):
        return self.__maskPanel.getSelectionMask()


def postProcessId21(processFile, mask):
    """
    Post process asked by Marine Cotte (ID21)

    TODO: Remove it outside if it is possible. Ask them if it is still used.
    """
    print("Selected %i datapoints on file %s" % (mask.sum(), processFile[0]))
    for datafile in processFile:
        data = fabio.open(datafile).data[numpy.where(mask)]
        print("On File: %s,\t mean= %s \t std= %s" % (datafile, data.mean(), data.std()))


def main():
    usage = "pyFAI-drawmask file1.edf file2.edf ..."
    version = "pyFAI-average version %s from %s" % (pyFAI.version, pyFAI.date)
    description = """
    Draw a mask, i.e. an image containing the list of pixels which are considered invalid
    (no scintillator, module gap, beam stop shadow, ...).
    This will open a window and let you draw on the first image
    (provided) with different tools (brush, rectangle selection...)
    When you are finished, click on the "Save and quit" button.
    """
    epilog = """The mask image is saved into file1-masked.edf.
    Optionally the script will print the number of pixel masked
    and the intensity masked (as well on other files provided in input)"""
    parser = ArgumentParser(usage=usage, description=description, epilog=epilog)
    parser.add_argument("-v", "--version", action='version', version=version)
    parser.add_argument("args", metavar='FILE', type=str, nargs='+',
                        help="Files to be processed")

    options = parser.parse_args()
    if len(options.args) < 1:
        parser.error("Incorrect number of arguments: please provide an image to draw a mask")

    processFile = pyFAI.utils.expand_args(options.args)

    app = qt.QApplication([])

    window = MaskImageWidget()
    image = fabio.open(processFile[0]).data
    window.setImageData(image)
    window.show()
    outfile = os.path.splitext(processFile[0])[0] + "-mask.edf"
    window.setOutputFile(outfile)

    print("Your mask-file will be saved into %s" % (outfile))

    app.exec_()

    mask = window.getSelectionMask()
    postProcessId21(processFile, mask)


if __name__ == "__main__":
    main()
