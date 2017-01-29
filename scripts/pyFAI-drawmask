#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Azimuthal integration
#             https://github.com/kif/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
#
#    Authors: Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#             V. Aramdo Solé <sole@esrf.fr>
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
pyFAI-drawmask

Use silx or PyMca module to define a mask
"""

__authors__ = ["Jerome Kieffer", "Valentin Valls"]
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "GPLv3+"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "26/10/2016"
__satus__ = "Production"
import os
import numpy
import logging
logging.basicConfig()
import fabio

try:
    import silx
    if silx.version_info < (0,2):
        raise ImportError("Silx version 0.2 or higher expected")
    import silx.gui.plot
    from silx.gui import qt
    BACKEND = "SILX"
except ImportError:
    try:
        import PyMca.MaskImageWidget as PyMcaMaskImageWidget
    except ImportError:
        import PyMca5.PyMca.MaskImageWidget as PyMcaMaskImageWidget
    from pyFAI.gui import qt
    BACKEND = "PYMCA"
except:
    BACKEND = None

import pyFAI.utils

try:
    from argparse import ArgumentParser
except ImportError:
    from pyFAI.third_party.argparse import ArgumentParser

_logger = logging.getLogger("drawmask")


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
            self.__outputFile = qt.QFileDialog.getSaveFileName(self)
            if self.__outputFile == "":
                # Save dialog cancelled
                self.__outputFile = None
                return
        if self.__outputFile is not None:
            mask = self.getSelectionMask()
            fabio.edfimage.edfimage(data=mask).write(self.__outputFile)
            print("Mask-file saved into %s" % (self.__outputFile))
            self.close()


if BACKEND == "SILX":

    class MaskImageWidget(AbstractMaskImageWidget):
        """
        Window application which allow to create a mask manually.
        It is based on Silx library widgets.
        """
        def __init__(self):
            AbstractMaskImageWidget.__init__(self)

            self.__plot2D = silx.gui.plot.Plot2D()
            self.__plot2D.setKeepDataAspectRatio(True)
            self.__plot2D.maskAction.setVisible(False)
            self.__maskPanel = silx.gui.plot.MaskToolsWidget.MaskToolsWidget(plot=self.__plot2D)
            try:
                colormap = {
                    'name': "inferno",
                    'normalization': 'log',
                    'autoscale': True,
                }
                self.__plot2D.setDefaultColormap(colormap)
            except:
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

elif BACKEND == "PYMCA":

    class MaskImageWidget(AbstractMaskImageWidget):
        """
        Window application which allow to create a mask manually.
        It is based on PyMCA library widgets.
        """
        def __init__(self):
            AbstractMaskImageWidget.__init__(self)

            self.__maskWidget = PyMcaMaskImageWidget.MaskImageWidget()

            widget = qt.QWidget()
            layout = qt.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.__maskWidget)
            layout.setStretch(0, 1)
            layout.addWidget(self._saveAndClose)
            widget.setLayout(layout)

            self.setCentralWidget(widget)

        def setImageData(self, image):
            self.__maskWidget.setImageData(image)

        def getSelectionMask(self):
            return self.__maskWidget.getSelectionMask()

else:
    raise Exception("Unsupported backend %s" % BACKEND)


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
