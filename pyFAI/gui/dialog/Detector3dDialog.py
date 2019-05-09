# coding: utf-8
# /*##########################################################################
#
# Copyright (C) 2016-2018 European Synchrotron Radiation Facility
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
#
# ###########################################################################*/

from __future__ import absolute_import

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "09/05/2019"

import numpy
import time
import functools
import logging

_logger = logging.getLogger(__name__)

from silx.utils import html
from silx.gui import qt
from silx.gui.plot3d.items import mesh
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui import colors


class CreateSceneThread(qt.QThread):

    progressValue = qt.Signal(int)

    def __init__(self, parent=None):
        super(CreateSceneThread, self).__init__(parent=parent)
        self.__detector = None
        self.__image = None
        self.__mask = None
        self.__colormap = None
        self.__geometry = None
        self.__last = None

    def setDetector(self, detector):
        self.__detector = detector

    def setImage(self, image):
        self.__image = image

    def setMask(self, mask):
        self.__mask = mask

    def setColormap(self, colormap):
        self.__colormap = colormap

    def setGeometry(self, geometry):
        self.__geometry = geometry

    def emitProgressValue(self, value, force=False):
        now = time.time()
        if not force and self.__last is not None and now - self.__last < 1.0:
            # Filter events every seconds
            return
        self.__last = now
        self.progressValue.emit(value)

    def errorString(self):
        return self.__error

    def isAborted(self):
        """
        Returns whether the theard has aborted or not.

        .. note:: Aborted thead are not finished theads.
        """
        return self.__isAborted

    def run(self):
        self.__isAborted = False
        try:
            result = self.runProcess()
        except Exception as e:
            _logger.error("Backtrace", exc_info=True)
            self.__error = str(e)
            self.__isAborted = True
        else:
            if not result:
                self.__error = "Task was aborted"
                self.__isAborted = True

    def runProcess(self):
        self.emitProgressValue(0, force=True)
        result = self.__createDetectorColormapMesh()
        return result

    def __createDetectorColormapMesh(self):
        if self.__geometry is not None:
            if self.__detector is not None:
                self.__geometry.detector = self.__detector
            pixels = self.__geometry.calc_pos_zyx(corners=True)
            pixels = numpy.array(pixels)
            pixels = numpy.moveaxis(pixels, 0, -1)
        else:
            pixels = self.__detector.get_pixel_corners()

        # Merge all pixels together
        pixels = pixels[...]
        vertices = numpy.reshape(pixels, (-1, 3))

        if self.__image is not None:
            pixelValues = self.__image
            pixelValues = pixelValues.reshape(-1)
        else:
            pixelValues = numpy.zeros(vertices.shape[0] // 4)

        if self.__mask is not None:
            mask = self.__mask.reshape(-1)
            if pixelValues.dtype.kind in "ui":
                pixelValues = pixelValues.astype(numpy.float)
            pixelValues[mask != 0] = numpy.float("nan")

        values = numpy.empty(shape=(vertices.shape[0]))
        values[0::4] = pixelValues
        values[1::4] = pixelValues
        values[2::4] = pixelValues
        values[3::4] = pixelValues

        plus = numpy.array([0, 1, 2, 2, 3, 0], dtype=numpy.uint32)
        indexes = (numpy.atleast_2d(4 * numpy.arange(vertices.shape[0] // 4, dtype=numpy.uint32)).T + plus).ravel()
        indexes = indexes.astype(numpy.uint32)

        colormap = self.__colormap
        if colormap is None:
            colormap = colors.Colormap(name="inferno", normalization=colors.Colormap.LOGARITHM)

        item = mesh.ColormapMesh()
        item.moveToThread(qt.QApplication.instance().thread())
        item.setData(mode="triangles",
                     position=vertices,
                     value=values,
                     indices=indexes,
                     copy=False)
        item.setColormap(colormap)
        self.__detectorItem = item
        return True

    def hasGeometry(self):
        return self.__geometry is not None

    def getDetectorItem(self):
        item = self.__detectorItem
        item.setLabel("Detector")
        self.__detectorItem = None
        return item

    def getSampleItem(self):
        item = mesh.Hexagon()

        # 3mm
        size = 0.003
        distance = self.__geometry.dist
        if size >= distance:
            size = distance / 2

        item.setData(color=(0, 1.0, 1.0, 0.5),
                     radius=size,
                     height=size,
                     position=(-size / 2, 0, 0),
                     rotation=(90, (0, 1, 0))
                     )
        item.setLabel("Sample")
        return item

    def getBeamItem(self):
        item = mesh.Cylinder()

        # 0.5mm
        size = 0.0005
        distance = self.__geometry.dist
        if size >= distance:
            size = distance / 6

        dist = self.__geometry.dist
        item.setData(color=(1.0, 0, 0),
                     radius=size,
                     height=dist,
                     position=(dist / 2, 0, 0),
                     rotation=(90, (0, 1, 0))
                     )
        item.setLabel("Beam")
        return item


class Detector3dDialog(qt.QDialog):
    """Dialog to display a selected geometry
    """

    def __init__(self, parent=None):
        super(Detector3dDialog, self).__init__(parent=parent)
        self.setWindowTitle("Display sample stage")
        self.__plot = SceneWindow(self)
        self.__plot.setVisible(False)

        self.__process = qt.QProgressBar(self)
        self.__process.setFormat("Processing data")
        self.__process.setRange(0, 100)

        self.__buttons = qt.QDialogButtonBox(self)
        self.__buttons.addButton(qt.QDialogButtonBox.Cancel)
        self.__buttons.accepted.connect(self.accept)
        self.__buttons.rejected.connect(self.reject)

        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__plot)
        layout.addWidget(self.__process)
        layout.addWidget(self.__buttons)

    def __detectorLoaded(self, thread):
        if thread.isAborted():
            template = "<html>3D preview cancelled:<br/>%s</html>"
            message = template % html.escape(thread.errorString())
            self.setVisible(False)
            qt.QMessageBox.critical(self, "Error", message)
            self.deleteLater()
            return
        self.__process.setVisible(False)
        self.__plot.setVisible(True)
        self.__buttons.clear()
        self.__buttons.addButton(qt.QDialogButtonBox.Close)
        self.adjustSize()

        sceneWidget = self.__plot.getSceneWidget()
        item = thread.getDetectorItem()
        sceneWidget.addItem(item)
        if thread.hasGeometry():
            # Display the full sample stage
            item = thread.getBeamItem()
            sceneWidget.addItem(item)
            item = thread.getSampleItem()
            sceneWidget.addItem(item)
        sceneWidget.resetZoom(face="left")

    def __detectorLoading(self, percent):
        self.__process.setValue(percent)

    def setData(self, detector=None, image=None, mask=None, colormap=None, geometry=None):
        thread = CreateSceneThread(self)
        thread.setGeometry(geometry)
        thread.setDetector(detector)
        thread.setImage(image)
        thread.setMask(mask)
        thread.setColormap(colormap)

        thread.finished.connect(functools.partial(self.__detectorLoaded, thread))
        thread.finished.connect(thread.deleteLater)
        thread.progressValue.connect(self.__detectorLoading)
        self.rejected.connect(thread.requestInterruption)
        thread.start()
