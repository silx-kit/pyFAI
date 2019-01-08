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
__date__ = "08/01/2019"

import sys
import os.path
import numpy

from silx.gui import qt
import silx.math.combo
from silx.gui.plot3d.items.mesh import Mesh
from silx.gui.plot3d.SceneWindow import SceneWindow
from silx.gui import colors


class Detector3dDialog(qt.QDialog):
    """Dialog to display a selected geometry
    """

    def __init__(self, parent=None):
        super(Detector3dDialog, self).__init__(parent=parent)
        self.__plot = SceneWindow(self)
        layout = qt.QVBoxLayout(self)
        layout.addWidget(self.__plot)

    def __createMesh(self, pixels, image, mask, colormap):
        """
        Create a 3D image from pyFAI pixel detector definitions, and a raw image
        from the detector.

        :rtype: items.DataItem3D
        """

        height, width, _, _, = pixels.shape
        nb_vertices = width * height * 6

        # Allocate contiguous memory
        positions_array = numpy.empty((nb_vertices, 3), dtype=numpy.float32)
        colors_array = numpy.empty((nb_vertices, 4), dtype=numpy.float32)

        # Merge all pixels together
        pixels = pixels[...]
        pixels.shape = -1, 4, 3

        # Normalize the colormap as a RGBA float lookup table
        lut = colormap.getNColors(256)
        lut = lut / 255.0

        cursor_color = colors.cursorColorForColormap(colormap.getName())
        cursor_color = numpy.array(colors.rgba(cursor_color))

        # Normalize the image as lookup table to colormap lookup table
        if image is not None:
            image = image.view()
            image.shape = -1
            image = numpy.log(image)
            info = silx.math.combo.min_max(image, min_positive=True, finite=True)
            image = (image - info.min_positive) / float(info.maximum - info.min_positive)
            image = (image * 255.0).astype(int)
            image = image.clip(0, 255)

        if mask is not None:
            mask = mask.view()
            mask.shape = -1

        masked_color = numpy.array([1.0, 0.0, 1.0, 1.0])
        default_color = numpy.array([0.8, 0.8, 0.8, 1.0])

        triangle_index = 0
        color_index = 0

        for npixel, pixel in enumerate(pixels):
            masked = False
            if mask is not None:
                masked = mask[npixel] != 0

            if masked:
                color = masked_color
            elif image is not None:
                color_id = image[color_index]
                color = lut[color_id]
            else:
                color = default_color

            positions_array[triangle_index + 0] = pixel[0]
            positions_array[triangle_index + 1] = pixel[1]
            positions_array[triangle_index + 2] = pixel[2]
            colors_array[triangle_index + 0] = color
            colors_array[triangle_index + 1] = color
            colors_array[triangle_index + 2] = color
            triangle_index += 3
            positions_array[triangle_index + 0] = pixel[2]
            positions_array[triangle_index + 1] = pixel[3]
            positions_array[triangle_index + 2] = pixel[0]
            colors_array[triangle_index + 0] = color
            colors_array[triangle_index + 1] = color
            colors_array[triangle_index + 2] = color
            triangle_index += 3
            color_index += 1

        mesh = Mesh()
        mesh.setData(position=positions_array, color=colors_array)
        return mesh

    def setData(self, detector, image=None, mask=None, colormap=None):
        pixels = detector.get_pixel_corners()

        acquisition_filename = sys.argv[2]
        if not os.path.exists(acquisition_filename):
            raise Exception("File not found")

        sceneWidget = self.__plot.getSceneWidget()

        if colormap is None:
            colormap = colors.Colormap("inferno")

        mesh = self.__createMesh(pixels, image, mask, colormap)
        sceneWidget.addItem(mesh)
