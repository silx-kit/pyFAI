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
__date__ = "17/05/2019"

import numpy
from .AbstractModel import AbstractModel


class PeakModel(AbstractModel):

    def __init__(self, parent=None):
        super(PeakModel, self).__init__(parent)
        self.__name = None
        self.__color = None
        self.__coords = numpy.zeros((0, 2))
        self.__ringNumber = None
        self.__isEnabled = True

    def __len__(self):
        return len(self.__coords)

    def isValid(self):
        return self.__name is not None and self.__ringNumber is not None

    def name(self):
        return self.__name

    def setName(self, name):
        self.__name = name
        self.wasChanged()

    def isEnabled(self):
        """
        True if this group have to be taken into acount.

        :rtype: bool
        """
        return self.__isEnabled

    def setEnabled(self, isEnabled):
        """
        Set if this group have to be taken into acount.

        :param bool isEnabled: True to enable this group.
        """
        if self.__isEnabled == isEnabled:
            return
        self.__isEnabled = isEnabled
        self.wasChanged()

    def color(self):
        return self.__color

    def setColor(self, color):
        self.__color = color
        self.wasChanged()

    def coords(self):
        """
        Returns coords as numpy array.

        The first index identify a coord, the seconf identify the coord
        dimensions.

        List of axis/ord can be reached like that.

        .. code-block:: python

            coords = group.coords()
            yy = coords[:, 0]
            xx = coords[:, 1]
        """
        return self.__coords

    def setCoords(self, coords):
        """
        Set coords as numpy array.

        :param numpy.ndarray coords: Array of coords (1st dimension is the
            index of the coord; the second dimension contains y as first index,
            and x as second index).
        """
        assert(isinstance(coords, numpy.ndarray))
        assert(len(coords.shape) == 2)
        assert(coords.shape[1] == 2)
        coords = numpy.ascontiguousarray(coords)
        coords.flags['WRITEABLE'] = False
        self.__coords = coords
        self.wasChanged()

    def mergeCoords(self, coords):
        """Merge new coords to the current list of coords.

        Duplicated values are removed from the new coords, and the is added
        the end of the previous list.

        :param [numpy.ndarray,PeakModel] coords:
        """
        if isinstance(coords, PeakModel):
            coords = coords.coords()
        assert(isinstance(coords, numpy.ndarray))
        assert(len(coords.shape) == 2)
        assert(coords.shape[1] == 2)

        # Shortcuts
        if len(coords) == 0:
            return
        if len(self.__coords) == 0:
            self.setCoords(coords)
            return

        # Convert to structured array to use setdiff1d
        dtype = self.__coords.dtype.descr * self.__coords.shape[1]
        previous_coords = self.__coords.view(dtype)
        coords = numpy.ascontiguousarray(coords)
        new_coords = coords.view(dtype)
        new_coords = numpy.setdiff1d(new_coords, previous_coords)
        if len(new_coords) == 0:
            return
        new_coords = new_coords.view(self.__coords.dtype)
        new_coords.shape = -1, 2
        self.__coords = numpy.vstack((self.__coords, new_coords))
        self.__coords = numpy.ascontiguousarray(self.__coords)
        self.wasChanged()

    def ringNumber(self):
        return self.__ringNumber

    def setRingNumber(self, ringNumber):
        assert(ringNumber >= 1)
        self.__ringNumber = ringNumber
        self.wasChanged()

    def copy(self, parent=None):
        peakModel = PeakModel(parent)
        peakModel.setName(self.name())
        peakModel.setColor(self.color())
        peakModel.setCoords(self.coords())
        peakModel.setRingNumber(self.ringNumber())
        peakModel.setEnabled(self.isEnabled())
        return peakModel

    def distanceTo(self, coord):
        """Returns the smallest distance to this coord.

        None is retruned if the group contains no peaks.

        :param Tuple[float,float] coord: Distance to mesure
        """
        if len(self.__coords) == 0:
            return None
        coords = self.coords()
        coord = numpy.array(coord)
        distances = numpy.linalg.norm(coords - coord, axis=1)
        return distances.min()
