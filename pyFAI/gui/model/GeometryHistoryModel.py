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
__date__ = "18/04/2019"

from .ListModel import ListModel
from .GeometryModel import GeometryModel
from .AbstractModel import AbstractModel


class StoredGeometry(AbstractModel):
    """
    Single element stored in the history of geometries.

    :param datetime.datetime time: time of the record
    :param GeometryModel geometry: Geometry to store
    :param float rms:
    """

    def __init__(self, parent, label, time, geometry, rms):
        super(StoredGeometry, self).__init__(parent=parent)
        # Store this values in a compact format
        d = geometry.distance().value()
        w = geometry.wavelength().value()
        p1 = geometry.poni1().value()
        p2 = geometry.poni2().value()
        r1 = geometry.rotation1().value()
        r2 = geometry.rotation2().value()
        r3 = geometry.rotation3().value()
        self.__geometry = (d, w, p1, p2, r1, r2, r3)
        self.__label = label
        self.__rms = rms
        self.__time = time

    def geometry(self):
        """
        :rtype: GeometryModel
        """
        geometry = GeometryModel()
        d, w, p1, p2, r1, r2, r3 = self.__geometry
        geometry.distance().setValue(d)
        geometry.wavelength().setValue(w)
        geometry.poni1().setValue(p1)
        geometry.poni2().setValue(p2)
        geometry.rotation1().setValue(r1)
        geometry.rotation2().setValue(r2)
        geometry.rotation3().setValue(r3)
        return geometry

    def label(self):
        """
        :rtype: str
        """
        return self.__label

    def rms(self):
        """
        :rtype: float
        """
        return self.__rms

    def time(self):
        """
        :rtype: datetime.datetime
        """
        return self.__time


class GeometryHistoryModel(ListModel):

    def appendGeometry(self, label, time, geometry, rms):
        """
        :param str label: Named geometry
        :param datetime.datetime time: time of the record
        :param GeometryModel geometry: Geometry to store
        :param float rms: Root mean share of this geometry
        """
        self.append(StoredGeometry(self, label, time, geometry, rms))
