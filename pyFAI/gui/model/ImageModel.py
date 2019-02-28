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
__date__ = "28/02/2019"

import numpy

from silx.gui import qt

from .DataModel import DataModel


class ImageModel(DataModel):

    def setValue(self, value):
        """Set the value of this image model."""
        if value is not None:
            if not isinstance(value, numpy.ndarray):
                raise TypeError("A numpy array is expected, but %s was found." % value.__class__.__name__)
            if len(value.shape) != 2:
                raise TypeError("A 2d array is expected, but %s was found." % value.shape)
            if value.dtype.kind not in "uif":
                raise TypeError("A numeric array is expected, but %s was found." % value.dtype.kind)
            previous = self.value()
            if previous is value:
                # Filter same images
                return
            if previous is not None and numpy.array_equal(value, previous):
                # Filter same images
                return
        super(ImageModel, self).setValue(value)


class ImageFilenameModel(DataModel):
    """Model storing an image using it's filename."""

    filenameChanged = DataModel.changed

    def hasFilename(self):
        """True if this model contains a filename.

        :rtype: bool
        """
        return self.value() is not None

    def filename(self):
        """Returns the filename associated with this model.

        :rtype: Union[None,str]
        """
        return self.value()

    def setFilename(self, filename):
        """
        Set a filename to this model

        :param str filename: The new filename
        """
        return self.setValue(filename)


class ImageFromFilenameModel(DataModel):
    """Model storing an image array which could come from a filename.

    This model deal with unsynchronized filename/data.
    """

    filenameChanged = qt.Signal()

    def __init__(self, parent=None):
        self.__filename = None
        self.__wasChanged = False
        DataModel.__init__(self, parent=parent)

    def setValue(self, value):
        """Set the value of this image model."""
        if value is not None:
            if not isinstance(value, numpy.ndarray):
                raise TypeError("A numpy array is expected, but %s was found." % value.__class__.__name__)
            if len(value.shape) != 2:
                raise TypeError("A 2d array is expected, but %s was found." % value.shape)
            if value.dtype.kind not in "uif":
                raise TypeError("A numeric array is expected, but %s was found." % value.dtype.kind)
            previous = self.value()
            if previous is value:
                # Filter same images
                return
            if previous is not None and numpy.array_equal(value, previous):
                # Filter same images
                return
        super(ImageFromFilenameModel, self).setValue(value)
        self.__isSynchronized = False

    def hasFilename(self):
        """True if this model contains a filename.

        :rtype: bool
        """
        return self.__filename is not None

    def filename(self):
        """Returns the filename associated with this model.

        :rtype: Union[None,str]
        """
        return self.__filename

    def unlockSignals(self):
        self.__filenameWasChanged()
        self.__wasChanged = False
        return DataModel.unlockSignals(self)

    def __filenameWasChanged(self):
        """Emit the change event in case of the model was not locked.

        :returns: True if the signal was emitted.
        """
        if self.isLocked():
            self.__wasChanged = True
            return False
        else:
            self.filenameChanged.emit()
            return True

    def setFilename(self, filename):
        """
        Set a filename to this model

        :param str filename: The new filename
        """
        self.__filename = filename
        self.__isSynchronized = False
        self.__filenameWasChanged()

    def setSynchronized(self, isSynchronized):
        """"
        Set if the filename and the data are synchronized.
        """
        self.__isSynchronized = isSynchronized

    def isSynchronized(self):
        """Returns True if the filename and the data are synchronized.

        Both contains the same data.
        """
        return self.__isSynchronized
