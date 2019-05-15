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
__date__ = "14/05/2019"

from .DataModel import DataModel


class FilenameModel(DataModel):
    """Model storing a filename and if the data is still synchronized.
    """

    def __init__(self, parent=None):
        DataModel.__init__(self, parent=parent)
        self.__isSynchronized = True

    def setValue(self, value):
        super(FilenameModel, self).setValue(value)
        self.__isSynchronized = False

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
        self.setValue(filename)

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
