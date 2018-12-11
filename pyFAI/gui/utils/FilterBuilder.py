# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2017-2018 European Synchrotron Radiation Facility
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
"""This module provides helper to build filter descriptions.
"""

from __future__ import division


__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "21/11/2018"

import functools
import collections


class FilterBuilder(object):
    """
    Helper to create extension filters for file dialog
    """
    COMPRESSED_IMAGE_EXTENSIONS = ["gz", "bz2"]
    """List of compressed file extension supported by fabio by default."""

    def __init__(self):
        self.__formats = collections.OrderedDict()
        self.__any = True
        self.__all = True

    def __normalizeExtensions(self, extensions):
        if isinstance(extensions, list):
            extensions = list(extensions)
        else:
            extensions = extensions.split(" ")
        return extensions

    def addImageFormat(self, description, extensions, compressedExtensions=True):
        """Add an image format to the filters

        :param str description: Description of the file format
        :param Union[str,List] extensions: Description of the file format
        :param bool compressedExtensions: Includes derived compressed files
            like gz and bz2.
        """
        extensions = self.__normalizeExtensions(extensions)
        if compressedExtensions:
            otherExtensions = []
            for ext in extensions:
                for subext in self.COMPRESSED_IMAGE_EXTENSIONS:
                    otherExtensions.append("%s.%s" % (ext, subext))
            extensions.extend(otherExtensions)
        self.addFileFormat(description, extensions)

    def addFileFormat(self, description, extensions):
        """Add a file format to the filters

        :param str description: Description of the file format
        :param Union[str,List] extensions: Description of the file format
        """
        extensions = self.__normalizeExtensions(extensions)
        self.__formats[description] = extensions

    def getFilters(self):
        """Returns the filters as supported by :meth:`qt.QFileDialog.setNameFilters`.

        :rtype: List[str]
        """
        filters = []
        if self.__all and len(self.__formats):
            allExtensions = functools.reduce(lambda a, b: a + b, self.__formats.values())
            allExtensions = ["*.%s" % ext for ext in allExtensions]
            allExtensions = " ".join(allExtensions)
            filters.append("All supported files (%s)" % allExtensions)
        for description, extensions in self.__formats.items():
            extensions = ["*.%s" % ext for ext in extensions]
            extensions = " ".join(extensions)
            filters.append("%s (%s)" % (description, extensions))
        if self.__any:
            filters.append("All files (*)")
        return filters
