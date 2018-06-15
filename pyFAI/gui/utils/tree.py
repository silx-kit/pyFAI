# coding: utf-8
#
#    Project: Azimuthal integration
#             https://github.com/silx-kit/pyFAI
#
#    Copyright (C) 2015-2018 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
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

from __future__ import absolute_import, print_function, division

"""Module with list <-> tree conversion"""

__author__ = "Jerome Kieffer"
__contact__ = "Jerome.Kieffer@ESRF.eu"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__date__ = "05/03/2018"
__status__ = "development"
__docformat__ = 'restructuredtext'

import os
import logging
from ...io import is_hdf5
logger = logging.getLogger(__name__)

from collections import namedtuple
DataSetNT = namedtuple("DataSet", ("path", "h5", "nframes"))

try:
    from ...ext._tree import TreeItem
except ImportError:
    logger.error("pyFAI.ext._tree did not import")

    class TreeItem(object):
        """
        Node of a tree ... Needs synchronization with Cython code
        Deprecated !

        Contains:
        self.order: depth from root
        add name: reconstitute the full name
        add comment field for dirname and filenames
        add reorder command which will sort all sub-trees
        add size property which calculate the size of the subtree
        add a next/previous method
        """
        def __init__(self, label=None, parent=None):
            self.children = []
            self.parent = parent
            self.label = label
            if parent:
                parent.add_child(self)
                self.order = parent.order + 1
            else:
                self.order = 0
            self.extra = None

        def add_child(self, child):
            self.children.append(child)

        def has_child(self, label):
            return label in [i.label for i in self.children]

        def get(self, label):
            for i in self.children:
                if i.label == label:
                    return i

        def __repr__(self):
            if self.parent:
                return "TreeItem %s->%s with children: " % (self.parent.label, self.label) + ", ".join([i.label for i in self.children])
            else:
                return "TreeItem %s with children: " % (self.label) + ", ".join([i.label for i in self.children])

        def sort(self):
            for child in self.children:
                child.sort()
            self.children.sort(key=lambda x: x.label)

        @property
        def name(self):
            if not self.parent:
                return self.label or ""
            elif self.order == 1:
                return self.label or ""
            elif self.order == 4:
                return os.path.join(self.parent.name, self.label)
            else:
                return "%s-%s" % (self.parent.name, self.label)

        def next(self):
            if self.parent is None:
                raise IndexError("Next does not exist")
            idx = self.parent.children.index(self)
            if idx < len(self.parent.children) - 1:
                return self.parent.children[idx + 1]
            else:
                return self.parent.next().children[0]

        def previous(self):
            if self.parent is None:
                raise IndexError("Previous does not exist")
            idx = self.parent.children.index(self)
            if idx > 0:
                return self.parent.children[idx - 1]
            else:
                return self.parent.previous().children[-1]

        def first(self):
            if self.children:
                return self.children[0].first()
            else:
                return self

        def last(self):
            if self.children:
                return self.children[-1].last()
            else:
                return self

        @property
        def size(self):
            if self.children:
                return sum([child.size for child in self.children])
            else:
                return 1


class DataSet(object):
    def __init__(self, path, h5=None, nframes=None, shape=None):
        self.path = path
        self.h5 = h5
        self.nframes = nframes
        self.shape = shape

    def as_tuple(self):
        return DataSetNT(self.path, self.h5, self.nframes)

    def is_hdf5(self):
        """Return True if the object is hdf5"""
        if self.h5 is None:
            self.h5 = is_hdf5(self.path)
        return bool(self.h5)

    def __len__(self):
        return self.nframes or 1


class ListDataSet(list):

    def commonroot(self):
        """
        :return: common directory
        """
        ll = [j.path.split(os.sep) for j in self]
        common = os.path.commonprefix(ll)
        if common:
            return os.sep.join(common + [""])

    def as_tree(self, sep=os.path.sep):
        """Convert the list into a tree

        :param sep: separator in the filenames
        :return: Root of the tree
        """
        prefix = self.commonroot()
        root = TreeItem()
        common = TreeItem(prefix, root)
        lprefix = len(prefix) if prefix else 0
        for dataset in self:
            base = dataset.path[lprefix:]
            elts = base.split(sep)
            element = common
            for item in elts:
                child = element.get(item)
                if not child:
                    child = TreeItem(item, element)
                element = child
        return root

    def empty(self):
        while self:
            self.pop()
