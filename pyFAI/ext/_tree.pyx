#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#    Project: Fast Azimuthal integration
#             https://github.com/pyFAI/pyFAI
#
#    Copyright (C) European Synchrotron Radiation Facility, Grenoble, France
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

__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "31/05/2016"
__copyright__ = "2011-2015, ESRF"
__contact__ = "jerome.kieffer@esrf.fr"

import cython
import os

cdef class TreeItem(object):
    """
    Node of a tree ...

    Contains:
    self.order: depth from root
    add name: reconstitute the full name
    add comment field for dirname and filenames
    add reorder command which will sort all sub-trees
    add size property which calculate the size of the subtree
    add a next/previous method
    """
    cdef public list children
    cdef public TreeItem parent
    cdef public str label
    cdef public int order
    cdef public str type #can by "dir", "file", "group" or "dataset"
    cdef public object extra

    def __init__(self, str label=None, TreeItem parent=None):
        self.children = []
        self.parent = parent
        self.label = label or ""
        if parent:
            parent.add_child(self)
            self.order = parent.order + 1
        else:
            self.order = 0
        self.extra = None

    cpdef add_child(self, TreeItem child):
        self.children.append(child)
        child.parent = self

    cpdef update(self, TreeItem new_root):
        """
        Add new children in tree
        """
        for new_child in new_root.children:
            child = self.get(new_child.label)
            if child:
                child.update(new_child)
            else:
                self.add_child(new_child)

    cpdef bint has_child(self, str label):
        return label in [i.label for i in self.children]

    cpdef TreeItem get(self, str label):
        for i in self.children:
            if i.label == label:
                return i

    def __repr__(self):
        if self.parent:
            return "TreeItem %s->%s with children: " % (self.parent.label, self.label) + ", ".join([i.label for i in self.children])
        else:
            return "Root TreeItem %s with children: %s" % (self.label, ", ".join([i.label for i in self.children]))

    def sort(self):
        for child in self.children:
            child.sort()
        self.children.sort(key=lambda x: x.label)

    cpdef TreeItem next(self):
        cdef int idx
        if self.parent is None:
            raise IndexError("Next does not exist")
        idx = self.parent.children.index(self)
        if idx < len(self.parent.children) - 1:
            return self.parent.children[idx + 1]
        else:
            return self.parent.next().children[0]

    cpdef TreeItem previous(self):
        cdef int idx
        if self.parent is None:
            raise IndexError("Previous does not exist")
        idx = self.parent.children.index(self)
        if idx > 0:
            return self.parent.children[idx - 1]
        else:
            return self.parent.previous().children[-1]

    cpdef TreeItem first(self):
        if self.children:
            return self.children[0].first()
        else:
            return self

    cpdef TreeItem last(self):
        if self.children:
            return self.children[-1].last()
        else:
            return self

    property size:
        def __get__(self):
            cdef int s = 0
            cdef TreeItem child
            if self.children:
                for child in self.children:
                    s += child.size
                return s
            else:
                return 1

    property name:
        def __get__(self):
            if self.order <= 1:
                return self.label or ""
            if self.parent.type == "file":
                return self.parent.name + ":" + self.label
            elif self.parent.type == "group":
                return self.parent.name + "/" + self.label
            else:
                return self.parent.name + os.sep + self.label
