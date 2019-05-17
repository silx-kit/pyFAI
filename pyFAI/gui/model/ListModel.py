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

import functools

from silx.gui import qt
from .AbstractModel import AbstractModel


class ChangeEvent(object):

    def __init__(self, index, item, added=False, removed=False, updated=False):
        """
        Define a change done on an item from the :class:`ListModel`.

        :param int index: The location where to put/remove the item (before the
            change) or the current index of the changed item
        :param object item: The item involved in this change
        :param bool updated: True if the item was changed
        :param bool added: True if the item was added
        :param bool removed: True if the item was removed
        """
        self.index = index
        self.item = item
        assert(updated + removed + added == 1)
        self.added = added
        self.removed = removed
        self.updated = updated


class ChangeListEvent(object):
    """A container of consecutive change events"""

    def __init__(self):
        self.__events = []
        self.__structural = 0
        self.__updates = 0

    def _append(self, event):
        """Append a new event to the list.

        :param ChangeEvent event: A new event
        """
        self.__events.append(event)
        if event.added or event.removed:
            self.__structural += 1
        if event.updated:
            self.__updates += 1

    def __len__(self):
        """
        Returns the number of atomic change events applied by this event

        :rtype: int
        """
        return len(self.__events)

    def __iter__(self):
        """
        Iterates throug of the change events.

        :rtype: Iterator[ChangeEvent]
        """
        for event in self.__events:
            yield event

    def __getitem__(self, key):
        """
        Returns an event at a location

        :rtype: ChangeEvent
        """
        return self.__events[key]

    def hasStructuralEvents(self):
        """True if a structural change (`added`, `removed`) is part of the changes

        :rtype: bool
        """
        return self.__structural > 0

    def hasOnlyStructuralEvents(self):
        """True if only structural change (`added`, `removed`) is part of the changes

        :rtype: bool
        """
        return self.__structural > 0 and self.__updates == 0

    def hasUpdateEvents(self):
        """True if an update change (`updated`) is part of the changes

        :rtype: bool
        """
        return self.__updates > 0

    def hasOnlyUpdateEvents(self):
        """True if only updates events (`updated`) is part of the changes

        :rtype: bool
        """
        return self.__structural == 0 and self.__updates > 0


class ListModel(AbstractModel):
    """
    List of `AbstractModel` managing signals when items are eadited, added and
    removed.

    Atomic events for each add/remove of items. To manage it in a better way,
    `structureAboutToChange` and `structureChanged`, in order to compute all
    the atomic events in a single time.

    :param parent: Owner of this model
    """

    changed = qt.Signal([], [ChangeListEvent])
    """Emitted at the end of a structural change."""

    structureChanged = qt.Signal()
    """Emitted at the end of a structural change."""

    contentChanged = qt.Signal()
    """Emitted when the content of the elements changed."""

    def __init__(self, parent=None):
        super(ListModel, self).__init__(parent)
        self.__cacheStructureEvent = None
        self.__cacheContentWasChanged = False
        self.__items = []

    def isValid(self):
        for item, _callback in self.__items:
            if not item.isValid():
                return
        return True

    def __len__(self):
        return len(self.__items)

    def __iter__(self):
        for item, _callback in self.__items:
            yield item

    def __getitem__(self, key):
        """
        Returns an item from it's index.
        """
        return self.__items[key][0]

    def index(self, item):
        """
        Returns the index of the item in the list structure
        """
        for i, (curentItem, _callback) in enumerate(self.__items):
            if item is curentItem:
                return i
        raise IndexError("Item %s is not in list" % item)

    def clear(self):
        """Remove all the items from the list."""
        self.lockSignals()
        # TODO: Could be improved
        for item, _callback in list(self.__items):
            self.remove(item)
        self.unlockSignals()

    def append(self, item):
        """Add a new item to the end of the list."""
        assert(isinstance(item, AbstractModel))
        index = len(self.__items)
        callback = functools.partial(self.__contentWasChanged, item)
        item.changed.connect(callback)
        self.__items.append((item, callback))
        event = ChangeEvent(index=index, item=item, added=True)
        self.__fireStructureChange(event)

    def remove(self, item):
        """Remove an item."""
        assert(isinstance(item, AbstractModel))
        index = self.index(item)
        callback = self.__items[index][1]
        del self.__items[index]
        item.changed.disconnect(callback)
        event = ChangeEvent(index=index, item=item, removed=True)
        self.__fireStructureChange(event)

    def __fireStructureChange(self, event):
        emitted = self.wasChanged()
        if emitted:
            self.structureChanged.emit()
            changeListEvent = ChangeListEvent()
            changeListEvent._append(event)
            self.changed[ChangeListEvent].emit(changeListEvent)
        else:
            if self.__cacheStructureEvent is None:
                self.__cacheStructureEvent = ChangeListEvent()
            self.__cacheStructureEvent._append(event)

    def __contentWasChanged(self, item):
        emitted = self.wasChanged()
        index = self.index(item)
        event = ChangeEvent(index=index, item=item, updated=True)
        if emitted:
            self.contentChanged.emit()
            changeListEvent = ChangeListEvent()
            changeListEvent._append(event)
            self.changed[ChangeListEvent].emit(changeListEvent)
        else:
            self.__cacheContentWasChanged = True
            if self.__cacheStructureEvent is None:
                self.__cacheStructureEvent = ChangeListEvent()
            self.__cacheStructureEvent._append(event)

    def unlockSignals(self):
        unlocked = AbstractModel.unlockSignals(self)
        if unlocked:
            if self.__cacheStructureEvent is not None:
                self.changed[ChangeListEvent].emit(self.__cacheStructureEvent)
                if self.__cacheStructureEvent.hasStructuralEvents():
                    self.structureChanged.emit()
                self.__cacheStructureEvent = None
            if self.__cacheContentWasChanged:
                self.contentChanged.emit()
                self.__cacheContentWasChanged = False
        return unlocked
