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
__date__ = "03/08/2018"

from silx.gui import qt


class ProxyAction(qt.QAction):
    """Create a QAction synchronized with a source action.

    This allow to intercept all the gettes and setters by inheritance.
    """

    def __init__(self, parent, source):
        super(ProxyAction, self).__init__(parent)
        self.__source = source
        self.__source.changed.connect(self.__sourceChanged)
        self.toggled.connect(self.__actionToggled)
        self.triggered.connect(self.__actionTriggered)
        self.hovered.connect(self.__actionHovered)
        self.__sourceChanged()

    def sourceAction(self):
        return self.__source

    def __sourceChanged(self):
        self.setCheckable(self.__source.isCheckable())
        self.setEnabled(self.__source.isEnabled())
        self.setFont(self.__source.font())
        self.setIcon(self.__source.icon())
        self.setIconText(self.__source.iconText())
        self.setIconVisibleInMenu(self.__source.isIconVisibleInMenu())
        self.setMenuRole(self.__source.menuRole())
        self.setShortcut(self.__source.shortcut())
        self.setShortcutContext(self.__source.shortcutContext())
        self.setShortcutVisibleInContextMenu(self.__source.isShortcutVisibleInContextMenu())
        self.setStatusTip(self.__source.statusTip())
        self.setText(self.__source.text())
        self.setToolTip(self.__source.toolTip())
        self.setVisible(self.__source.isVisible())
        self.setWhatsThis(self.__source.whatsThis())

    def __actionToggled(self):
        self.__source.toggled.emit()

    def __actionTriggered(self):
        self.__source.triggered.emit()

    def __actionHovered(self):
        self.__source.hovered.emit()

    def hover(self):
        self.__source.hover()

    def toggle(self):
        self.__source.toggle()

    def trigger(self):
        self.__source.trigger()
