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
__date__ = "19/03/2019"

import logging
import weakref

from silx.gui import qt
from silx.gui import icons


_logger = logging.getLogger(__name__)


class ChoiceToolButton(qt.QToolButton):
    """ToolButton providing a set of actions to select.

    The action is only triggered when the button is clicked.

    The method :meth:`addDefaultAction` is used to define new actions that can
    be selected.
    """

    def __init__(self, parent=None):
        qt.QToolButton.__init__(self, parent=parent)
        self.__isWaiting = False
        self.__waitingIcon = None
        self.__defaultAction = None
        self.clicked.connect(self.__clicked)

    def __clicked(self):
        default = self.defaultAction()
        if default is not None:
            default.trigger()

    def __updateWaitingIcon(self, icon):
        self.setIcon(icon)

    def __updateIcon(self):
        if self.__isWaiting:
            self.setIcon(self.__waitingIcon.currentIcon())
        else:
            default = self.defaultAction()
            if default is not None:
                self.setIcon(default.icon())

    def setWaiting(self, isWaiting):
        """Enable a waiting state.

        :param bool isWaiting: If true switch the widget to waiting state
        """
        if self.__isWaiting == isWaiting:
            return
        self.__isWaiting = isWaiting
        if isWaiting:
            self.__waitingIcon = icons.getWaitIcon()
            self.__waitingIcon.register(self)
            self.__waitingIcon.iconChanged.connect(self.__updateWaitingIcon)
        else:
            self.__waitingIcon.iconChanged.disconnect(self.__updateWaitingIcon)
            self.__waitingIcon.unregister(self)
            self.__waitingIcon = None
        self.__updateIcon()

    def defaultAction(self):
        """Returns the default selected action.
        """
        return self.__defaultAction

    def setDefaultAction(self, action):
        """Set the default action.

        Reimplement the default behaviour to avoid to add this action to the
        list of available actions.
        """
        if self.__defaultAction is action:
            return
        self.__defaultAction = action
        if not self.__isWaiting:
            self.setIcon(action.icon())
        self.setText(action.text())
        self.setToolTip(action.toolTip())

    def __selectActionTriggered(self, selectActionRef):
        selectAction = selectActionRef()
        if selectAction is None:
            return
        action = selectAction._action()
        if action is None:
            return
        self.setDefaultAction(action)

    def addDefaultAction(self, action):
        """Add an action that can be selected to set the default action
        displayed by the tool button.

        :param qt.QAction action: An action to execute when selected on the
            menu and then clicked.
        :rtype: qt.QAction
        :returns: An action triggered when the provided action is selected.
        """
        default = self.defaultAction()
        if default is None:
            self.setDefaultAction(action)
        selectAction = qt.QAction(self)
        selectAction.setIcon(action.icon())
        selectAction.setToolTip(action.toolTip())
        selectAction.setText(action.text())

        action.setVisible(False)

        selectAction._action = weakref.ref(action)
        selectActionRef = weakref.ref(selectAction)

        selectAction.triggered.connect(lambda: self.__selectActionTriggered(selectActionRef))
        self.addAction(selectAction)
        return selectAction
