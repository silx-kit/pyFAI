# coding: utf-8
# /*##########################################################################
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
# ############################################################################*/
"""Context shared through all the application"""

__authors__ = ["V. Valls"]
__license__ = "MIT"
__date__ = "10/05/2019"

import weakref
import logging
import functools
import os

from silx.gui import qt

from ..utils import stringutil


_logger = logging.getLogger(__name__)


class ApplicationContext(object):

    __instance = None

    @staticmethod
    def _releaseSingleton():
        ApplicationContext.__instance = None

    @staticmethod
    def instance():
        """
        :rtype: CalibrationContext
        """
        assert(ApplicationContext.__instance is not None)
        return ApplicationContext.__instance

    def __init__(self, settings=None):
        assert(ApplicationContext.__instance is None)
        self.__parent = None
        self.__dialogStates = {}
        self.__dialogGeometry = {}
        self.__settings = settings
        ApplicationContext.__instance = self

    def saveSettings(self):
        """Save the settings of all the application"""
        # Synchronize the file storage
        self.__settings.sync()

    def restoreWindowLocationSettings(self, groupName, window):
        """Restore the window settings using this settings object

        :param qt.QSettings settings: Initialized settings
        """
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return

        settings.beginGroup(groupName)
        size = settings.value("size", qt.QSize())
        pos = settings.value("pos", qt.QPoint())
        isFullScreen = settings.value("full-screen", False)
        try:
            if not isinstance(isFullScreen, bool):
                isFullScreen = stringutil.to_bool(isFullScreen)
        except ValueError:
            isFullScreen = False
        settings.endGroup()

        if not pos.isNull():
            window.move(pos)
        if not size.isNull():
            window.resize(size)
        if isFullScreen:
            window.showFullScreen()

    def saveWindowLocationSettings(self, groupName, window):
        """Save the window settings to this settings object

        :param qt.QSettings settings: Initialized settings
        """
        settings = self.__settings
        if settings is None:
            _logger.debug("Settings not set")
            return

        isFullScreen = bool(window.windowState() & qt.Qt.WindowFullScreen)
        if isFullScreen:
            # show in normal to catch the normal geometry
            window.showNormal()

        settings.beginGroup(groupName)
        settings.setValue("size", window.size())
        settings.setValue("pos", window.pos())
        settings.setValue("full-screen", isFullScreen)
        settings.endGroup()

        if isFullScreen:
            window.showFullScreen()

    def setParent(self, parent):
        self.__parent = weakref.ref(parent)

    def parent(self):
        if self.__parent is None:
            return None
        return self.__parent()

    def __configureDialog(self, dialog):
        dialogState = self.__dialogStates.get(type(dialog), None)
        if dialogState is None:
            currentDirectory = os.getcwd()
            dialog.setDirectory(currentDirectory)
        else:
            dialog.restoreState(dialogState)
        geometry = self.__dialogGeometry.get(type(dialog), None)
        if geometry is not None:
            dialog.setGeometry(geometry)

    def __saveDialogState(self, dialog):
        self.__dialogStates[type(dialog)] = dialog.saveState()
        self.__dialogGeometry[type(dialog)] = dialog.geometry()

    def createFileDialog(self, parent, previousFile=None):
        """Create a file dialog configured with a default path.

        :rtype: qt.QFileDialog
        """
        dialog = qt.QFileDialog(parent)
        dialog.finished.connect(functools.partial(self.__saveDialogState, dialog))
        self.__configureDialog(dialog)

        if previousFile is not None:
            if os.path.exists(previousFile):
                if os.path.isdir(previousFile):
                    directory = previousFile
                else:
                    directory = os.path.dirname(previousFile)
                dialog.setDirectory(directory)

        return dialog

    def createImageFileDialog(self, parent, previousFile=None):
        """Create an image file dialog configured with a default path.

        :rtype: silx.gui.dialog.ImageFileDialog.ImageFileDialog
        """
        from silx.gui.dialog.ImageFileDialog import ImageFileDialog
        dialog = ImageFileDialog(parent)
        dialog.finished.connect(functools.partial(self.__saveDialogState, dialog))

        if hasattr(self, "getRawColormap"):
            colormap = self.getRawColormap()
            colormap = colormap.copy()
            colormap.setVRange(None, None)
            dialog.setColormap(colormap)

        self.__configureDialog(dialog)

        if previousFile is not None:
            dialog.selectUrl(previousFile)

        return dialog
