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

__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"
__license__ = "MIT"
__date__ = "13/08/2018"

from silx.gui import qt


class ElidedLabel(qt.QLabel):
    """QLabel with an edile property.

    If the value is elided, the full content is displayed as tool tip.
    """

    def __init__(self, parent):
        super(qt.QLabel, self).__init__(parent)
        self.__text = ""
        self.__toolTip = ""
        self.__valueAsToolTip = False
        self.__elideMode = qt.Qt.ElideRight
        self.__updateMinimumSize()

    def resizeEvent(self, event):
        self.__updateText()
        return qt.QLabel.resizeEvent(self, event)

    def setFont(self, font):
        qt.QLabel.setFont(self, font)
        self.__updateMinimumSize()
        self.__updateText()

    def __updateMinimumSize(self):
        metrics = qt.QFontMetrics(self.font())
        width = metrics.width("...")
        self.setMinimumWidth(width)

    def __updateText(self):
        metrics = qt.QFontMetrics(self.font())
        elidedText = metrics.elidedText(self.__text, self.__elideMode, self.width())
        qt.QLabel.setText(self, elidedText)
        self.__setValueAsTooltip(elidedText != self.__text)

    def __updateTooltip(self):
        if self.__valueAsToolTip:
            qt.QLabel.setToolTip(self, self.__text + "<br/>" + self.__toolTip)
        else:
            qt.QLabel.setToolTip(self, self.__toolTip)

    def __setValueAsTooltip(self, enable):
        if self.__valueAsToolTip == enable:
            return
        self.__valueAsToolTip = enable
        self.__updateTooltip()

    # Properties

    def setText(self, text):
        self.__text = text
        self.__updateText()

    def getText(self):
        return self.__text

    text = qt.pyqtProperty(str, getText, setText)

    def setToolTip(self, toolTip):
        self.__toolTip = toolTip
        self.__updateTooltip()

    def getToolTip(self):
        return self.__toolTip

    toolTip = qt.pyqtProperty(str, getToolTip, setToolTip)

    def setElideMode(self, elideMode):
        self.__elideMode = elideMode
        self.__updateText()

    def getElideMode(self):
        return self.__elideMode

    elideMode = qt.pyqtProperty(qt.Qt.TextElideMode, getToolTip, setToolTip)
